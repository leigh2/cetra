#!/usr/bin/env python3

import numpy as np
from .utils import *
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.autoinit
import os
import warnings
from tqdm.auto import tqdm
from time import time
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from copy import deepcopy

# read the cuda source code
cu_src_file_path = os.path.join(os.path.dirname(__file__), "cetra.cu")
with open(cu_src_file_path, "r") as cu_src_file:
    cu_src = SourceModule(cu_src_file.read())

# extract the kernels
# light curve resampling kernels
_resample_k1 = cu_src.get_function("resample_k1")
_resample_k2 = cu_src.get_function("resample_k2")
# transit search kernels
_monotransit_search = cu_src.get_function("monotransit_search")
_periodic_search_k1 = cu_src.get_function("periodic_search_k1")
_periodic_search_k2 = cu_src.get_function("periodic_search_k2")
# detrending kernels
_detrender_k1 = cu_src.get_function("detrender_k1")
_detrender_k2 = cu_src.get_function("detrender_k2")
_detrender_k3 = cu_src.get_function("detrender_k3")


class LightCurve(object):
    """
    A light curve
    """
    def __init__(self, times, fluxes, flux_errors):
        """
        Basic light curve validation and class instance initialisation.

        Parameters
        ----------
        times : array-like
            Sequence of light curve observation time points in days.
        fluxes : array-like
            Sequence of light curve relative (to baseline) flux points.
        flux_errors : array-like
            Sequence of light curve relative (to baseline) flux error points.
        """
        # all arrays must be the same length
        if len(fluxes) != len(times):
            raise RuntimeError("len(fluxes) != len(times)")
        if len(flux_errors) != len(times):
            raise RuntimeError("len(flux_errors) != len(times)")

        # make sure the input arrays contain no NaNs
        if np.any(np.isnan(times)):
            raise ValueError("one or more time values is NaN")
        if np.any(np.isnan(flux_errors)):
            raise ValueError("one or more flux error values is NaN")

        # NaN fluxes are allowed - but they must have infinite error
        if np.any(np.logical_and(np.isnan(fluxes), ~np.isinf(flux_errors))):
            raise ValueError("One or more flux values with finite error is NaN. "
                             "NaN fluxes are allowed, but they must have infinite "
                             "error.")

        # make sure the fluxes and errors are valid
        # zero flux is valid, zero flux error is not
        if np.any(fluxes < 0.0):
            raise ValueError("one or more flux values is negative")
        if np.any(flux_errors <= 0.0):
            raise ValueError("one or more flux error values is zero or negative")

        # make sure that the duration of the light curve is positive
        self.epoch_baseline = np.ptp(times)
        if self.epoch_baseline <= 0:
            raise RuntimeError("The light curve duration is invalid")

        # warn if the flux doesn't appear to have been normalised
        mean_flux = np.mean(fluxes)
        if mean_flux < 0.99 or mean_flux > 1.01:
            warnings.warn(
                "The mean flux is far from 1.0, has it been normalised by the baseline flux?",
                UserWarning
            )

        # chronological sort, store as instance variables
        srt_idx = np.argsort(times)
        self.time = times[srt_idx]
        self.relative_flux = fluxes[srt_idx]
        self.relative_flux_error = flux_errors[srt_idx]

        # provide an array of times starting at zero
        self.reference_time = np.min(times)
        self.offset_time = self.time - self.reference_time

        # flux errors to weights
        self.flux_weight = 1.0 / self.relative_flux_error ** 2

        # to save a lot of 1-f operations later, lets do it now
        self.offset_flux = 1.0 - self.relative_flux
        self.offset_flux_error = self.relative_flux_error

        # log-likelihood of constant flux model
        self.flat_loglike = np.nansum(
            - 0.5 * self.offset_flux ** 2 / self.offset_flux_error**2
            - 0.5 * np.log(2 * np.pi * self.offset_flux_error**2)
        )

        # get the number of elements
        self.num_points = len(self.time)

        # attempt to determine the cadence
        # delta t in milliseconds
        ms_per_day = Constants.seconds_per_day * 1000
        _dt = np.round(np.diff(self.time * ms_per_day)).astype(np.int64)
        uq, ct = np.unique(_dt, return_counts=True)
        if ct.max() > 1:
            # modal value if possible
            modal_dt = uq[np.argmax(ct)]
            self.cadence = modal_dt / ms_per_day
        else:
            # median otherwise
            self.cadence = np.nanmedian(_dt) / ms_per_day
        self.cadence_range = _dt.min() / ms_per_day, _dt.max() / ms_per_day

    def __len__(self):
        return self.num_points

    def copy(self):
        return deepcopy(self)

    def mask_transit(self, transit):
        """
        Mask a transit (single or periodic)

        Parameters
        ----------
        transit : Transit
            The transit to mask

        Returns
        -------
        LightCurve instance with the transit removed, np.ndarray that is `True`
        while in-transit and `False` otherwise.
        """
        # todo maybe allow some user-specific border as fraction of the duration

        # are we periodic?
        periodic = transit.period is not None

        # compute the phase
        phase = self.time - transit.t0
        if periodic:
            phase %= transit.period

        # generate the mask
        itr_mask = (phase > 0) & (phase < transit.duration)

        # generate a new LightCurve
        LC_new = LightCurve(
            times=self.time[~itr_mask],
            fluxes=self.relative_flux[~itr_mask],
            flux_errors=self.relative_flux_error[~itr_mask]
        )

        return LC_new, itr_mask

    def pad(self, num_points_start, num_points_end):
        """
        Pad the light curve with null data. Useful in certain
        circumstances.

        Parameters
        ----------
        num_points_start : int
            The number of points to prepend to the light curve.
        num_points_end : int
            The number of points to append to the light curve.

        Returns
        -------
        A new LightCurve instance padded with null data at the start and/or end.
        """
        # prepend
        _t = self.time[0] - np.arange(num_points_start, 0, -1) * self.cadence
        new_times = np.insert(self.time, 0, _t)
        new_rflux = np.insert(
            self.relative_flux, 0,
            np.full(num_points_start, np.nan, dtype=self.relative_flux.dtype)
        )
        new_eflux = np.insert(
            self.relative_flux_error, 0,
            np.full(num_points_start, np.inf, dtype=self.relative_flux_error.dtype)
        )

        # append
        _t = self.time[-1] + np.arange(1, num_points_end+1) * self.cadence
        new_times = np.append(new_times, _t)
        new_rflux = np.append(
            new_rflux,
            np.full(num_points_end, np.nan, dtype=self.relative_flux.dtype)
        )
        new_eflux = np.append(
            new_eflux,
            np.full(num_points_end, np.inf, dtype=self.relative_flux_error.dtype)
        )

        return LightCurve(new_times, new_rflux, new_eflux)

    def resample(self, new_cadence, cuda_blocksize=1024):
        """
        Resample the light curve at a new cadence.

        Parameters
        ----------
        new_cadence : float
            The desired output observation cadence (i.e. the time between
            samples).
        cuda_blocksize : int, optional
            The number of threads per block. Should be a multiple of 32 and
            less than or equal to 1024. Default 1024.

        Returns
        -------
        A LightCurve instance with the new sampling cadence
        """
        # input check
        if new_cadence <= 0.0 or not np.isfinite(new_cadence):
            raise ValueError("New cadence must be finite and greater than zero")

        # generate the new time sequence
        new_time = np.arange(
            start=self.time.min(),
            stop=self.time.max()+new_cadence,
            step=new_cadence
        )

        # send some arrays to the gpu
        _time = to_gpu(self.offset_time, np.float64)
        _flux = to_gpu(self.offset_flux, np.float64)
        _ferr = to_gpu(self.offset_flux_error, np.float64)
        # initialise output arrays on the gpu
        _rflux_out = gpuarray.zeros(new_time.size, dtype=np.float64)
        _err_out = gpuarray.zeros(new_time.size, dtype=np.float64)
        _sum_fw = gpuarray.zeros(new_time.size, dtype=np.float64)
        _sum_w = _err_out  # we can reuse this array if we're careful!
        # type specification
        _cadence = np.float64(new_cadence)
        _n_elem = np.int32(self.num_points)
        _n_elem_out = np.int32(new_time.size)

        # set the cuda block and grid sizes
        _blocksize = (cuda_blocksize, 1, 1)
        _gridsize1 = (int(np.ceil(_n_elem / cuda_blocksize)), 1, 1)
        _gridsize2 = (int(np.ceil(_n_elem_out / cuda_blocksize)), 1, 1)

        # run the kernel to sum the fluxes and their weights
        _resample_k1(
            _time, _cadence, _flux, _ferr, _n_elem, _sum_fw, _sum_w,
            block=_blocksize, grid=_gridsize1
        )
        # now run the division kernel
        _resample_k2(
            _sum_fw, _sum_w, _rflux_out, _err_out, _n_elem_out,
            block=_blocksize, grid=_gridsize2
        )

        return LightCurve(new_time, _rflux_out.get(), _err_out.get())


@dataclass
class Transit(object):
    """
    A transit (periodic or not)
    """
    lightcurve: LightCurve
    t0: float
    duration: float
    depth: float
    log_likelihood: float
    period: float = None
    t0_error: float = None
    duration_error: float = None
    depth_error: float = None
    period_error: float = None

    def __str__(self):
        t0_str = f"            t0: {self.t0:.5f}"
        duration_str = f"      duration: {self.duration:.5f}"
        depth_str = f"         depth: {self.depth*1e6:.1f}"
        period_str = f"        period: {self.period:.6f}" if self.period is not None else ""

        t0_str += f" ± {self.t0_error:.5f} days\n" if self.t0_error is not None else " days\n"
        duration_str += f" ± {self.duration_error:.5f} days\n" if self.duration_error is not None else " days\n"
        depth_str += f" ± {self.depth_error*1e6:.1f} ppm\n" if self.depth_error is not None else " ppm\n"
        snr_str = f"           SNR: {self.depth/self.depth_error:.1f}\n" if self.depth_error is not None else ""
        period_str += f" ± {self.period_error:.6f}" if self.period_error is not None else ""
        period_str += " days\n" if self.period is not None else ""

        return f"{t0_str}{duration_str}{depth_str}{snr_str}{period_str}"\
               f"log-likelihood: {self.log_likelihood:.4e}\n"\
               f"   lc duration: {self.lightcurve.epoch_baseline:.2f} days\n"\
               f"    lc cadence: {self.lightcurve.cadence*Constants.seconds_per_day:.0f} seconds"

    def refine_lsq(
            self,
            model: np.ndarray,
            t0_bounds: tuple,
            duration_bounds: tuple,
            depth_bounds: tuple,
            period_bounds: tuple = None
    ):
        """
        Refine the transit parameters using bounded least squares fitting

        Parameters
        ----------
        model : np.ndarray
            Numpy array of the original transit model.
        t0_bounds : tuple
            Tuple of lower and upper bounds on t0.
        duration_bounds : tuple
            Tuple of lower and upper bounds on duration.
        depth_bounds : tuple
            Tuple of lower and upper bounds on depth.
        period_bounds : tuple, optional
            Tuple of lower and upper bounds on period. Required if the transit
            is periodic.

        Returns
        -------
        A new Transit instance
        """
        raise NotImplementedError("This function needs modification after the shift to t0")

        # verify that period bounds are given if required
        if self.period is not None and period_bounds is None:
            raise RuntimeError(
                "period bounds must be given if the transit is periodic"
            )

        # interpolator using the original transit model
        _tm = interp1d(
            np.linspace(0.0, 1.0, model.size),
            1.0 - model,
            kind='linear',
            copy=True,
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True
        )

        # the model function
        def _mod_func(_time, _t0, _duration, _depth, _period=None):
            if period_bounds is None:
                # _t0, _duration, _depth = params
                _phase = _time - _t0
                _model = _tm(_phase / _duration) * _depth
            else:
                # _t0, _duration, _depth, _period = params
                _phase = (_time - _t0) % _period
                _model = _tm(_phase / _duration) * _depth

            return _model

	# set the starting point and bounds arrays
        if period_bounds is None:
            p0 = np.array([self.t0, self.duration, self.depth])
            bounds = (
                np.array([t0_bounds[0], duration_bounds[0], depth_bounds[0]]),
                np.array([t0_bounds[1], duration_bounds[1], depth_bounds[1]]),
            )
        else:
            p0 = np.array([self.t0, self.duration, self.depth, self.period])
            bounds = (
                np.array([t0_bounds[0], duration_bounds[0], depth_bounds[0], period_bounds[0]]),
                np.array([t0_bounds[1], duration_bounds[1], depth_bounds[1], period_bounds[1]]),
            )

	# run the bounded least squares optimisation
        popt, pcov = curve_fit(
            _mod_func, self.lightcurve.time, self.lightcurve.offset_flux, p0,
            sigma=self.lightcurve.offset_flux_error, absolute_sigma=False,
            bounds=bounds, x_scale='jac'
        )
        perr = np.sqrt(np.diag(pcov))

        # extract parameters
        t0 = popt[0], perr[0]
        duration = popt[1], perr[1]
        depth = popt[2], perr[2]
        if self.period is not None:
            period = popt[3], perr[3]
        else:
            period = None, None

        # log-likelihood of refined model
        model = _mod_func(self.lightcurve.time, *popt)
        log_likelihood = np.nansum(
            - 0.5 * (model - self.lightcurve.offset_flux) ** 2 / self.lightcurve.offset_flux_error**2
            - 0.5 * np.log(2 * np.pi * self.lightcurve.offset_flux_error**2)
        )

        return Transit(
            lightcurve=self.lightcurve,
            t0=t0[0],
            duration=duration[0],
            depth=depth[0],
            log_likelihood=log_likelihood,
            period=period[0],
            t0_error=t0[1],
            duration_error=duration[1],
            depth_error=depth[1],
            period_error=period[1]
        )


@dataclass
class MonotransitResult(object):
    """
    A monotransit search result
    """
    raw_lightcurve: LightCurve
    resampled_lightcurve: LightCurve
    duration_array: np.ndarray
    t0_array: np.ndarray
    loglike_constant: float
    likeratio_array: np.ndarray
    depth_array: np.ndarray
    depth_variance_array: np.ndarray
    input_model: np.ndarray

    def get_max_likelihood(self) -> float:
        """
        Return the maximum likelihood value

        Returns
        -------
        The maximum likelihood value
        """
        return self.loglike_constant + np.nanmax(self.likeratio_array)

    def get_BIC(self) -> float:
        """
        Return the Bayesian information criterion

        Returns
        -------
        The Bayesian information criterion
        """
        k_monotransit = 4  # t0, depth, duration, error variance
        lnn = np.log(self.raw_lightcurve.num_points)
        return k_monotransit * lnn - 2 * self.get_max_likelihood()

    def get_max_likelihood_parameters(self, lsq_refine=False):
        """
        Return the parameters of the maximum likelihood transit

        Parameters
        ----------
        lsq_refine : bool, optional
            If `True` refines the transit parameters using least squares.
            Default `False`.

        Returns
        -------
        A single Transit object with the maximum likelihood where
        `lsq_refine=False`. Otherwise it returns two Transit objects. The
        first the maximum likelihood based on the grid results, the second
        from an lsq refinement of the first.
        """
        d, t = np.unravel_index(
            np.nanargmax(self.likeratio_array), self.likeratio_array.shape
        )
        return self.get_params(d, t, lsq_refine=lsq_refine)

    def get_max_snr_parameters(self, absolute_depth=False, lsq_refine=False):
        """
        Return the parameters of the maximum SNR transit

        Parameters
        ----------
        absolute_depth : bool, optional
            If `True`, computes SNR as |S/N|, otherwise SNR is S/N.
            `False` by default.
        lsq_refine : bool, optional
            If `True` refines the transit parameters using least squares.
            Default `False`.

        Returns
        -------
        A single Transit object with the largest SNR where `lsq_refine=False`.
        Otherwise it returns two Transit objects. The first the largest SNR
        based on the grid results, the second from an lsq refinement of the
        first.
        """
        snr_array = self.depth_array / np.sqrt(self.depth_variance_array)
        if absolute_depth:
            snr_array = np.abs(snr_array)
        d, t = np.unravel_index(
            np.nanargmax(snr_array), snr_array.shape
        )
        return self.get_params(d, t, lsq_refine=lsq_refine)

    def get_params(self, duration_index: int, t0_index: int, lsq_refine: bool = False):
        """
        Find the parameters of the transit given duration and t0 indices

        Parameters
        ----------
        duration_index : int
            Duration index
        t0_index : int
            t0 index
        lsq_refine : bool, optional
            If `True` refines the transit parameters using least squares.
            Default `False`.

        Returns
        -------
        A single Transit object where `lsq_refine=False`. Otherwise it returns
        two Transit objects. The first based on the grid results, the second
        from the lsq refinement.
        """
        # find or compute the requested parameters
        t0 = self.t0_array[t0_index] + self.resampled_lightcurve.reference_time
        duration = self.duration_array[duration_index]
        depth = self.depth_array[duration_index, t0_index]
        depth_error = np.sqrt(self.depth_variance_array[duration_index, t0_index])
        loglike = self.loglike_constant + self.likeratio_array[duration_index, t0_index]

        # generate the grid search Transit object
        grid_transit = Transit(
            lightcurve=self.raw_lightcurve,
            t0=t0,
            duration=duration,
            depth=depth,
            depth_error=depth_error,
            log_likelihood=loglike
        )

        if not lsq_refine:
            # return the grid search Transit object
            return grid_transit
        else:
            # lsq bounds on parameters
            # t0
            t0_bounds = get_bounds(self.t0_array, t0_index, t0)
            # duration
            duration_bounds = get_bounds(self.duration_array, duration_index, duration)
            # depth (5 sigma is more than enough)
            depth_bounds = (
                depth - 5 * depth_error,
                depth + 5 * depth_error
            )
            # run the lsq refinement
            lsq_transit = grid_transit.refine_lsq(
                model=self.input_model,
                t0_bounds=t0_bounds,
                duration_bounds=duration_bounds,
                depth_bounds=depth_bounds
            )
            # return both Transit objects
            return grid_transit, lsq_transit


@dataclass
class PeriodogramResult(object):
    """
    A periodic search result
    """
    raw_lightcurve: LightCurve
    resampled_lightcurve: LightCurve
    duration_array: np.ndarray
    t0_array: np.ndarray
    period_array: np.ndarray
    loglike_constant: float
    likeratio_array: np.ndarray
    depth_array: np.ndarray
    depth_variance_array: np.ndarray
    duration_index_array: np.ndarray
    t0_index_array: np.ndarray
    monotransit_result: MonotransitResult
    input_model: np.ndarray

    def get_log_likelihoods(self) -> np.ndarray:
        """
        Return the array of log-likelihoods for the periods

        Returns
        -------
        An array of log-likelihoods
        """
        return self.loglike_constant + self.likeratio_array

    def get_BIC(self) -> float:
        """
        Return the Bayesian information criteria for the array of periods

        Returns
        -------
        Array of the Bayesian information criteria
        """
        k_periodic = 5  # period, t0, depth, duration, error variance
        lnn = np.log(self.raw_lightcurve.num_points)
        return k_periodic * lnn - 2 * self.get_log_likelihoods()

    def get_max_likelihood_parameters(self, lsq_refine=False):
        """
        Return the parameters of the maximum likelihood period

        Parameters
        ----------
        lsq_refine : bool, optional
            If `True` refines the transit parameters using least squares.
            Default `False`.

        Returns
        -------
        A single Transit object with the maximum likelihood where
        `lsq_refine=False`. Otherwise it returns two Transit objects. The
        first the maximum likelihood based on the grid results, the second
        from an lsq refinement of the first.
        """
        idx = np.nanargmax(self.likeratio_array)
        return self.get_params(idx, lsq_refine=lsq_refine)

    def get_max_snr_parameters(self, absolute_depth=False, lsq_refine=False):
        """
        Return the parameters of the maximum SNR period

        Parameters
        ----------
        absolute_depth : bool, optional
            If `True`, computes SNR as |S/N|, otherwise SNR is S/N.
            `False` by default.
        lsq_refine : bool, optional
            If `True` refines the transit parameters using least squares.
            Default `False`.

        Returns
        -------
        A single Transit object with the largest SNR where `lsq_refine=False`.
        Otherwise it returns two Transit objects. The first the largest SNR
        based on the grid results, the second from an lsq refinement of the
        first.
        """
        snr_array = self.depth_array / np.sqrt(self.depth_variance_array)
        if absolute_depth:
            snr_array = np.abs(snr_array)
        idx = np.nanargmax(snr_array)
        return self.get_params(idx, lsq_refine=lsq_refine)

    def get_params(self, period_index: int, lsq_refine: bool = False):
        """
        Find the parameters of the maximum likelihood transit given a period
        index

        Parameters
        ----------
        period_index : int
            The period index
        lsq_refine : bool, optional
            If `True` refines the transit parameters using least squares.
            Default `False`.

        Returns
        -------
        A single Transit object where `lsq_refine=False`. Otherwise it returns
        two Transit objects. The first based on the grid results, the second
        from the lsq refinement.

        Returns
        -------
        A Transit object
        """
        t0_index = self.t0_index_array[period_index]
        duration_index = self.duration_index_array[period_index]
        # find or compute the requested parameters
        period = self.period_array[period_index]
        t0 = self.t0_array[t0_index] + self.resampled_lightcurve.reference_time
        duration = self.duration_array[duration_index]
        depth = self.depth_array[period_index]
        depth_error = np.sqrt(self.depth_variance_array[period_index])
        loglike = self.loglike_constant + self.likeratio_array[period_index]

        # make sure that t0 is the first transit
        while (t0 - period) > self.raw_lightcurve.time.min():
            t0 -= period

        # generate the grid search Transit object
        grid_transit = Transit(
            lightcurve=self.raw_lightcurve,
            t0=t0,
            duration=duration,
            depth=depth,
            depth_error=depth_error,
            log_likelihood=loglike,
            period=period
        )

        if not lsq_refine:
            # return the grid search Transit object
            return grid_transit
        else:
            # lsq bounds on parameters
            # period
            period_bounds = get_bounds(self.period_array, period_index, period)
            # t0
            t0_bounds = get_bounds(
                self.t0_array + self.resampled_lightcurve.reference_time, t0_index, t0
            )
            # duration
            duration_bounds = get_bounds(self.duration_array, duration_index, duration)
            # depth (5 sigma is more than enough)
            depth_bounds = (
                depth - 5 * depth_error,
                depth + 5 * depth_error
            )
            # run the lsq refinement
            lsq_transit = grid_transit.refine_lsq(
                model=self.input_model,
                t0_bounds=t0_bounds,
                duration_bounds=duration_bounds,
                depth_bounds=depth_bounds,
                period_bounds=period_bounds
            )
            return grid_transit, lsq_transit


class TransitDetector(object):
    """
    A tool for identifying transit-like signals in stellar light curves
    """

    def __init__(
            self, times, flux, flux_error, durations=None,
            min_duration=0.02, max_duration=1.0, duration_log_step=1.1,
            t0_stride_fraction=None, t0_stride_length=None,
            resample_cadence=None,
            transit_model=None, transit_model_size=1024,
            verbose=True
    ):
        """
        Initialise the transit detector.

        Parameters
        ----------
        times : ndarray
            Sequence of light curve observation time points in days.
        flux : ndarray
            Sequence of light curve relative (to baseline) flux points.
        flux_error : ndarray
            Sequence of light curve relative (to baseline) flux error points.
        durations : array-like, optional
            User-specified duration grid. If not provided, the module computes
            a grid using the minimum and maximum durations and the log step.
        min_duration : float, optional
            Minimum transit duration to check in days, default 0.02.
        max_duration : float, optional
            Maximum transit duration to check in days, default 1.0.
        duration_log_step : float, optional
            The log-spacing of the durations to be used if the duration grid
            is to be internally determined. Default 1.1.
        t0_stride_fraction : float, optional
            The fraction of the minimum duration that determines the length of
            each t0 stride. The default is 1% of the minimum duration. This
            argument should be left unset if a specific t0 stride length is
            given using the `t0_stride_length` argument.
        t0_stride_length : float, optional
            The t0 stride length (in seconds) for the monotransit search. The
            default is to use 1% of the minimum duration (i.e.
            `t0_stride_fraction = 0.01`). This argument should be left unset if
            `t0_stride_fraction` is set.
        resample_cadence : float, optional
            The cadence (in seconds) to use for resampling the light curve. By
            default, the module will try to detect the underlying cadence.
        transit_model : array-like or string, optional
            This model will be used, if provided, instead of the default.
            The model should be an offset relative to baseline flux.
            Maximum deviation should be 1 in order for the output depths to be
            immediately meaningful. The module does not force transit depth to
            be positive. (So e.g. a flare model can be provided and the depths
            returned will be negative.)
            The default model is for a transit with the following parameters:
                'rp': 0.03,
                'b': 0.32,
                'u': (0.4804, 0.1867),
                'period': 10.0,
                'semimajor_axis': 20.0.
            Alternatively, one of the strings 'b32', 'b93' or 'b99' can be
            supplied. In this case, a model with impact parameter 0.32 (the
            default, above), 0.93 or 0.99, respectively, will be used. All
            other parameters are the same as for the default model.
        transit_model_size : int, optional
            The transit model size, ideally a power of 2. Default 1024.
            The larger the value the smaller the max error in the model.
            A value of 1024 gives a maximum error of ~1%, 2048 is ~0.5%,
            512 is ~2%.
        verbose : bool, optional
            If True (the default), reports various messages.
        """
        # validate the input light curve
        self.lc_in = LightCurve(times, flux, flux_error)

        if verbose:
            # report the input light curve cadence
            print(f"input light curve has {self.lc_in.num_points} elements\n"
                  f"cadence: {self.lc_in.cadence * Constants.seconds_per_day:.0f}s "
                  f"(range: {self.lc_in.cadence_range[0] * Constants.seconds_per_day:.0f}s -> "
                  f"{self.lc_in.cadence_range[1] * Constants.seconds_per_day:.0f}s)\n"
                  f"constant flux model log-likelihood: {self.lc_in.flat_loglike:.3e}")

        # resample the new light curve to regularise the cadence
        # a regularised cadence is necessary so that we can cheaply determine
        # the array location of a given point in time
        # resampled points containing no observed data have null fluxes and
        # infinite flux errors
        if resample_cadence is None:
            self.lc = self.lc_in.resample(self.lc_in.cadence)
        else:
            self.lc = self.lc_in.resample(resample_cadence / Constants.seconds_per_day)

        if verbose:
            # report the resampled light curve cadence
            print(f"resampled light curve has {self.lc.num_points} elements\n"
                  f"cadence: {self.lc.cadence * Constants.seconds_per_day:.0f}s "
                  f"(range: {self.lc.cadence_range[0] * Constants.seconds_per_day:.0f}s -> "
                  f"{self.lc.cadence_range[1] * Constants.seconds_per_day:.0f}s)")

        if durations is not None:
            # use the provided duration grid
            self.durations = np.asarray(durations)
        else:
            # generate the duration grid
            self.durations = duration_grid(
                min_duration=min_duration,
                max_duration=max_duration,
                log_step=duration_log_step
            )
        # how many durations are there?
        self.duration_count = len(self.durations)

        # do nothing if the duration grid is too sparse
        if self.duration_count == 0:
            warnings.warn("The duration grid is empty")
            return

        if verbose:
            # report the duration grid size
            print(f"{self.duration_count} durations, "
                  f"{self.durations[0]:.2f} -> {self.durations[-1]:.2f} days")

        # Pre-pad the light curve with null data to make simpler the algorithm
        # that searches for transits that begin before the data. This requires
        # a regular observing cadence, another benefit of our earlier
        # resampling operation.
        num_prepad = int(np.ceil(0.5 * np.max(self.durations) / self.lc.cadence))
        self.lc = self.lc.pad(num_prepad, 0)
        if verbose:
            print(f"prepended {num_prepad} null points to the light curve")

        if isinstance(transit_model, list) or isinstance(transit_model, np.ndarray):
            if verbose:
                print("user-provided transit model")
            self.input_model = np.asarray(transit_model)
            # todo some checks of user-input models?
        else:
            if transit_model is None:
                transit_model = "b32"
            if transit_model in ['b32', 'b93', 'b99']:
                _impact_params = {'b32': 0.32, 'b93': 0.93, 'b99': 0.99}
                if verbose:
                    print(f"using default transit model with impact parameter: "
                          f"{_impact_params[transit_model]}")
                tmod_file_path = os.path.join(os.path.dirname(__file__),
                                              f"transit_model_{transit_model}.npz")
                tmod = np.load(tmod_file_path)
                self.input_model = tmod["model_array"]
            else:
                raise RuntimeError(
                    "transit_model not recognised. "
                    "Is it an array, or one of 'b32', 'b93' or 'b99'?"
                )

        # generate the transit model
        self.transit_model = interpolate_model(
            self.input_model, transit_model_size
        )
        # store the transit model size as an instance variable
        self.transit_model_size = int(transit_model_size)

        # to save a lot of 1-f transit model operations later, lets do it now
        self.offset_transit_model = 1.0 - self.transit_model
        # send the offset transit model to the gpu
        self.offset_transit_model_gpu = to_gpu(self.offset_transit_model, np.float32)

        if verbose:
            # report the transit model size
            print(f"transit model size: {self.transit_model_size} elements")
            # report the nearest neighbour error
            _nn_error = nn_model_error(self.transit_model_size, self.input_model)
            print(f"maximum nearest-neighbour error: {100 * _nn_error[0]:.2e}%")
            print(f"   mean nearest-neighbour error: {100 * _nn_error[1]:.2e}%")

        # determine the t0 stride length
        if t0_stride_length is None and t0_stride_fraction is None:
            self.t0_stride_length = np.min(self.durations) * 0.01
        elif t0_stride_fraction is None and t0_stride_length is not None:
            self.t0_stride_length = t0_stride_length / Constants.seconds_per_day
        elif t0_stride_fraction is not None and t0_stride_length is None:
            self.t0_stride_length = np.min(self.durations) * t0_stride_fraction
        elif t0_stride_fraction is not None and t0_stride_length is not None:
            warnings.warn(
                "Both t0_stride_fraction and t0_stride_length are set, using "
                "the smaller of the two"
            )
            _fracval = np.min(self.durations) * t0_stride_fraction
            _lenval = t0_stride_length / Constants.seconds_per_day
            self.t0_stride_length = min(_fracval, _lenval)
        if verbose:
            print(f"t0 stride length: {self.t0_stride_length * Constants.seconds_per_day:.3f} seconds")
        # generate the grid of t0s
        self.t0s = np.arange(0, self.lc.offset_time[-1], self.t0_stride_length)
        self.num_t0_strides = self.t0s.size
        if verbose:
            print(f"{self.num_t0_strides:d} t0 strides")

        # initialise instance variables that get populated later
        self.periods = None
        self.period_count = None
        self.like_ratio_2d_gpu = None
        self.depth_2d_gpu = None
        self.var_depth_2d_gpu = None
        self.offset_flux_gpu = None
        self.flux_weight_gpu = None
        self.offset_time_gpu = None
        self.durations_gpu = None
        self.monotransit_result = None
        self.periodogram_result = None

    def period_search(
            self, periods=None,
            min_period=0.0, max_period=np.inf, n_transits_min=2,
            pgrid_R_star=1.0, pgrid_M_star=1.0, pgrid_oversample=3,
            ignore_astrophysics=False, max_duration_fraction=0.12,
            min_star_mass=0.1, max_star_mass=1.0,
            min_star_radius=0.13, max_star_radius=3.5,
            random_order=True, verbose=True
    ):
        """
        Compute the periodogram.

        Parameters
        ----------
        periods : array-like, optional
            User-provided period grid. If not provided, the module determines
            an optimal period grid using the various other kwargs.
        min_period : float, optional
            The minimum period to consider. Set to 0.0 for no lower limit,
            which in practice means it's limited by the astrophysics (or
            the sampling cadence), this is the default setting.
        max_period : float, optional
            The maximum period to consider. Set to inf for no upper limit,
            which in practice means it's limited by the light curve duration
            and the minimum number of required transits, this is the default
            setting.
        n_transits_min : int, optional
            The minimum number of transits that must have coverage. This
            parameter impacts the period grid, in that the maximum period is
            limited to the epoch baseline divided by n_transits_min. The
            default requirement is that there are 2 transits.
        pgrid_R_star : float, optional
            The stellar radius (in solar radii) to use for period grid
            determination, default 1.0.
        pgrid_M_star : float, optional
            The stellar mass (in solar masses) to use for period grid
            determination, default 1.0.
        pgrid_oversample : int, optional
            Oversample the period grid by this factor, default 3. Increasing
            this improves detection efficiency but at a higher computational
            cost.
        ignore_astrophysics : bool, optional
            If `True`, the duration is only required to be less than the
            period times the max_duration_fraction. If `False`, the duration
            is also required to be astrophysically plausible (the default
            behaviour).
        max_duration_fraction : float, optional
            Maximum duration as a fraction of the period, default 0.12.
        min_star_mass : float, optional
            Minimum star mass to consider in solar masses, default 0.1.
        max_star_mass : float, optional
            Maximum star mass to consider in solar masses, default 1.0.
        min_star_radius : float, optional
            Minimum star radius to consider in solar radii, default 0.13.
        max_star_radius : float, optional
            Maximum star radius to consider in solar radii, default 3.5.
        random_order : bool, optional
            If True (the default), the period grid is shuffled before
            computation. This provides a more accurate estimated compute time
            and progress bar. If False then the periodogram is populated in
            ascending order of period.
        verbose : bool, optional
            If True (the default), provides a progress bar and estimated time
            to completion courtesy of the tqdm package. Also reports the
            parameters of the maximum likelihood and maximum SNR periods.
            Otherwise, only some basic info/warnings are reported.

        Returns
        -------
        PeriodogramResult
        """
        if self.monotransit_result is None:
            warnings.warn(
                "A periodic signal search cannot be run without first running "
                "a monotransit search"
            )
            return
        else:
            if verbose:
                print("commencing periodic signal search")

        if periods is not None:
            # use the provided period grid
            self.periods = np.asarray(periods)
        else:
            # generate the period grid
            self.periods = period_grid(
                self.lc.epoch_baseline, min_period=min_period,
                max_period=max_period, n_transits_min=n_transits_min,
                R_star=pgrid_R_star, M_star=pgrid_M_star,
                oversampling_factor=pgrid_oversample
            )
        self.period_count = len(self.periods)

        # do nothing if the period grid is too sparse
        if self.period_count == 0:
            warnings.warn("The period grid is empty")
            return

        # initialise a dictionary to record the results in
        periodogram = {
            'like_ratio': np.full(self.period_count, np.nan, dtype=np.float32),
            'depth': np.full(self.period_count, np.nan, dtype=np.float32),
            'var_depth': np.full(self.period_count, np.nan, dtype=np.float32),
            't0_idx': np.full(self.period_count, -1, dtype=np.int32),
            'duration_idx': np.full(self.period_count, -1, dtype=np.int32)
        }

        # deal with the ordering of the search
        if random_order:
            order = np.random.permutation(self.period_count)  # random
        else:
            order = np.arange(self.period_count)  # ascending

        if verbose:
            print(f"testing {self.period_count} periods from "
                  f"{self.periods.min():.2e} to "
                  f"{self.periods.max():.2e} days")

        # record the start time
        t0 = time()

        # iterate through the period grid
        for i in tqdm(range(self.period_count), total=self.period_count, disable=not verbose):
            n = order[i]
            period = float(self.periods[n])

            # check this period
            ret = self.check_period(
                period, ignore_astrophysics=ignore_astrophysics,
                max_duration_fraction=max_duration_fraction,
                min_star_mass=min_star_mass, max_star_mass=max_star_mass,
                min_star_radius=min_star_radius, max_star_radius=max_star_radius
            )

            # record the results
            periodogram['like_ratio'][n] = ret[0]
            periodogram['depth'][n] = ret[1]
            periodogram['var_depth'][n] = ret[2]
            periodogram['t0_idx'][n] = ret[3]
            periodogram['duration_idx'][n] = ret[4]

        # record the stop time and report elapsed
        t1 = time()
        if verbose:
            print(f"completed in {t1 - t0:.3f} seconds")

        # generate the periodic search result
        self.periodogram_result = PeriodogramResult(
            raw_lightcurve=self.lc_in,
            resampled_lightcurve=self.lc,
            duration_array=self.durations,
            t0_array=self.t0s,
            period_array=self.periods,
            loglike_constant=self.lc.flat_loglike,
            likeratio_array=periodogram['like_ratio'],
            depth_array=periodogram['depth'],
            depth_variance_array=periodogram['var_depth'],
            duration_index_array=periodogram['duration_idx'],
            t0_index_array=periodogram['t0_idx'],
            monotransit_result=self.monotransit_result,
            input_model=self.input_model
        )

        if verbose:
            # report the best candidates
            print(f"max likelihood periodic signal:\n"
                  f"{self.periodogram_result.get_max_likelihood_parameters()}\n"
                  f"max SNR periodic signal:\n"
                  f"{self.periodogram_result.get_max_snr_parameters()}")

        return self.periodogram_result

    def check_period(
            self, period,
            max_duration_fraction=0.12,
            min_star_mass=0.1, max_star_mass=1.0,
            min_star_radius=0.13, max_star_radius=3.5,
            ignore_astrophysics=False
    ):
        """
        Find the maximum likelihood ratio (vs. constant flux model), depth,
        depth variance, t0 and duration for a given period.

        Parameters
        ----------
        period : float
            The period to check
        max_duration_fraction : float, optional
            Maximum duration as a fraction of the period, default 0.12.
        min_star_mass : float, optional
            Minimum star mass to consider in solar masses, default 0.1.
        max_star_mass : float, optional
            Maximum star mass to consider in solar masses, default 1.0.
        min_star_radius : float, optional
            Minimum star radius to consider in solar radii, default 0.13.
        max_star_radius : float, optional
            Maximum star radius to consider in solar radii, default 3.5.
        ignore_astrophysics : bool, optional
            If `True`, the duration is only required to be less than the
            period times the max_duration_fraction. If `False`, the duration
            is also required to be astrophysically plausible (the default
            behaviour).

        Returns
        -------
        Tuple of len=5 containing the maximum likelihood ratio and the
        corresponding depth, depth variance, t0 index and duration index for
        the input period.
        """

        if not ignore_astrophysics:
            # set minimum and maximum durations as a fraction of the period
            duration_min = max_t14(
                star_radius=min_star_radius, star_mass=min_star_mass,
                period=period,
                upper_limit=max_duration_fraction, small_planet=True
            )
            duration_max = max_t14(
                star_radius=max_star_radius, star_mass=max_star_mass,
                period=period,
                upper_limit=max_duration_fraction, small_planet=False
            )
            # convert to time units
            duration_min *= period
            duration_max *= period

        else:
            # no minimum duration, max set by user
            duration_min = 0.0
            duration_max = max_duration_fraction * period

        # the duration should always be less than the period, so that transit
        # windows can never overlap
        if duration_max > period:
            warnings.warn("the max duration should not be larger than the period")
            duration_max = period

        # which durations to run
        _durations_in_range = np.where(
            (self.durations >= duration_min) & (self.durations <= duration_max)
        )[0]

        # return some null output if no valid durations
        if _durations_in_range.size == 0:
            return np.nan, np.nan, np.nan, -1, -1

        # find the indices and number of durations that are in range
        first_d_in_range_idx = np.int32(_durations_in_range[0])
        last_d_in_range_idx = np.int32(_durations_in_range[-1])
        num_d_in_range = np.int32(_durations_in_range.size)

        # maximum possible number of transits
        max_transit_count = np.int32(np.floor(self.lc.epoch_baseline / period) + 1)
        # the period in strides
        _period_in_strides = np.float32(period / self.t0_stride_length)

        # block and grid sizes
        # the second and third block dimensions should always have size 1
        block_size_k1 = 512, 1, 1
        grid_size_k1 = (
            int(np.ceil(_period_in_strides / block_size_k1[0])),
            int(num_d_in_range)
        )
        # shared memory size, space for 2x single-precision floats per thread
        # we need to do a max() reduction while also finding the thread index
        # of the maximum value
        smem_size_k1 = int(2 * 4 * block_size_k1[0])

        # temporary arrays for the reduction operation
        tmp_size = grid_size_k1[0] * grid_size_k1[1]
        # first pass output
        tmp_likerat_gpu = gpuarray.empty(tmp_size, dtype=np.float32)
        tmp_depth_gpu = gpuarray.empty(tmp_size, dtype=np.float32)
        tmp_var_depth_gpu = gpuarray.empty(tmp_size, dtype=np.float32)
        tmp_dur_index_gpu = gpuarray.empty(tmp_size, dtype=np.int32)
        tmp_t0_index_gpu = gpuarray.empty(tmp_size, dtype=np.int32)
        # second pass output
        sgl_likerat_gpu = gpuarray.empty(1, dtype=np.float32)
        sgl_depth_gpu = gpuarray.empty(1, dtype=np.float32)
        sgl_var_depth_gpu = gpuarray.empty(1, dtype=np.float32)
        sgl_dur_index_gpu = gpuarray.empty(1, dtype=np.int32)
        sgl_t0_index_gpu = gpuarray.empty(1, dtype=np.int32)

        # type specification
        _tm_size = np.int32(self.transit_model_size)
        _duration_count = np.int32(self.duration_count)
        _all_t0_stride_count = np.int32(self.num_t0_strides)

        # run the first kernel
        _periodic_search_k1(
            _period_in_strides,
            self.like_ratio_2d_gpu, self.depth_2d_gpu, self.var_depth_2d_gpu,
            _all_t0_stride_count,
            first_d_in_range_idx, last_d_in_range_idx, max_transit_count,
            tmp_likerat_gpu, tmp_depth_gpu, tmp_var_depth_gpu,
            tmp_dur_index_gpu, tmp_t0_index_gpu,
            block=block_size_k1, grid=grid_size_k1, shared=smem_size_k1
        )

        # run parameters for the second kernel
        # block and grid sizes
        block_size_k2 = 1024, 1, 1  # the second and third dimensions should always have size 1
        grid_size_k2 = (
            int(np.ceil(tmp_size / block_size_k2[0])),
            int(1)
        )
        # shared memory size, space for 2x single-precision floats per thread
        # we need to do a max() reduction while also finding the thread id of the maximum value
        smem_size_k2 = int(2 * 4 * block_size_k2[0])

        # final reduction operation to obtain the best parameters
        _periodic_search_k2(
            tmp_likerat_gpu, tmp_depth_gpu, tmp_var_depth_gpu, tmp_dur_index_gpu, tmp_t0_index_gpu,
            sgl_likerat_gpu, sgl_depth_gpu, sgl_var_depth_gpu, sgl_dur_index_gpu, sgl_t0_index_gpu,
            np.int32(tmp_size),
            block=block_size_k2, grid=grid_size_k2, shared=smem_size_k2
        )

        # read outputs
        lrat_out = sgl_likerat_gpu.get()[0]
        depth_out = sgl_depth_gpu.get()[0]
        vdepth_out = sgl_var_depth_gpu.get()[0]
        t0_idx_out = sgl_t0_index_gpu.get()[0]
        dur_idx_out = sgl_dur_index_gpu.get()[0]

        return lrat_out, depth_out, vdepth_out, t0_idx_out, dur_idx_out

    def monotransit_search(self, n_warps=4096, verbose=True):
        """
        Perform a grid search in t_start and duration for transit-like signals
        in the light curve.

        Parameters
        ----------
        n_warps : int, optional
            The number of warps to use, default 4096. We want this to be around
            a low integer multiple of the number of concurrent warps able to
            run on the GPU.
            The A100 has 108 SMs * 64 warps = 6912 concurrent warps.
            The RTX A5000 has 64 SMs * 48 warps = 3072 concurrent warps.
            Striding in this way limits the number of reads of the transit
            model from global into shared memory. This value shouldn't
            exceed the number of t0 strides times the number of durations.
        verbose : bool, optional
            If `True`, reports the parameters of the maximum likelihood and
            maximum SNR transits. Default is `True`.

        Returns
        -------
        MonotransitResult
        """
        if verbose:
            print("commencing monotransit search")

        # initialise output arrays on the gpu
        # t0s are along the rows, durations along columns
        # i.e.
        # [[t0,d0  t1,d0  t2,d0]
        #  [t0,d1  t1,d1  t2,d1]
        #  [t0,d2  t1,d2  t2,d2]]
        # Numpy and C are row-major
        outshape_2d = self.duration_count, self.num_t0_strides
        self.like_ratio_2d_gpu = gpuarray.empty(outshape_2d, dtype=np.float32)
        self.depth_2d_gpu = gpuarray.empty(outshape_2d, dtype=np.float32)
        self.var_depth_2d_gpu = gpuarray.empty(outshape_2d, dtype=np.float32)

        # send the light curve to the gpu
        self.offset_time_gpu = to_gpu(self.lc.offset_time, np.float32)
        self.offset_flux_gpu = to_gpu(self.lc.offset_flux, np.float32)
        self.flux_weight_gpu = to_gpu(self.lc.flux_weight, np.float32)

        # send the durations to the gpu
        self.durations_gpu = to_gpu(self.durations, np.float32)

        # block and grid sizes
        block_size = 32, 1, 1  # 1 warp per block
        grid_size = int(np.ceil(n_warps / outshape_2d[0])), int(outshape_2d[0])
        # shared memory size - space for the transit model plus 2 elements per
        #                      thread, with 4 bytes per element
        smem_size = int(4 * (self.transit_model_size + 2 * block_size[0]))

        # type specification
        _cadence = np.float32(self.lc.cadence)
        _t0_stride_length = np.float32(self.t0_stride_length)
        _n_elem = np.int32(self.lc.num_points)
        _tm_size = np.int32(self.transit_model_size)
        _duration_count = np.int32(self.duration_count)
        _t0_stride_count = np.int32(self.num_t0_strides)

        # record the start time
        t0 = time()

        # run the kernel
        _monotransit_search(
            self.offset_time_gpu, self.offset_flux_gpu, self.flux_weight_gpu, _cadence, _n_elem,
            self.offset_transit_model_gpu, _tm_size,
            self.durations_gpu, _duration_count,
            _t0_stride_length, _t0_stride_count,
            self.like_ratio_2d_gpu, self.depth_2d_gpu, self.var_depth_2d_gpu,
            block=block_size, grid=grid_size, shared=smem_size
        )

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time and report the elapsed time
        t1 = time()
        if verbose:
            print(f"completed in {t1 - t0:.3f} seconds")

        # generate the monotransit search result
        self.monotransit_result = MonotransitResult(
            raw_lightcurve=self.lc_in,
            resampled_lightcurve=self.lc,
            duration_array=self.durations,
            t0_array=self.t0s,
            loglike_constant=self.lc.flat_loglike,
            likeratio_array=self.like_ratio_2d_gpu.get(),
            depth_array=self.depth_2d_gpu.get(),
            depth_variance_array=self.var_depth_2d_gpu.get(),
            input_model=self.input_model,
        )

        if verbose:
            # report the best candidates
            print(f"max likelihood monotransit:\n"
                  f"{self.monotransit_result.get_max_likelihood_parameters()}\n"
                  f"max SNR monotransit:\n"
                  f"{self.monotransit_result.get_max_snr_parameters()}")

        return self.monotransit_result

    def get_trend(self, kernel_width, min_depth_ppm=10.0, min_obs_count=20, n_warps=4096, verbose=True):
        """
        Obtain the light curve trend, after a preliminary filtering of transit signals.

        Parameters
        ----------
        kernel_width : float
            Width of the detrending kernel in days. This might be motivated by
            some prior knowledge about the activity or rotation rate of the
            target, but should be longer than the maximum transit duration.
        min_depth_ppm : float, optional
            Minimum transit depth to consider in ppm. Default 10 ppm.
        min_obs_count : int, optional
            Minimum number of observations required in the kernel window. Default 20.
        n_warps : int, optional
            The number of warps to use, default 4096. We want this to be around
            a low integer multiple of the number of concurrent warps able to
            run on the GPU.
            The A100 has 108 SMs * 64 warps = 6912 concurrent warps.
            The RTX A5000 has 64 SMs * 48 warps = 3072 concurrent warps.
            Striding in this way limits the number of reads of the transit
            model from global into shared memory. This value shouldn't
            exceed the number of t0 strides times the number of durations.
        verbose : bool, optional
            If `True`, reports with additional verbosity.

        Returns
        -------
        None
        """
        if verbose:
            print("commencing detrending")

        # initialise output arrays on the gpu
        # tss are along the rows, durations along columns
        # i.e.
        # [[t0,d0  t1,d0  t2,d0]
        #  [t0,d1  t1,d1  t2,d1]
        #  [t0,d2  t1,d2  t2,d2]]
        # Numpy and C are row-major
        outshape_2d = self.duration_count, self.num_t0_strides
        BIC_ratio_gpu = gpuarray.empty(outshape_2d, dtype=np.float32)
        ll_tr_gpu = gpuarray.empty(outshape_2d, dtype=np.float32)

        # send the light curve to the gpu
        self.offset_time_gpu = to_gpu(self.lc.offset_time, np.float32)
        self.offset_flux_gpu = to_gpu(self.lc.offset_flux, np.float32)
        self.flux_weight_gpu = to_gpu(self.lc.flux_weight, np.float32)

        # transit mask array
        transit_mask_gpu = gpuarray.zeros(self.lc.num_points, dtype=np.int32)
        # trend array
        flux_trend_gpu = gpuarray.empty(self.lc.num_points, dtype=np.float32)

        # send the durations to the gpu
        self.durations_gpu = to_gpu(self.durations, np.float32)

        # block and grid sizes
        block_size = 32, 1, 1  # 1 warp per block
        grid_size = int(np.ceil(n_warps / outshape_2d[0])), int(outshape_2d[0])
        # shared memory size -
        #     space for the transit model as f32
        #     + 15 elements per thread for the f64 intermediate arrays
        #     + 1 element per thread for the i32 obs count array
        #                      thread, with 4 bytes per element
        smem_size = int(
            4 * self.transit_model_size
            + 8 * 15 * block_size[0]
            + 4 * 1 * block_size[0]
        )

        # type specification
        _kernel_half_width = np.int32(np.ceil(0.5 * kernel_width / self.lc.cadence))
        _cadence = np.float32(self.lc.cadence)
        _t0_stride_length = np.float32(self.t0_stride_length)
        _min_depth_ppm = np.float32(min_depth_ppm)
        _n_elem = np.int32(self.lc.num_points)
        _tm_size = np.int32(self.transit_model_size)
        _duration_count = np.int32(self.duration_count)
        _t0_stride_count = np.int32(self.num_t0_strides)
        _min_in_window = np.int32(min_obs_count)

        # record the start time
        # t0 = time()

        # run the kernel
        _detrender_k1(
            self.offset_time_gpu, self.offset_flux_gpu, self.flux_weight_gpu,
            _kernel_half_width, _min_depth_ppm, _min_in_window, _cadence, _n_elem,
            self.offset_transit_model_gpu, _tm_size,
            self.durations_gpu, _duration_count,
            _t0_stride_length, _t0_stride_count,
            BIC_ratio_gpu, ll_tr_gpu,
            block=block_size, grid=grid_size, shared=smem_size
        )

        # initialise t0 and duration masks
        _t0_mask = np.full_like(self.lc.offset_time, np.nan).astype(np.float32)
        _d_mask = np.full_like(self.lc.offset_time, np.nan).astype(np.float32)
        # identify the transits
        _ll = ll_tr_gpu.get()
        _ll[BIC_ratio_gpu.get() < 0] = np.nan
        transits = []
        while np.any(np.isfinite(_ll)):
            d, t = np.unravel_index(
                np.nanargmax(_ll), _ll.shape
            )
            duration = self.durations[d]
            t0 = self.t0s[t]
            transits.append((t0, duration))
            for n, d in enumerate(self.durations):
                itr1d = np.abs(self.lc.offset_time - t0) < (0.5 * (duration + d))
                if not np.any(np.isfinite(_t0_mask[itr1d])):
                    # don't update the mask if a higher likelihood model is already there
                    _t0_mask[itr1d] = t0
                    _d_mask[itr1d] = duration
                itr2d = np.abs(self.t0s - t0) < (0.5 * (duration + d))
                _ll[n, itr2d] = np.nan
        print("found", len(transits), "transits")

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(_t0_mask)
        plt.figure()
        plt.plot(_d_mask)
        plt.show()

        return ll_tr_gpu.get(), BIC_ratio_gpu.get()

        # determine the BIC threshold
        # todo user specifiable threshold
        BIC_threshold = np.float32(
            np.nanmedian(BIC_ratio_gpu.get())
        )
        BIC_threshold = np.float32(
            np.nanpercentile(BIC_ratio_gpu.get(), 90)
        )
        print(BIC_threshold)

        # new grid shape for kernel 2
        grid_size2 = int(np.ceil(outshape_2d[1]/block_size[0])), int(outshape_2d[0])

        # run the kernel
        _detrender_k2(
            _cadence, transit_mask_gpu, _n_elem,
            self.durations_gpu, _duration_count,
            _t0_stride_length, _t0_stride_count,
            BIC_ratio_gpu, BIC_threshold,
            block=block_size, grid=grid_size2
        )

        # reweight in-transit points to zero
        mask = transit_mask_gpu.get()
        new_weights = self.lc.flux_weight.copy()
        new_weights[mask == 1] = 0.0
        # send the new weights to the gpu
        new_weights_gpu = to_gpu(new_weights, np.float32)

        # new grid shape for kernel 3
        grid_size3 = int(np.ceil(_n_elem/block_size[0])), int(1)

        # run the kernel
        _detrender_k3(
            self.offset_time_gpu, self.offset_flux_gpu, new_weights_gpu,
            _kernel_width, _cadence, _n_elem, flux_trend_gpu,
            block=block_size, grid=grid_size3
        )

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time and report the elapsed time
        t1 = time()
        if verbose:
            print(f"completed in {t1 - t0:.3f} seconds")

        return BIC_ratio_gpu.get(), mask, flux_trend_gpu.get()


if __name__ == "__main__":
    raise ImportError("Don't try to run this code directly")
