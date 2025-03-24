#!/usr/bin/env python3

# Copyright (c) 2024 Leigh C. Smith - lsmith@ast.cam.ac.uk
# Exceptions are noted within their relevant locations within this file
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import pycuda.driver as drv
import numpy as np
import warnings
from scipy.interpolate import interp1d
from copy import deepcopy
from time import time
from dataclasses import dataclass
from tqdm.auto import tqdm


class Constants(object):
    seconds_per_day = 86400  # s
    G = 6.67430e-11  # m**3 * kg**-1 * s**-2
    solar_radius = 6.957e+8  # m, IAU
    solar_mass = 1.9891e+30  # kg, IAU
    jupiter_radius = 7.1492e+7  # m, IAU, equatorial


# read the cuda source code
cu_src_file_path = os.path.join(os.path.dirname(__file__), "cetra.cu")
with open(cu_src_file_path, "r") as cu_src_file:
    cu_src = SourceModule(cu_src_file.read())

# extract the kernels
# light curve resampling kernels
_resample_kernel_1 = cu_src.get_function("resample_k1")
_resample_kernel_2 = cu_src.get_function("resample_k2")
# transit search kernels
_linear_search_kernel = cu_src.get_function("linear_search")
_periodic_search_k1 = cu_src.get_function("periodic_search_k1")
_periodic_search_k2 = cu_src.get_function("periodic_search_k2")
# detrending kernels
_detrender_quadfit = cu_src.get_function("detrender_quadfit")
_detrender_qtrfit = cu_src.get_function("detrender_qtrfit")
_detrender_calc_IC = cu_src.get_function("detrender_calc_IC")
_detrender_fit_trend = cu_src.get_function("detrender_fit_trend")


class LightCurve(object):
    """
    A light curve
    """
    def __init__(self, times, fluxes, flux_errors, resample_cadence=None, verbose=True):
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
        resample_cadence : float, optional
            The cadence (in seconds) to use for resampling the light curve. By
            default, the module will try to detect the underlying cadence.
        verbose : bool, optional
            If True (the default), reports various messages.
        """
        # get the number of elements
        self.input_num_points = len(times)

        # all arrays must be the same length
        if len(fluxes) != self.input_num_points:
            raise RuntimeError("len(fluxes) != len(times)")
        if len(flux_errors) != self.input_num_points:
            raise RuntimeError("len(flux_errors) != len(times)")

        # make sure the input time and flux arrays contain no NaNs
        if np.any(np.isnan(times)):
            raise ValueError("one or more time values is NaN")
        if np.any(np.isnan(flux_errors)):
            raise ValueError("one or more flux error values is NaN")

        # NaN fluxes are allowed - but they must have infinite error
        if np.any(np.logical_and(np.isnan(fluxes), ~np.isinf(flux_errors))):
            raise ValueError(
                "One or more flux values with finite error is NaN. NaN fluxes "
                "are allowed, but they must have infinite error."
            )

        # make sure the fluxes and errors are valid
        # zero flux is valid, zero flux error is not
        if np.any(fluxes < 0.0):
            raise ValueError("one or more flux values is negative")
        if np.any(flux_errors <= 0.0):
            raise ValueError("one or more flux error values is zero or negative")

        # make sure that the duration of the light curve is positive
        self.input_epoch_baseline = np.ptp(times)
        if self.input_epoch_baseline <= 0:
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
        self.input_time = times[srt_idx]
        self.input_flux = fluxes[srt_idx]
        self.input_flux_error = flux_errors[srt_idx]

        # attempt to determine the cadence
        # delta t in milliseconds
        ms_per_day = Constants.seconds_per_day * 1000
        _dt = np.round(np.diff(self.input_time * ms_per_day)).astype(np.int64)
        uq, ct = np.unique(_dt, return_counts=True)
        if ct.max() > 1:
            # modal value if possible
            modal_dt = uq[np.argmax(ct)]
            self.input_cadence = modal_dt / ms_per_day
        else:
            # median otherwise
            self.input_cadence = np.nanmedian(_dt) / ms_per_day
        self.input_cadence_range = _dt.min() / ms_per_day, _dt.max() / ms_per_day

        # the time point of the first and last observations
        self.input_time_start = np.min(self.input_time)
        self.input_time_end = np.max(self.input_time)

        # these will be populated if we run detrending on the light curve
        self.offset_trend = None
        self.error_multiplier = None
        # this will be populated if we mask transit(s) in the light curve
        # will be an array where True is in-transit
        self.transit_mask = None

        # resample the new light curve to regularise the cadence
        # a regularised cadence is necessary so that we can cheaply determine
        # the array location of a given point in time
        # resampled points containing no observed data have null fluxes and
        # infinite flux errors
        if resample_cadence is None:
            self.cadence = self.input_cadence
        else:
            self.cadence = resample_cadence / Constants.seconds_per_day
        # perform the resampling
        self.time, self.flux, self.flux_error = self.resample(self.cadence)

        # the number of points in the resampled light curve
        self.size = len(self.time)

        # the offset time is an array of times starting at zero,
        # it is related to the input time sequence as:
        # offset_time = input_time - reference_time
        self.reference_time = self.time[0]
        self.offset_time = self.time - self.reference_time

        # flux errors to weights
        self.flux_weight = 1.0 / self.flux_error ** 2

        # to save a lot of 1-f operations later, lets do it now
        self.offset_flux = 1.0 - self.flux
        self.offset_flux_error = self.flux_error

        # log-likelihood of constant flux model
        self.flat_loglike = np.nansum(
            - 0.5 * self.offset_flux ** 2 / self.offset_flux_error**2
            - 0.5 * np.log(2 * np.pi * self.offset_flux_error**2)
        )

        if verbose:
            # report the input light curve information
            print(str(self))

    def __str__(self):
        return f"""input light curve has {self.input_num_points} elements, \
cadence: {self.input_cadence * Constants.seconds_per_day:.0f}s \
(range: {self.input_cadence_range[0] * Constants.seconds_per_day:.0f}s -> \
{self.input_cadence_range[1] * Constants.seconds_per_day:.0f}s)
constant flux model log-likelihood: {self.flat_loglike:.3e}
resampled light curve has {self.size} elements, \
cadence: {self.cadence * Constants.seconds_per_day:.0f}s"""

    def resample(self, new_cadence, cuda_blocksize=1024):
        """
        Resample the light curve at a new cadence.
        (Note: Any existing transit masks will be lost, it's recommended
        to remask again using the Transit objects.)

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

        # transit mask warning
        if self.transit_mask is not None:
            warnings.warn("Transit masks are removed on resampling")

        # generate the new time sequence
        new_times = np.arange(
            start=self.input_time_start,
            stop=self.input_time_end + new_cadence,
            step=new_cadence
        )

        # send some arrays to the gpu
        # send the time as an offset from the first
        _time = to_gpu(self.input_time - self.input_time_start, np.float64)
        # send the flux and error arrays as they are
        _flux = to_gpu(self.input_flux, np.float64)
        _ferr = to_gpu(self.input_flux_error, np.float64)
        # initialise output arrays on the gpu
        _rflux_out = gpuarray.zeros(new_times.size, dtype=np.float64)
        _err_out = gpuarray.zeros(new_times.size, dtype=np.float64)
        _sum_fw = gpuarray.zeros(new_times.size, dtype=np.float64)
        _sum_w = gpuarray.zeros(new_times.size, dtype=np.float64)
        # type specification
        _cadence = np.float64(new_cadence)
        _n_elem = np.int32(self.input_num_points)
        _n_elem_out = np.int32(new_times.size)

        # set the cuda block and grid sizes
        _blocksize = (cuda_blocksize, 1, 1)
        _gridsize1 = (int(np.ceil(_n_elem / cuda_blocksize)), 1, 1)
        _gridsize2 = (int(np.ceil(_n_elem_out / cuda_blocksize)), 1, 1)

        # run the kernel to sum the fluxes and their weights
        _resample_kernel_1(
            _time, _flux, _ferr, _cadence, _n_elem, _sum_fw, _sum_w,
            block=_blocksize, grid=_gridsize1
        )
        # now run the division kernel
        _resample_kernel_2(
            _sum_fw, _sum_w, _rflux_out, _err_out, _n_elem_out,
            block=_blocksize, grid=_gridsize2
        )

        return new_times, _rflux_out.get(), _err_out.get()

    def pad(self, num_points_prepend, num_points_append, verbose=True):
        """
        Pad the light curve with null data.
        (Note: Any existing transit masks will be lost, it's recommended
        to remask again using the Transit objects.)

        Parameters
        ----------
        num_points_prepend : int
            The number of points to prepend to the light curve.
        num_points_append : int
            The number of points to append to the light curve.
        verbose : bool, optional
            If True (the default), reports various messages.
        """
        # transit mask warning
        if self.transit_mask is not None:
            warnings.warn("Transit masks are removed on padding")

        # prepend
        _t0 = self.time[0] - np.arange(num_points_prepend, 0, -1) * self.cadence
        _f0 = np.full(num_points_prepend, np.nan, dtype=self.flux.dtype)
        _ef0 = np.full(num_points_prepend, np.inf, dtype=self.flux_error.dtype)
        self.time = np.insert(self.time, 0, _t0)
        self.flux = np.insert(self.flux, 0, _f0)
        self.flux_error = np.insert(self.flux_error, 0, _ef0)
        if self.offset_trend is not None:
            # prepend NaNs
            self.offset_trend = np.insert(self.offset_trend, 0, _f0)

        # append
        _t1 = self.time[-1] + np.arange(1, num_points_append + 1) * self.cadence
        _f1 = np.full(num_points_append, np.nan, dtype=self.flux.dtype)
        _ef1 = np.full(num_points_append, np.inf, dtype=self.flux_error.dtype)
        self.time = np.append(self.time, _t1)
        self.flux = np.append(self.flux, _f1)
        self.flux_error = np.append(self.flux_error, _ef1)
        if self.offset_trend is not None:
            # append NaNs
            self.offset_trend = np.append(self.offset_trend, _f1)


        # update derivative instance variables
        self.size = len(self.time)
        self.reference_time = self.time[0]
        self.offset_time = self.time - self.reference_time
        self.flux_weight = 1.0 / self.flux_error ** 2
        self.offset_flux = 1.0 - self.flux
        self.offset_flux_error = self.flux_error
        # the log-likelihood is the same

        if verbose:
            print(f"padded {num_points_prepend} null points to the start "
                  f"and {num_points_append} null points to the end of the "
                  f"light curve")

    def mask_transit(self, transit, duration_multiplier=1.0, return_mask=False):
        """
        Mask a transit

        Parameters
        ----------
        transit : Transit
            The transit to mask (can be single or have period)
        duration_multiplier : float, optional
            A multiplier on the transit duration, applied to ensure that all
            in-transit flux is removed. Default 1.0, i.e. no border.
        return_mask : bool, optional
            If `True`, returns the mask for this transit only. Default `False`.

        Returns
        -------
        1D ndarray with in-transit points as `True` and out of transit as
        `False`, if `return_mask==True`.
        """
        # compute the phase
        phase = self.time - transit.t0
        # if periodic
        if transit.period is not None:
            phase %= transit.period
            phase[phase > (0.5 * transit.period)] -= transit.period

        # duration to mask
        mask_duration = transit.duration * duration_multiplier

        # generate the mask
        in_transit = np.abs(phase) < (0.5 * mask_duration)

        if self.transit_mask is None:
            # if this is a new mask, set the transit mask
            self.transit_mask = in_transit
        else:
            # if this is not a new mask, update the existing
            self.transit_mask += in_transit

        # return the mask for this transit only, if requested
        if return_mask:
            return in_transit

    def copy(self):
        """
        Return a copy of this LightCurve instance

        Returns
        -------
        A copy of this LightCurve instance
        """
        return deepcopy(self)


class TransitModel(object):
    """
    A transit model
    """
    def __init__(self, transit_model, downsamples=1024, verbose=True):
        """
        Initialise the TransitModel class

        Parameters
        ----------
        transit_model : array-like or string
            transit_model can be either a 1D array containing floating point
            offsets relative to baseline flux, or a string corresponding to
            one of three internal models.
            If a 1D array is supplied, the maximum offset from baseline flux
            should be 1. Ideally the array length will be greater than the
            value of `downsamples` (which is 1024 elements by default).
            CETRA does not force transit depth to be positive. (So e.g. a
            flare model can be provided and the depths returned will be
            negative.)
            Alternatively, one of the strings 'b32', 'b93' or 'b99' can be
            supplied. In this case, a model with impact parameter 0.32,
            0.93 or 0.99, respectively, will be used. All other parameters
            are:
                'rp': 0.03,
                'u': (0.4804, 0.1867),
                'period': 10.0,
                'semimajor_axis': 20.0.
        downsamples : int, optional
            Downsample the transit model to this number of samples for use by
            the GPU kernels. Ideally a power of 2. It's best to be
            conservative with this value, since the volume of GPU shared
            memory (in which the transit model is stored) is fairly limited.
            The default value is 1024 elements. The larger the value, the
            smaller the max error in the model. A value of 1024 gives a
            maximum error of ~1%, 2048 is ~0.5%, 512 is ~2%.
        verbose : bool, optional
            If True (the default), reports various messages.
        """
        # now deal with any transit model inputs
        if isinstance(transit_model, list) or isinstance(transit_model, np.ndarray):
            # use the user-provided transit model
            if np.any(np.isnan(transit_model)):
                raise ValueError("transit_model cannot contain NaNs")
            elif np.any(np.isinf(transit_model)):
                raise ValueError("transit_model cannot contain Infs")
            else:
                self.input_model = np.asarray(transit_model)

            if verbose:
                print(f"user-provided transit model with {len(transit_model)} elements")

        elif transit_model in ['b32', 'b93', 'b99']:
            # the user has specified one of the internal transit models
            _impact_params = {'b32': 0.32, 'b93': 0.93, 'b99': 0.99}
            tmod_file_path = os.path.join(os.path.dirname(__file__),
                                          f"transit_model_{transit_model}.npz")
            tmod = np.load(tmod_file_path)
            self.input_model = tmod["model_array"]

            if verbose:
                print(f"using internal transit model with impact parameter: "
                      f"{_impact_params[transit_model]}")

        else:
            # the user input is invalid
            raise RuntimeError(
                "transit_model not recognised. "
                "Is it an array, or one of 'b32', 'b93' or 'b99'?"
            )

        # generate the internal transit model
        # self.interpolator is a 1d interpolator for the transit
        # model with x values between 0 and 1
        # self.transit_model is a 1d array of the transit model with a given
        # number of evenly spaced samples
        self.interpolator, self.model = self.interpolate(downsamples)
        # store the transit model size as an instance variable
        self.size = int(downsamples)

        # to save a lot of 1-f transit model operations later, lets do it now
        self.offset_model = 1.0 - self.model
        # send the offset transit model to the gpu
        self.offset_model_gpu_f4 = to_gpu(self.offset_model, np.float32)
        self.offset_model_gpu_f8 = to_gpu(self.offset_model, np.float64)

        if verbose:
            # report the transit model size
            print(f"transit model size: {self.size} elements")
            # report the nearest neighbour error
            _nn_error = self.nn_error()
            print(f"maximum nearest-neighbour error: {100 * _nn_error[0]:.2e}%")
            print(f"   mean nearest-neighbour error: {100 * _nn_error[1]:.2e}%")

    def interpolate(self, samples, kind='linear'):
        """
        Resample an input model using interpolation

        Parameters
        ----------
        samples : int
            The required number of samples
        kind : str, optional
            A scipy.interpolate.interp1d interpolation method.
            'linear' is the default.

        Returns
        -------
        Model with the required number of samples,
        and the interpolator object
        """
        i1d = interp1d(
            np.linspace(0, 1, len(self.input_model)), self.input_model,
            kind=kind, copy=True, bounds_error=False, fill_value=1.0,
            assume_sorted=True
        )
        return i1d, i1d(np.linspace(0, 1, samples))

    def nn_error(self):
        """
        Return the maximum and mean error in the model due to the use of
        nearest-neighbour interpolation.
        """
        s_inter = (self.size - 1) * 2 + 1
        _, m_all = self.interpolate(s_inter)
        m_orig = m_all[0::2]
        m_inter = m_all[1::2]
        frac_diff = m_orig[1:] - m_inter
        return np.max(np.abs(frac_diff)), np.mean(np.abs(frac_diff))

    def get_model_lc(self, times, transit):
        """
        Get orbital phase and model flux at the given time points for the
        given Transit object.

        Parameters
        ----------
        times : array-like
            The time points for which the model flux is sought.
        transit : Transit
            The Transit object for the given transit.

        Returns
        -------
        Two 1D ndarrays of orbital phases and model fluxes for the given time
        points
        """
        # compute the phase
        phase = times - transit.t0
        # if periodic
        if transit.period is not None:
            phase %= transit.period
            phase[phase > (0.5 * transit.period)] -= transit.period

        # compute the model flux
        model = self.interpolator(phase / transit.duration + 0.5)
        # set the depth
        model = 1.0 - transit.depth * (1.0 - model)

        return phase, model


@dataclass
class Transit(object):
    """
    A transit object
    """
    t0: float
    duration: float
    depth: float
    depth_error: float
    period: float = None


@dataclass
class LinearResult(object):
    """
    A linear search result
    """
    light_curve: LightCurve
    transit_model: TransitModel
    duration_array: np.ndarray
    t0_array: np.ndarray
    like_ratio_array: np.ndarray
    depth_array: np.ndarray
    depth_variance_array: np.ndarray

    def get_max_likelihood_parameters(self):
        """
        Return the parameters of the maximum likelihood TCE

        Returns
        -------
        The maximum likelihood TCE as a Transit object.
        """
        d, t = np.unravel_index(
            np.nanargmax(self.like_ratio_array), self.like_ratio_array.shape
        )
        return self.get_params(d, t)

    def get_max_snr_parameters(self, absolute_depth=False):
        """
        Return the parameters of the maximum SNR TCE

        Parameters
        ----------
        absolute_depth : bool, optional
            If `True`, computes SNR as |S/N|, otherwise SNR is S/N.
            `False` by default.

        Returns
        -------
        The maximum SNR TCE as a Transit object.
        """
        snr_array = self.depth_array / np.sqrt(self.depth_variance_array)
        if absolute_depth:
            snr_array = np.abs(snr_array)
        d, t = np.unravel_index(np.nanargmax(snr_array), snr_array.shape)
        return self.get_params(d, t)

    def get_params(self, duration_index: int, t0_index: int):
        """
        Find the parameters of a TCE given duration and t0 indices

        Parameters
        ----------
        duration_index : int
            Duration index
        t0_index : int
            t0 index

        Returns
        -------
        The corresponding TCE as a Transit object
        """
        # find or compute the requested parameters
        t0 = self.t0_array[t0_index]
        duration = self.duration_array[duration_index]
        depth = self.depth_array[duration_index, t0_index]
        depth_error = np.sqrt(self.depth_variance_array[duration_index, t0_index])

        # generate the grid search Transit object
        return Transit(
            t0=t0,
            duration=duration,
            depth=depth,
            depth_error=depth_error
        )


@dataclass
class PeriodicResult(object):
    """
    A periodic search result
    """
    linear_result: LinearResult
    period_array: np.ndarray
    like_ratio_array: np.ndarray
    depth_array: np.ndarray
    depth_variance_array: np.ndarray
    duration_index_array: np.ndarray
    t0_index_array: np.ndarray

    def __post_init__(self):
        """
        post-initialisation setup steps
        """
        # reference some arrays from the linear result directly from this
        # class
        self.light_curve = self.linear_result.light_curve
        self.transit_model = self.linear_result.transit_model
        self.duration_array = self.linear_result.duration_array
        self.t0_array = self.linear_result.t0_array

    def get_max_likelihood_parameters(self):
        """
        Return the parameters of the maximum likelihood TCE

        Returns
        -------
        The maximum likelihood TCE as a Transit object.
        """
        idx = np.nanargmax(self.like_ratio_array)
        return self.get_params(idx)

    def get_max_snr_parameters(self, absolute_depth=False):
        """
        Return the parameters of the maximum SNR TCE

        Parameters
        ----------
        absolute_depth : bool, optional
            If `True`, computes SNR as |S/N|, otherwise SNR is S/N.
            `False` by default.

        Returns
        -------
        The maximum SNR TCE as a Transit object.
        """
        snr_array = self.depth_array / np.sqrt(self.depth_variance_array)
        if absolute_depth:
            snr_array = np.abs(snr_array)
        idx = np.nanargmax(snr_array)
        return self.get_params(idx)

    def get_params(self, period_index: int):
        """
        Find the parameters of a TCE given a period index

        Parameters
        ----------
        period_index : int
            The period index

        Returns
        -------
        The corresponding TCE as a Transit object
        """
        t0_index = self.t0_index_array[period_index]
        duration_index = self.duration_index_array[period_index]
        # find or compute the requested parameters
        period = self.period_array[period_index]
        t0 = self.t0_array[t0_index]
        duration = self.duration_array[duration_index]
        depth = self.depth_array[period_index]
        depth_error = np.sqrt(self.depth_variance_array[period_index])

        # we might want t0 to be the first complete transit, but it could
        # come back before the start of the data. quick fix...
        while t0 < self.light_curve.input_time_start:
            t0 += period

        # generate the grid search Transit object
        return Transit(
            t0=t0,
            duration=duration,
            depth=depth,
            depth_error=depth_error,
            period=period
        )


class TransitDetector(object):
    """
    A tool for identifying transit-like signals in stellar light curves
    """

    def __init__(
            self, light_curve, transit_model=None, durations=None,
            min_duration=0.02, max_duration=1.0, duration_log_step=1.1,
            t0_stride_fraction=0.01, verbose=True
    ):
        """
        Initialise the transit detector.

        Parameters
        ----------
        light_curve : LightCurve
            The light curve.
        transit_model : TransitModel, optional
            This TransitModel will be used, if provided, instead of the default.
            The default model is for a transit with the following parameters:
                'rp': 0.03,
                'b': 0.32,
                'u': (0.4804, 0.1867),
                'period': 10.0,
                'semimajor_axis': 20.0.
        durations : array-like, optional
            User-specified grid of durations in days. If not provided, the
            module computes a grid using the minimum and maximum durations
            and the log step.
        min_duration : float, optional
            Minimum transit duration to check in days, default 0.02.
            Unnecessary if an array of durations is provided.
        max_duration : float, optional
            Maximum transit duration to check in days, default 1.0.
            Unnecessary if an array of durations is provided.
        duration_log_step : float, optional
            The log-spacing of the durations to be used if the duration grid
            is to be internally determined. Default 1.1.
            Unnecessary if an array of durations is provided.
        t0_stride_fraction : float, optional
            The fraction of the minimum duration that determines the length of
            each t0 stride. The default is 1% of the minimum duration.
        verbose : bool, optional
            If True (the default), reports various messages.
        """
        # store a copy of the input LightCurve as an instance variable
        if isinstance(light_curve, LightCurve):
            self.lc = light_curve.copy()
        else:
            raise RuntimeError("`light_curve` should be a LightCurve instance")

        # validate/generate the duration grid
        self.durations = duration_grid(
            durations=durations,
            min_duration=min_duration,
            max_duration=max_duration,
            log_step=duration_log_step,
            verbose=verbose
        )
        # how many durations are there?
        self.duration_count = len(self.durations)
        # send the durations to the gpu
        self.durations_gpu_f4 = to_gpu(self.durations, np.float32)

        # Pad the light curve with null data to make simpler the algorithm
        # that searches for transits that begin before or end after the data.
        # This requires a regular observing cadence, another benefit of the
        # light curve resampling operation.
        # pad by half of the maximum duration
        num_pad = int(np.ceil(0.5 * np.max(self.durations) / self.lc.cadence))
        self.lc.pad(num_pad, num_pad, verbose=verbose)

        # deal with the transit model
        if transit_model is None:
            self.transit_model = TransitModel('b32', verbose=verbose)
        elif isinstance(transit_model, TransitModel):
            self.transit_model = transit_model
        else:
            raise RuntimeError("transit_model not recognised")

        # determine the t0 stride length
        self.t0_stride_length = np.min(self.durations) * t0_stride_fraction
        if verbose:
            print(f"t0 stride length: {self.t0_stride_length * Constants.seconds_per_day:.3f} seconds")
        # generate the grid of t0s
        # we go from the start to the end of the PADDED light curve
        self.t0_array = np.arange(self.lc.time[0], self.lc.time[-1], self.t0_stride_length)
        self.num_t0_strides = len(self.t0_array)
        if verbose:
            print(f"{self.num_t0_strides:d} t0 strides")

        # initialise instance variables that get populated later
        self.periods = None
        self.period_count = None
        self.like_ratio_2d_gpu = None
        self.depth_2d_gpu = None
        self.var_depth_2d_gpu = None
        self.linear_result = None
        self.periodic_result = None

    def get_trend(
            self,
            detection_kernel_width, detrending_kernel_width,
            IC_type=0, dIC_threshold=10.0,
            min_depth_ppm=10.0, min_obs_count=20,
            full_output=False, n_warps=4096,
            verbose=True
    ):
        """
        Obtain the trend of the light curve using a quadratic+transit model,
        after a preliminary detections of likely transit signals.

        Parameters
        ----------
        detection_kernel_width : float
            Width of the detection kernel in days. This might be motivated by
            some prior knowledge about the activity or rotation rate of the
            target, but should be longer than the maximum transit duration.
        detrending_kernel_width : float
            Width of the detrending kernel in days. This might be motivated by
            some prior knowledge about the activity or rotation rate of the
            target, but should be longer than the maximum transit duration.
            todo note that in the detrending kernel, when the locations of the
             transits are 'known', only a single depth value is fitted within
             a kernel width. This means transits closer together than the
             kernel width are likely to be poorly modelled as they'll have the
             same depth. Perhaps it's better to simply mask the detected
             transits after all...
        IC_type : int
            The information criterion type.
            0 is Bayesian (default), 1 is Akaike.
        dIC_threshold : float
            The information criterion difference threshold to use to select
            regions in the t0,duration space in which the quadratic+transit
            model is more likely than the quadratic model. The statistic is
            defined as dIC = IC_quadratic - IC_transit, so larger dIC values
            mean the quad+transit model is more preferred. Default 10.
        min_depth_ppm : float, optional
            Minimum transit depth to consider in ppm. Default 10 ppm.
        min_obs_count : int, optional
            Minimum number of observations required in the kernel window. Default 20.
        full_output : bool, optional
            If `True`, returns some intermediate arrays. If `False` (the default),
            nothing is returned.
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
        If `full_output == True`, then arrays of:
            1) The number of data points in each detection kernel window
                ndarray of len(t0_array)
            2) The log-likelihoods of the quadratic models
                ndarray of len(t0_array)
            3) The log-likelihoods of the quadratic+transit models
                ndarray of shape (len(durations), len(t0_array))
            4) The delta IC array
                ndarray of shape (len(durations), len(t0_array))
            5) The transit(s) model
                ndarray of len(light curve)
            6) The fitted trend array
                ndarray of len(light curve)
            7) The estimated error multiplier
                single floating point value
        """
        # verify the IC type input
        if IC_type not in [0, 1]:
            raise ValueError("Information criterion type must be int(0) or int(1).")

        if verbose:
            print("obtaining trend")

        # step 1, fit the basic quadratic models while storing some
        # intermediate variables.
        # compute the log-likelihoods
        # these are the same for all durations, so we can save time (vs. a
        # previous version of this code) by doing it only once for all
        # durations

        # log-likelihood of the quadratic model
        ll_quad = gpuarray.empty(self.num_t0_strides, dtype=np.float64)
        # intermediate variables:
        #   key for variable names:
        #       s: sum
        #       w: weight
        #       x: time
        #       y: flux
        sw = gpuarray.zeros(self.num_t0_strides, np.float64)
        swx = gpuarray.zeros(self.num_t0_strides, np.float64)
        swy = gpuarray.zeros(self.num_t0_strides, np.float64)
        swxx = gpuarray.zeros(self.num_t0_strides, np.float64)
        swxy = gpuarray.zeros(self.num_t0_strides, np.float64)
        swxxx = gpuarray.zeros(self.num_t0_strides, np.float64)
        swxxy = gpuarray.zeros(self.num_t0_strides, np.float64)
        swxxxx = gpuarray.zeros(self.num_t0_strides, np.float64)
        # number of data points in window
        num_pts = gpuarray.zeros(self.num_t0_strides, np.int32)

        # block and grid sizes
        block_size_k1 = 512, 1, 1
        grid_size_k1 = int(np.ceil(self.num_t0_strides / block_size_k1[0])), 1
        # no shared memory needed for this kernel

        # type specification
        _detection_kernel_half_width = np.int32(
            np.ceil(0.5 * detection_kernel_width / self.lc.cadence)
        )
        _cadence = np.float32(self.lc.cadence)
        _t0_stride_length = np.float32(self.t0_stride_length)
        _min_in_window = np.int32(min_obs_count)
        _n_elem = np.int32(self.lc.size)
        _t0_stride_count = np.int32(self.num_t0_strides)

        # send the light curve to the gpu
        _time = to_gpu(self.lc.offset_time, np.float64)
        _flux = to_gpu(self.lc.offset_flux, np.float64)
        _wght = to_gpu(self.lc.flux_weight, np.float64)

        # record the start time
        _k1_start_time = time()

        # run the quadratic fit kernel
        _detrender_quadfit(
            _time, _flux, _wght,
            _detection_kernel_half_width, _min_in_window, _cadence, _n_elem,
            _t0_stride_length, _t0_stride_count,
            sw, swx, swy, swxx, swxy, swxxx, swxxy, swxxxx,
            num_pts, ll_quad,
            block=block_size_k1, grid=grid_size_k1
        )

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time
        _k1_stop_time = time()
        # report elapsed
        if verbose:
            print(f"kernel 1 completed in {_k1_stop_time - _k1_start_time:.3f} seconds")

        # step 2, fit the quadratic plus transit models for a range of transit
        # durations.
        # compute the log-likelihoods

        # initialise output arrays on the gpu
        # t0s are along the rows, durations along columns
        # i.e.
        # [[t0,d0  t1,d0  t2,d0]
        #  [t0,d1  t1,d1  t2,d1]
        #  [t0,d2  t1,d2  t2,d2]]
        # Numpy and C are row-major
        outshape_2d = self.duration_count, self.num_t0_strides
        ll_qtr = gpuarray.empty(outshape_2d, dtype=np.float64)

        # block and grid sizes
        block_size_k2 = 32, 1, 1  # 1 warp per block
        grid_size_k2 = int(np.ceil(n_warps / outshape_2d[0])), int(outshape_2d[0])
        # shared memory size -
        #     space for the transit model as f64
        #     + 6 elements per thread for the f64 intermediate arrays
        smem_size_k2 = int(
            8 * self.transit_model.size
            + 8 * 6 * block_size_k2[0]
        )

        # type specification
        _min_depth_ppm = np.float32(min_depth_ppm)
        _tm_size = np.int32(self.transit_model.size)
        _duration_count = np.int32(self.duration_count)

        # record the start time
        _k2_start_time = time()

        # run the quadratic fit kernel
        _detrender_qtrfit(
            _time, _flux, _wght, _detection_kernel_half_width,
            _min_depth_ppm, _min_in_window, _cadence, _n_elem,
            self.transit_model.offset_model_gpu_f8, _tm_size,
            self.durations_gpu_f4, _duration_count,
            _t0_stride_length, _t0_stride_count,
            sw, swx, swy, swxx, swxy, swxxx, swxxy, swxxxx,
            num_pts, ll_qtr,
            block=block_size_k2, grid=grid_size_k2, shared=smem_size_k2
        )

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time
        _k2_stop_time = time()
        # report elapsed
        if verbose:
            print(f"kernel 2 completed in {_k2_stop_time - _k2_start_time:.3f} seconds")

        # step 3, compute the information criteria

        # create the output array
        delta_IC = gpuarray.empty(outshape_2d, dtype=np.float64)

        # block and grid sizes
        block_size_k3 = 512, 1, 1
        grid_size_k3 = (int(np.ceil(self.num_t0_strides / block_size_k3[0])),
                        int(np.ceil(self.duration_count / block_size_k3[1])))

        # record the start time
        _k3_start_time = time()

        # run the information criteria calculation kernel
        _detrender_calc_IC(
            ll_quad, ll_qtr, num_pts,
            _min_in_window, np.int32(0),
            _duration_count, _t0_stride_count,
            delta_IC,
            block=block_size_k3, grid=grid_size_k3
        )

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time
        _k3_stop_time = time()
        # report elapsed
        _search_stop_time = time()
        if verbose:
            print(f"kernel 3 completed in {_k3_stop_time - _k3_start_time:.3f} seconds")

        # step 4, build the model containing the likely transits

        # record the start time
        _s4_start_time = time()

        # the transit mask
        # we're looking to specify the transits in the light curve time array
        # NOT the t0 array that we've been using up until now
        _transit_mask = np.zeros(self.lc.size, dtype=np.float64)

        # grab the delta IC array from the device
        _delta_IC = delta_IC.get()
        # nullify elements below the dIC threshold
        _delta_IC[_delta_IC < dIC_threshold] = np.nan

        # collapse the dIC array to obtain the duration index of the highest
        # dIC for a given t0 column
        # must be mindful of the fact that there are likely many all-nan
        # columns
        # this might be done more efficiently on the GPU, but let's get it
        # working first and see how slow it is
        allnan = np.count_nonzero(~np.isnan(_delta_IC), axis=0) == 0
        _delta_IC[np.isnan(_delta_IC)] = -np.inf
        idx = np.argmax(_delta_IC, axis=0, keepdims=True)
        # these are the non-null t0s and their corresponding best durations
        # and dICs
        _t0 = self.t0_array[~allnan]
        _dur = self.durations[idx[0][~allnan]]
        _dIC = np.take_along_axis(_delta_IC, idx, axis=0)[0][~allnan]

        # now work from the most to least likely transit, populating the
        # transit mask array
        while np.any(np.isfinite(_dIC)):
            max_dIC_idx = np.nanargmax(_dIC)
            t0 = _t0[max_dIC_idx]
            dur = _dur[max_dIC_idx]
            # generate the model with just this transit
            _tm = 1.0 - self.transit_model.interpolator((self.lc.time - t0)/dur + 0.5)
            # check whether this transit mask overlaps an existing transit
            if not np.any(_transit_mask[_tm > 0] > 0):
                # if not, add this transit to the mask
                _transit_mask += _tm
            # nullify this region in the _dIC array
            _dIC[np.abs(_t0 - t0) < (0.5 * dur)] = np.nan

        # record the stop time
        _s4_stop_time = time()
        # report elapsed
        _search_stop_time = time()
        if verbose:
            print(f"step 4 completed in {_s4_stop_time - _s4_start_time:.3f} seconds")

        # step 5, fit the light curve plus complete transit model

        # send the transit(s) mask to the gpu
        _transit_mask_gpu_f8 = to_gpu(_transit_mask, np.float64)

        # create the trend array on the gpu
        _trend = gpuarray.empty(self.lc.size, dtype=np.float64)

        # block and grid sizes
        block_size_k4 = 512, 1, 1
        grid_size_k4 = int(np.ceil(self.lc.size / block_size_k4[0])), 1

        # type specification for detrending kernel
        _detrending_kernel_half_width = np.int32(
            np.ceil(0.5 * detrending_kernel_width / self.lc.cadence)
        )

        # record the start time
        _k4_start_time = time()

        # run the fitting kernel
        _detrender_fit_trend(
            _time, _flux, _wght, _transit_mask_gpu_f8,
            _detrending_kernel_half_width, _min_in_window,
            _n_elem, _trend,
            block=block_size_k4, grid=grid_size_k4
        )

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time
        _k4_stop_time = time()
        # report elapsed
        if verbose:
            print(f"kernel 4 completed in {_k4_stop_time - _k4_start_time:.3f} seconds")

        # read the trend from the device
        trend = _trend.get()

        # measure the error multiplier
        detrended_offset_flux = 1 - (self.lc.flux / (1-trend))
        std_resids = np.abs(detrended_offset_flux) / self.lc.offset_flux_error
        to_include = np.isfinite(std_resids) & (_transit_mask == 0.0)
        error_multiplier = np.median(std_resids[to_include]) * 1.4826

        # record the results
        self.lc.offset_trend = trend
        self.lc.error_multiplier = error_multiplier

        if full_output:
            return (num_pts.get(), ll_quad.get(), ll_qtr.get(),
                    delta_IC.get(), _transit_mask, trend, error_multiplier)

    def linear_search(self, n_warps=4096, verbose=True):
        """
        Perform a grid search in t0 and duration for transit-like signals
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
        LinearResult
        """
        if verbose:
            print("commencing linear search")

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

        # block and grid sizes
        block_size = 32, 1, 1  # 1 warp per block
        grid_size = int(np.ceil(n_warps / outshape_2d[0])), int(outshape_2d[0])
        # shared memory size - space for the transit model plus 3 elements per
        #                      thread, with 4 bytes per element
        smem_size = int(4 * (self.transit_model.size + 3 * block_size[0]))

        # type specification
        _cadence = np.float32(self.lc.cadence)
        _n_elem = np.int32(self.lc.size)
        _tm_size = np.int32(self.transit_model.size)
        _duration_count = np.int32(self.duration_count)
        _t0_stride_length = np.float32(self.t0_stride_length)
        _t0_stride_count = np.int32(self.num_t0_strides)

        # subtract the trend if available
        if self.lc.offset_trend is not None:
            if verbose:
                print("trend available, removing")
            working_flux = 1 - (self.lc.flux / (1-self.lc.offset_trend))
        else:
            working_flux = self.lc.offset_flux
        # apply the error multiplier if available
        if self.lc.error_multiplier is not None:
            if verbose:
                print(f"error multiplier (x{self.lc.error_multiplier:.3f}) available, applying")
            working_weights = self.lc.flux_weight / self.lc.error_multiplier**2
        else:
            working_weights = self.lc.flux_weight.copy()
        # apply the transit mask if available
        if self.lc.transit_mask is not None:
            # points in previously masked transits are given zero weight
            working_weights[self.lc.transit_mask] = 0.0

        # send the relevant arrays to the gpu
        _time = to_gpu(self.lc.offset_time, np.float32)
        _flux = to_gpu(working_flux, np.float32)
        _weight = to_gpu(working_weights, np.float32)

        # record the start time
        _kernal_start_time = time()

        # run the kernel
        _linear_search_kernel(
            _time, _flux, _weight, _cadence, _n_elem,
            self.transit_model.offset_model_gpu_f4, _tm_size,
            self.durations_gpu_f4, _duration_count,
            _t0_stride_length, _t0_stride_count,
            self.like_ratio_2d_gpu, self.depth_2d_gpu, self.var_depth_2d_gpu,
            block=block_size, grid=grid_size, shared=smem_size
        )

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time and report the elapsed time
        _kernel_stop_time = time()
        if verbose:
            print(f"completed in {_kernel_stop_time - _kernal_start_time:.3f} seconds")

        # generate the LinearResult
        self.linear_result = LinearResult(
            light_curve=self.lc.copy(),
            transit_model=self.transit_model,
            duration_array=self.durations,
            t0_array=self.t0_array,
            like_ratio_array=self.like_ratio_2d_gpu.get(),
            depth_array=self.depth_2d_gpu.get(),
            depth_variance_array=self.var_depth_2d_gpu.get()
        )

        return self.linear_result

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
        Run a search for periodic signals within a LinearResult.

        Parameters
        ----------
        periods : array-like, optional
            User-provided period grid. If not provided, the module determines
            an optimal period grid using various other kwargs.
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
            limited to the light curve epoch baseline divided by
            n_transits_min. The default is to require 2 transits.
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
            to completion courtesy of the tqdm package. Otherwise, only some
            basic info/warnings are reported.

        Returns
        -------
        PeriodicResult
        """
        if self.linear_result is None:
            warnings.warn(
                "A periodic signal search cannot be run without first running "
                "a linear search"
            )
            return
        else:
            if verbose:
                print("commencing periodic signal search")

        if periods is not None:
            # use the provided period grid
            _periods = np.asarray(periods)
        else:
            # generate the period grid
            _periods = period_grid(
                self.lc.input_epoch_baseline, min_period=min_period,
                max_period=max_period, n_transits_min=n_transits_min,
                R_star=pgrid_R_star, M_star=pgrid_M_star,
                oversampling_factor=pgrid_oversample
            )

        # make sure there are no duplicates (also sorts)
        self.periods = np.unique(_periods)
        self.period_count = len(self.periods)

        # do nothing if the period grid is empty
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
        _search_start_time = time()

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

        # make sure the operation is finished before recording the time
        # (device calls are asynchronous)
        drv.Context.synchronize()

        # record the stop time and report elapsed
        _search_stop_time = time()
        if verbose:
            print(f"completed in {_search_stop_time - _search_start_time:.3f} seconds")

        # generate the PeriodicResult
        self.periodic_result = PeriodicResult(
            linear_result=self.linear_result,
            period_array=self.periods,
            like_ratio_array=periodogram['like_ratio'],
            depth_array=periodogram['depth'],
            depth_variance_array=periodogram['var_depth'],
            duration_index_array=periodogram['duration_idx'],
            t0_index_array=periodogram['t0_idx']
        )

        return self.periodic_result

    def check_period(
            self, period,
            max_duration_fraction=0.12,
            min_star_mass=0.1, max_star_mass=1.0,
            min_star_radius=0.13, max_star_radius=3.5,
            ignore_astrophysics=False
    ):
        """
        Find the maximum likelihood ratio (vs. constant flux model), depth,
        depth variance, number of data points, t0 and duration for a given
        period.

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
        Tuple of len=6 containing the maximum likelihood ratio and the
        corresponding depth, depth variance, number of data points,
        t0 index and duration index for the input period.
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
        _idx_durations_in_range = np.where(
            (self.durations >= duration_min) & (self.durations <= duration_max)
        )[0]
        num_d_in_range = np.int32(_idx_durations_in_range.size)

        # return some null output if no valid durations
        if num_d_in_range == 0:
            return np.nan, np.nan, np.nan, -1, -1

        # find the indices and number of durations that are in range
        first_d_in_range_idx = np.int32(_idx_durations_in_range[0])
        last_d_in_range_idx = np.int32(_idx_durations_in_range[-1])

        # maximum possible number of transits
        _lc_baseline = self.lc.time[-1] - self.lc.time[0]
        max_transit_count = np.int32(np.floor(_lc_baseline / period) + 1)
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
        _tm_size = np.int32(self.transit_model.size)
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
        block_size_k2 = 1024, 1, 1  # the second and third dimensions should have size 1
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


def to_gpu(arr, dtype: np.dtype):
    """
    Convenience function for sending an array to the gpu with a specific data
    type.

    Parameters
    ----------
    arr: array-like
        The array to send to the gpu
    dtype: dtype
        The numpy data type to use

    Returns
    -------
    A gpuarray
    """
    return gpuarray.to_gpu(np.ascontiguousarray(arr, dtype=dtype))


def duration_grid(
        durations=None,
        min_duration=0.02,
        max_duration=1.0,
        log_step=1.1,
        verbose=True
):
    """
    Validate or generate the duration grid

    Parameters
    ----------
    durations : array-like, optional
        User-specified grid of durations in days. If not provided, the
        module computes a grid using the minimum and maximum durations
        and the log step.
    min_duration : float, optional
        Minimum transit duration to check in days, default 0.02.
    max_duration : float, optional
        Maximum transit duration to check in days, default 1.0.
    log_step : float, optional
        The log-spacing of the durations to be used if the duration grid
        is to be internally determined. Default 1.1.
    verbose : bool, optional
        If True (the default), reports various messages.

    Returns
    -------
    The validated duration grid
    """
    if durations is None:
        if log_step <= 1.0:
            raise ValueError("duration log step must be greater than 1.0")
        else:
            durations = [min_duration]
            while durations[-1] < max_duration:
                next_duration = durations[-1] * log_step
                if next_duration <= max_duration:
                    durations.append(next_duration)
                else:
                    break
            durations = np.asarray(durations)

    # sort the array (in case input is unsorted)
    _durations = np.sort(durations)

    # raise an error if the duration grid is too sparse
    if len(_durations) == 0:
        raise RuntimeError("The duration grid is empty")

    if verbose:
        # report the duration grid size
        print(f"{len(_durations)} durations, "
              f"{_durations[0]:.2f} -> {_durations[-1]:.2f} days")

    return _durations


def period_grid(
        epoch_baseline, min_period=0.0, max_period=np.inf,
        n_transits_min=2, R_star=1.0, M_star=1.0, oversampling_factor=3
):
    """
    Generates the optimal period grid.
    Grabbed this nice code from TLS. Thanks Hippke and Heller!

    Original copyright for this code belongs to Michael Hippke, it was
    published under an MIT license. Some modifications made by L.C. Smith.

    Parameters
    ----------
    epoch_baseline : float
        The length of the light curve in days.
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
    R_star : float, optional
        The stellar radius (in solar radii) to use, default 1.0.
    M_star : float, optional
        The stellar mass (in solar masses) to use, default 1.0.
    oversampling_factor : int, optional
        Oversample the period grid by this factor, default 3. Increasing
        this improves detection efficiency but at a higher computational
        cost.

    Returns
    -------
    P_days : ndarray
        Period grid in days
    """
    # unit conversions
    M_star *= Constants.solar_mass
    R_star *= Constants.solar_radius
    epoch_baseline = epoch_baseline * Constants.seconds_per_day

    # boundary conditions
    f_min = n_transits_min / epoch_baseline
    f_max = 1.0 / (2 * np.pi) * np.sqrt(Constants.G * M_star / (3 * R_star) ** 3)

    # Ofir et al. 2014 stuff
    A = (
            (2 * np.pi) ** (2.0 / 3)
            / np.pi
            * R_star
            / (Constants.G * M_star) ** (1.0 / 3)
            / (epoch_baseline * oversampling_factor)
    )
    C = f_min ** (1.0 / 3) - A / 3.0
    N_opt = (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3) * 3 / A

    X = np.arange(N_opt) + 1
    f_x = (A / 3 * X + C) ** 3
    P_x = 1 / f_x
    # convert period grid from seconds to days
    P_x /= Constants.seconds_per_day

    # check limits
    P_x = P_x[np.where((P_x >= min_period) & (P_x <= max_period))[0]]

    # ascending order
    P_x = np.sort(P_x)

    return P_x


def max_t14(star_radius, star_mass, period, upper_limit=0.12, small_planet=False):
    """
    Compute the maximum transit duration.
    No need to completely reinvent the wheel here, thanks Hippke and Heller!
    https://github.com/hippke/tls/blob/master/transitleastsquares/grid.py#L10

    Original copyright for this code belongs to Michael Hippke, it was
    published under an MIT license. Some modifications made by L.C. Smith.

    Parameters
    ----------
    star_radius : float
        Stellar radius in units of solar radii
    star_mass : float
        Stellar mass in units of solar masses
    period : float
        Period in units of days
    upper_limit : float, optional
        Maximum transit duration as a fraction of period (default 0.12)
    small_planet : bool, optional
        If True, uses the small planet assumption (i.e. the planet radius relative
        to the stellar radius is negligible). Otherwise, uses a 2* Jupiter radius
        planet (the default).

    Returns
    -------
    duration: float
        Maximum transit duration as a fraction of the period
    """
    # unit conversions
    period = period * Constants.seconds_per_day
    star_radius = Constants.solar_radius * star_radius
    star_mass = Constants.solar_mass * star_mass

    if small_planet:
        # small planet assumption
        radius = star_radius
    else:
        # planet size 2 R_jup
        radius = star_radius + 2 * Constants.jupiter_radius

    # pi * G * mass
    piGM = np.pi * Constants.G * star_mass
    # transit duration
    T14max = radius * (4 * period / piGM) ** (1 / 3)

    # convert to fractional
    result = T14max / period

    # impose upper limit
    if result > upper_limit:
        result = upper_limit

    return result


def concatenate_lightcurves(lc_list, resample_cadence=None):
    """
    Concatenate multiple LightCurve objects to produce a single LightCurve
    object. If LightCurves overlap in time space their trends will be lost.
    If this is the case, it's recommended that their trends are instead
    subtracted by the user before generating a new LightCurve instance.
    (Note: Any existing transit masks will be lost, it's recommended to
    remask again using the Transit objects.)

    Parameters
    ----------
    lc_list : array-like
        Array of LightCurve objects.
    resample_cadence : float, optional
        The cadence (in seconds) to use for resampling the light curve. By
        default, the cadence of the first listed light curve will be used.

    Returns
    -------
    A new LightCurve instance.
    """
    # input validation
    try:
        if not isinstance(lc_list[0], LightCurve):
            raise TypeError()
    except TypeError:
        raise TypeError("lc_list must be an array-like object containing LightCurves")
    except IndexError:
        raise IndexError("lc_list contains no items")

    # transit mask warning
    for lc in lc_list:
        if lc.transit_mask is not None:
            warnings.warn("Transit masks are removed on concatenation")

    # grab the data from the combined light curve arrays
    times = []
    fluxes = []
    errors = []
    tlims = []
    for lc in lc_list:
        times.append(lc.input_time)
        fluxes.append(lc.input_flux)
        errors.append(lc.input_flux_error)
        tlims.append((np.min(lc.input_time), np.max(lc.input_time)))

    # apply the error multiplier now if present
    for n, lc in enumerate(lc_list):
        if lc.error_multiplier is not None:
            errors[n] *= lc.error_multiplier

    # concatenate the light curve data
    times, fluxes, errors = map(np.concatenate, [times, fluxes, errors])

    # set a resampling cadence if one isn't given
    if resample_cadence is None:
        resample_cadence = lc_list[0].cadence * Constants.seconds_per_day

    # create a new LightCurve
    new_lc = LightCurve(
        times=times, fluxes=fluxes, flux_errors=errors,
        resample_cadence=resample_cadence
    )

    # now we must migrate any previously determined trends, but only do this
    # if they all have them
    if np.all([lc.offset_trend is not None for lc in lc_list]):
        # now check that the light curves don't overlap in time space
        _overlap = False
        for n, lim0 in enumerate(tlims[:-1]):
            for lim1 in tlims[n+1:]:
                if (lim1[1] >= lim0[0] >= lim1[0]
                        or lim1[1] >= lim0[1] >= lim1[0]):
                    _overlap = True

        # raise an error if they overlap
        if _overlap:
            raise RuntimeError(
                "LightCurve objects overlap, trend cannot be propagated"
            )
        # otherwise, carry on

        # extract the trends
        _time = []
        _trend = []
        for lc in lc_list:
            _time.append(lc.time)
            _trend.append(lc.offset_trend)
        _time, _trend = map(np.concatenate, [_time, _trend])
        # must be in chronological order
        _order = np.argsort(_time)
        _time, _trend = map(lambda arr: arr[_order], [_time, _trend])
        # now do some simple linear interpolation to obtain the new trend
        new_trend = np.interp(new_lc.time, _time, _trend, left=np.nan, right=np.nan)
        # apply the trend to the new light curve
        new_lc.offset_trend = new_trend

    # return the new light curve
    return new_lc


if __name__ == "__main__":
    raise RuntimeError("Don't try to run this module directly")
