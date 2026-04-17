#!/usr/bin/env python3

# Copyright (c) 2024 Leigh C. Smith - lsmith@ast.cam.ac.uk
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
import unittest
import numpy as np
from types import SimpleNamespace
from cetra import (LightCurve, TransitModel, TransitDetector, Transit,
                   LinearResult, PeriodicResult, concatenate_lightcurves)
from cetra.cetra import duration_grid, period_grid
from cetra.gjt_durations import (
    get_period_min, get_axis, get_stable_orbits,
    get_transit_duration_limits
)

seconds_per_day = 60. * 60. * 24.
_here = os.path.dirname(os.path.abspath(__file__))


def _flat_lc(n=600, cadence_s=1800.0):
    """Helper: create a flat synthetic light curve with no transits."""
    cadence = cadence_s / seconds_per_day
    times = np.arange(n, dtype=float) * cadence
    fluxes = np.ones(n)
    errors = np.full(n, 1e-4)
    return LightCurve(times, fluxes, errors, verbose=False)


# ---------------------------------------------------------------------------
# Pure-numpy utility tests (no GPU pipeline invoked)
# ---------------------------------------------------------------------------

class TestDurationGrid(unittest.TestCase):

    def test_auto_generation_monotonic(self):
        d = duration_grid(min_duration=0.05, max_duration=0.5, log_step=1.1, verbose=False)
        self.assertTrue(np.all(np.diff(d) > 0))

    def test_auto_generation_bounds(self):
        d = duration_grid(min_duration=0.05, max_duration=0.5, log_step=1.1, verbose=False)
        self.assertGreaterEqual(d[0], 0.05)
        self.assertLessEqual(d[-1], 0.5)

    def test_auto_generation_log_spacing(self):
        log_step = 1.2
        d = duration_grid(min_duration=0.1, max_duration=1.0, log_step=log_step, verbose=False)
        ratios = d[1:] / d[:-1]
        np.testing.assert_allclose(ratios, log_step, rtol=1e-10)

    def test_invalid_log_step_one_raises(self):
        with self.assertRaises(ValueError):
            duration_grid(log_step=1.0, verbose=False)

    def test_invalid_log_step_below_one_raises(self):
        with self.assertRaises(ValueError):
            duration_grid(log_step=0.9, verbose=False)

    def test_user_provided_sorted(self):
        user = np.array([0.5, 0.1, 0.3])
        d = duration_grid(durations=user, verbose=False)
        np.testing.assert_array_equal(d, np.sort(user))

    def test_empty_grid_raises(self):
        with self.assertRaises(RuntimeError):
            duration_grid(durations=np.array([]), verbose=False)

    def test_single_element_grid(self):
        d = duration_grid(durations=np.array([0.25]), verbose=False)
        self.assertEqual(len(d), 1)
        self.assertAlmostEqual(d[0], 0.25)

    def test_min_equals_max_gives_one_element(self):
        d = duration_grid(min_duration=0.1, max_duration=0.1, log_step=1.1, verbose=False)
        self.assertEqual(len(d), 1)


class TestPeriodGrid(unittest.TestCase):

    def test_ascending_order(self):
        p = period_grid(epoch_baseline=30.0)
        self.assertTrue(np.all(np.diff(p) > 0))

    def test_max_period_respects_n_transits_min(self):
        baseline = 30.0
        n = 2
        p = period_grid(epoch_baseline=baseline, n_transits_min=n)
        self.assertLessEqual(p[-1], baseline / n + 1e-6)

    def test_min_period_limit(self):
        p = period_grid(epoch_baseline=30.0, min_period=5.0)
        self.assertGreaterEqual(p[0], 5.0)

    def test_max_period_limit(self):
        p = period_grid(epoch_baseline=30.0, max_period=10.0)
        self.assertLessEqual(p[-1], 10.0 + 1e-6)

    def test_longer_baseline_more_periods(self):
        p1 = period_grid(epoch_baseline=30.0)
        p2 = period_grid(epoch_baseline=90.0)
        self.assertGreater(len(p2), len(p1))

    def test_oversampling_effect(self):
        p1 = period_grid(epoch_baseline=30.0, oversampling_factor=1)
        p3 = period_grid(epoch_baseline=30.0, oversampling_factor=3)
        self.assertGreater(len(p3), len(p1))


class TestGJTDurations(unittest.TestCase):

    def test_period_min_decreases_with_density(self):
        p_low = get_period_min(0.5)
        p_high = get_period_min(5.0)
        self.assertGreater(p_low, p_high)

    def test_period_min_positive(self):
        self.assertGreater(get_period_min(1.41), 0.0)

    def test_get_axis_scalar_solar(self):
        # Solar density star (1.41 g/cm^3), 1-year orbit ~ 215 Rsun
        axis = get_axis(1.41, 365.25)
        self.assertGreater(axis, 100.0)
        self.assertLess(axis, 500.0)

    def test_get_axis_array_monotonic(self):
        periods = np.array([1.0, 10.0, 100.0])
        axes = get_axis(1.41, periods)
        self.assertEqual(len(axes), 3)
        self.assertTrue(np.all(np.diff(axes) > 0), "longer period should give larger axis")

    def test_get_stable_orbits_ecc_range(self):
        axes = np.array([5.0, 10.0, 50.0])
        result = get_stable_orbits(axes)
        self.assertTrue(np.all(result.ecc >= 0.0))
        self.assertTrue(np.all(result.ecc < 1.0))

    def test_get_stable_orbits_enforces_min_separation(self):
        axes = np.array([1.0, 2.0, 10.0])
        result = get_stable_orbits(axes, min_separation=3.0)
        self.assertTrue(np.all(result.axis >= 3.0))

    def test_transit_duration_limits_shape(self):
        periods = np.linspace(1.0, 30.0, 50)
        limits, _, _ = get_transit_duration_limits(
            periods,
            density_bounds=(0.5, 5.0),
            stellar_radius_bounds=(3.5, 0.13),
        )
        self.assertEqual(len(limits.short), 50)
        self.assertEqual(len(limits.long), 50)

    def test_transit_duration_limits_short_less_than_long(self):
        periods = np.linspace(2.0, 30.0, 50)
        limits, _, _ = get_transit_duration_limits(
            periods,
            density_bounds=(0.5, 5.0),
            stellar_radius_bounds=(3.5, 0.13),
        )
        finite = np.isfinite(limits.short) & np.isfinite(limits.long)
        self.assertTrue(np.any(finite), "expected some finite duration limits")
        self.assertTrue(np.all(limits.short[finite] < limits.long[finite]))

    def test_transit_duration_limits_nan_at_short_periods(self):
        periods = np.array([0.01, 0.1, 1.0, 10.0])
        limits, _, _ = get_transit_duration_limits(
            periods,
            density_bounds=(0.5, 5.0),
            stellar_radius_bounds=(3.5, 0.13),
        )
        self.assertTrue(np.any(np.isnan(limits.short)))

    def test_transit_duration_limits_circular_flag(self):
        periods = np.linspace(3.0, 30.0, 20)
        limits_ecc, _, _ = get_transit_duration_limits(
            periods,
            density_bounds=(0.5, 5.0),
            stellar_radius_bounds=(3.5, 0.13),
            circular_orbits=False,
        )
        limits_circ, _, _ = get_transit_duration_limits(
            periods,
            density_bounds=(0.5, 5.0),
            stellar_radius_bounds=(3.5, 0.13),
            circular_orbits=True,
        )
        finite = np.isfinite(limits_ecc.long) & np.isfinite(limits_circ.long)
        # Eccentric orbits allow longer transits than circular
        self.assertTrue(np.any(limits_ecc.long[finite] >= limits_circ.long[finite]))


class TestTransit(unittest.TestCase):

    def test_creation(self):
        tr = Transit(t0=5.0, duration=0.1, depth=0.001, depth_error=0.0001)
        self.assertEqual(tr.t0, 5.0)
        self.assertEqual(tr.duration, 0.1)

    def test_period_defaults_to_none(self):
        tr = Transit(t0=5.0, duration=0.1, depth=0.001, depth_error=0.0001)
        self.assertIsNone(tr.period)

    def test_period_set_when_provided(self):
        tr = Transit(t0=5.0, duration=0.1, depth=0.001, depth_error=0.0001, period=3.0)
        self.assertEqual(tr.period, 3.0)

    def test_repr_contains_expected_fields(self):
        tr = Transit(t0=1.0, duration=0.1, depth=0.01, depth_error=0.001)
        r = repr(tr)
        for field in ('t0', 'duration', 'depth', 'SNR'):
            self.assertIn(field, r)

    def test_repr_snr_value(self):
        tr = Transit(t0=1.0, duration=0.1, depth=0.01, depth_error=0.001)
        # SNR = depth / depth_error = 10.0
        self.assertIn('10.00', repr(tr))

    def test_copy_is_independent(self):
        tr = Transit(t0=1.0, duration=0.1, depth=0.01, depth_error=0.001)
        tr2 = tr.copy()
        tr2.t0 = 999.0
        self.assertEqual(tr.t0, 1.0)


# ---------------------------------------------------------------------------
# GPU-dependent tests
# ---------------------------------------------------------------------------

class TestLightCurveValidation(unittest.TestCase):
    """Test LightCurve input validation; errors are raised before GPU calls."""

    def _valid(self, n=300):
        t = np.linspace(0, 10, n)
        f = np.ones(n)
        e = np.full(n, 1e-4)
        return t, f, e

    def test_mismatched_flux_length(self):
        t, f, e = self._valid()
        with self.assertRaises(RuntimeError):
            LightCurve(t, f[:-1], e, verbose=False)

    def test_mismatched_error_length(self):
        t, f, e = self._valid()
        with self.assertRaises(RuntimeError):
            LightCurve(t, f, e[:-1], verbose=False)

    def test_nan_in_times(self):
        t, f, e = self._valid()
        t[10] = np.nan
        with self.assertRaises(ValueError):
            LightCurve(t, f, e, verbose=False)

    def test_nan_in_errors(self):
        t, f, e = self._valid()
        e[10] = np.nan
        with self.assertRaises(ValueError):
            LightCurve(t, f, e, verbose=False)

    def test_inf_in_times(self):
        t, f, e = self._valid()
        t[10] = np.inf
        with self.assertRaises(ValueError):
            LightCurve(t, f, e, verbose=False)

    def test_nan_flux_with_finite_error(self):
        t, f, e = self._valid()
        f[10] = np.nan
        with self.assertRaises(ValueError):
            LightCurve(t, f, e, verbose=False)

    def test_nan_flux_with_inf_error_is_valid(self):
        t, f, e = self._valid()
        f[10] = np.nan
        e[10] = np.inf
        lc = LightCurve(t, f, e, verbose=False)
        self.assertIsNotNone(lc)

    def test_negative_flux(self):
        t, f, e = self._valid()
        f[10] = -0.001
        with self.assertRaises(ValueError):
            LightCurve(t, f, e, verbose=False)

    def test_zero_flux_error(self):
        t, f, e = self._valid()
        e[10] = 0.0
        with self.assertRaises(ValueError):
            LightCurve(t, f, e, verbose=False)

    def test_negative_flux_error(self):
        t, f, e = self._valid()
        e[10] = -1e-5
        with self.assertRaises(ValueError):
            LightCurve(t, f, e, verbose=False)

    def test_zero_epoch_baseline(self):
        t = np.ones(10)
        f = np.ones(10)
        e = np.full(10, 1e-4)
        with self.assertRaises(RuntimeError):
            LightCurve(t, f, e, verbose=False)

    def test_unnormalized_flux_warning(self):
        t, f, e = self._valid()
        f = f * 1000.0
        with self.assertWarns(UserWarning):
            LightCurve(t, f, e, verbose=False)


class TestLightCurveAttributes(unittest.TestCase):

    def setUp(self):
        n = 600
        cadence = 1800.0 / seconds_per_day
        self.times = np.arange(n, dtype=float) * cadence
        self.fluxes = np.ones(n)
        self.errors = np.full(n, 1e-4)
        self.lc = LightCurve(self.times, self.fluxes, self.errors, verbose=False)

    def test_cadence_detection(self):
        expected = 1800.0 / seconds_per_day
        self.assertAlmostEqual(self.lc.cadence, expected, places=8)

    def test_chronological_sort(self):
        idx = np.random.permutation(len(self.times))
        lc = LightCurve(self.times[idx], self.fluxes[idx], self.errors[idx], verbose=False)
        self.assertTrue(np.all(np.diff(lc.input_time) > 0))

    def test_offset_time_starts_at_zero(self):
        self.assertEqual(self.lc.offset_time[0], 0.0)

    def test_flux_weight(self):
        finite = np.isfinite(self.lc.flux_error)
        expected = 1.0 / self.lc.flux_error[finite] ** 2
        np.testing.assert_allclose(self.lc.flux_weight[finite], expected)

    def test_offset_flux(self):
        finite = np.isfinite(self.lc.flux)
        np.testing.assert_allclose(
            self.lc.offset_flux[finite], 1.0 - self.lc.flux[finite]
        )

    def test_flat_loglike_is_finite(self):
        self.assertTrue(np.isfinite(self.lc.flat_loglike))

    def test_size_matches_resampled_length(self):
        self.assertEqual(self.lc.size, len(self.lc.time))

    def test_reference_time_is_first_time(self):
        self.assertEqual(self.lc.reference_time, self.lc.time[0])

    def test_copy_is_independent(self):
        lc2 = self.lc.copy()
        lc2.flux[0] = 999.0
        self.assertNotEqual(self.lc.flux[0], 999.0)


class TestLightCurvePad(unittest.TestCase):

    def setUp(self):
        self.lc = _flat_lc()

    def test_size_after_padding(self):
        original_size = self.lc.size
        self.lc.pad(10, 20, verbose=False)
        self.assertEqual(self.lc.size, original_size + 30)

    def test_prepend_nan_flux(self):
        self.lc.pad(5, 0, verbose=False)
        self.assertTrue(np.all(np.isnan(self.lc.flux[:5])))

    def test_prepend_inf_error(self):
        self.lc.pad(5, 0, verbose=False)
        self.assertTrue(np.all(np.isinf(self.lc.flux_error[:5])))

    def test_append_nan_flux(self):
        original_size = self.lc.size
        self.lc.pad(0, 5, verbose=False)
        self.assertTrue(np.all(np.isnan(self.lc.flux[original_size:])))

    def test_append_inf_error(self):
        original_size = self.lc.size
        self.lc.pad(0, 5, verbose=False)
        self.assertTrue(np.all(np.isinf(self.lc.flux_error[original_size:])))

    def test_offset_time_starts_at_zero_after_padding(self):
        self.lc.pad(10, 10, verbose=False)
        self.assertEqual(self.lc.offset_time[0], 0.0)

    def test_transit_mask_warning(self):
        tr = Transit(t0=self.lc.time[300], duration=0.1, depth=0.01, depth_error=0.001)
        self.lc.mask_transit(tr)
        with self.assertWarns(UserWarning):
            self.lc.pad(5, 5, verbose=False)


class TestLightCurveMaskTransit(unittest.TestCase):

    def setUp(self):
        self.lc = _flat_lc()
        mid_time = self.lc.time[len(self.lc.time) // 2]
        self.tr = Transit(t0=mid_time, duration=0.1, depth=0.001, depth_error=0.0001)

    def test_transit_mask_none_initially(self):
        self.assertIsNone(self.lc.transit_mask)

    def test_in_transit_region_masked(self):
        self.lc.mask_transit(self.tr)
        self.assertTrue(np.any(self.lc.transit_mask))

    def test_out_of_transit_not_masked(self):
        self.lc.mask_transit(self.tr)
        self.assertTrue(np.any(~self.lc.transit_mask))

    def test_mask_accumulation(self):
        t1 = Transit(t0=self.lc.time[100], duration=0.1, depth=0.001, depth_error=1e-4)
        t2 = Transit(t0=self.lc.time[450], duration=0.1, depth=0.001, depth_error=1e-4)
        self.lc.mask_transit(t1)
        count1 = np.sum(self.lc.transit_mask)
        self.lc.mask_transit(t2)
        count2 = np.sum(self.lc.transit_mask)
        self.assertGreater(count2, count1)

    def test_return_mask_shape(self):
        mask = self.lc.mask_transit(self.tr, return_mask=True)
        self.assertIsNotNone(mask)
        self.assertEqual(len(mask), self.lc.size)

    def test_duration_multiplier_widens_mask(self):
        mask1 = self.lc.mask_transit(self.tr, return_mask=True)
        lc2 = self.lc.copy()
        mask2 = lc2.mask_transit(self.tr, duration_multiplier=2.0, return_mask=True)
        self.assertGreater(np.sum(mask2), np.sum(mask1))

    def test_periodic_transit_masks_multiple_windows(self):
        # 30-day LC with 5-day period → ~6 transit windows
        period = 5.0
        ptr = Transit(
            t0=self.lc.time[50], duration=0.05, depth=0.001,
            depth_error=1e-4, period=period
        )
        mask = self.lc.mask_transit(ptr, return_mask=True)
        # Count distinct masked windows via False→True transitions
        n_windows = np.sum(np.diff(mask.astype(int)) == 1)
        self.assertGreater(n_windows, 1)


class TestTransitModel(unittest.TestCase):

    def test_b32_loads(self):
        tm = TransitModel('b32', verbose=False)
        self.assertEqual(tm.size, 1024)

    def test_b93_loads(self):
        tm = TransitModel('b93', verbose=False)
        self.assertIsNotNone(tm.model)

    def test_b99_loads(self):
        tm = TransitModel('b99', verbose=False)
        self.assertIsNotNone(tm.model)

    def test_invalid_string_raises(self):
        with self.assertRaises(RuntimeError):
            TransitModel('b50', verbose=False)

    def test_array_input(self):
        model = np.ones(2000)
        model[900:1100] = 0.9
        tm = TransitModel(model, verbose=False)
        self.assertIsNotNone(tm.model)

    def test_nan_in_array_raises(self):
        model = np.ones(1000)
        model[500] = np.nan
        with self.assertRaises(ValueError):
            TransitModel(model, verbose=False)

    def test_inf_in_array_raises(self):
        model = np.ones(1000)
        model[500] = np.inf
        with self.assertRaises(ValueError):
            TransitModel(model, verbose=False)

    def test_custom_downsample_size(self):
        tm = TransitModel('b32', downsamples=512, verbose=False)
        self.assertEqual(tm.size, 512)

    def test_offset_model_equals_one_minus_model(self):
        tm = TransitModel('b32', verbose=False)
        np.testing.assert_allclose(tm.offset_model, 1.0 - tm.model)

    def test_nn_error_within_expected_range(self):
        tm = TransitModel('b32', verbose=False)
        max_err, mean_err = tm.nn_error()
        self.assertLess(max_err, 0.02)  # < 2% with 1024 samples

    def test_get_model_lc_output_length(self):
        tm = TransitModel('b32', verbose=False)
        lc = _flat_lc()
        tr = Transit(t0=lc.time[300], duration=0.1, depth=0.001, depth_error=1e-4)
        phases, model_flux = tm.get_model_lc(lc.time, tr)
        self.assertEqual(len(phases), len(lc.time))
        self.assertEqual(len(model_flux), len(lc.time))

    def test_get_model_lc_flux_at_most_one(self):
        tm = TransitModel('b32', verbose=False)
        lc = _flat_lc()
        tr = Transit(t0=lc.time[300], duration=0.1, depth=0.001, depth_error=1e-4)
        _, model_flux = tm.get_model_lc(lc.time, tr)
        self.assertTrue(np.all(model_flux <= 1.0 + 1e-9))

    def test_get_model_lc_periodic(self):
        tm = TransitModel('b32', verbose=False)
        lc = _flat_lc()
        tr = Transit(t0=lc.time[100], duration=0.1, depth=0.001, depth_error=1e-4, period=5.0)
        phases, model_flux = tm.get_model_lc(lc.time, tr)
        self.assertEqual(len(model_flux), len(lc.time))

    def test_copy_has_same_model(self):
        tm = TransitModel('b32', verbose=False)
        tm2 = tm.copy()
        np.testing.assert_array_equal(tm2.model, tm.model)
        self.assertEqual(tm2.size, tm.size)


class TestLinearResult(unittest.TestCase):
    """Unit tests for LinearResult using mock arrays."""

    def setUp(self):
        n_dur, n_t0 = 4, 80
        self.n_dur = n_dur
        self.n_t0 = n_t0
        self.dur_arr = np.array([0.05, 0.1, 0.2, 0.5])
        self.t0_arr = np.linspace(0, 20, n_t0)
        self.like_arr = np.random.rand(n_dur, n_t0).astype(np.float32)
        self.depth_arr = (np.random.rand(n_dur, n_t0) * 0.01).astype(np.float32)
        self.var_arr = np.ones((n_dur, n_t0), dtype=np.float32) * 1e-6
        self.lr = LinearResult(
            light_curve=None,
            transit_model=None,
            duration_array=self.dur_arr,
            t0_array=self.t0_arr,
            like_ratio_array=self.like_arr,
            depth_array=self.depth_arr,
            depth_variance_array=self.var_arr,
        )

    def test_get_params_t0(self):
        tr = self.lr.get_params(2, 30)
        self.assertEqual(tr.t0, self.t0_arr[30])

    def test_get_params_duration(self):
        tr = self.lr.get_params(1, 10)
        self.assertEqual(tr.duration, self.dur_arr[1])

    def test_get_params_depth(self):
        d_idx, t_idx = 0, 5
        tr = self.lr.get_params(d_idx, t_idx)
        self.assertAlmostEqual(tr.depth, float(self.depth_arr[d_idx, t_idx]))

    def test_get_params_depth_error(self):
        d_idx, t_idx = 0, 5
        tr = self.lr.get_params(d_idx, t_idx)
        self.assertAlmostEqual(tr.depth_error, np.sqrt(self.var_arr[d_idx, t_idx]))

    def test_get_params_period_is_none(self):
        tr = self.lr.get_params(0, 0)
        self.assertIsNone(tr.period)

    def test_get_params_returns_transit(self):
        tr = self.lr.get_params(0, 0)
        self.assertIsInstance(tr, Transit)

    def test_deprecation_get_max_likelihood_parameters(self):
        with self.assertWarns(DeprecationWarning):
            _ = self.lr.get_max_likelihood_parameters()

    def test_deprecation_get_max_snr_parameters(self):
        with self.assertWarns(DeprecationWarning):
            _ = self.lr.get_max_snr_parameters()


class TestPeriodicResult(unittest.TestCase):
    """Unit tests for PeriodicResult using mock arrays."""

    def setUp(self):
        n_dur, n_t0, n_per = 4, 80, 50
        dur_arr = np.array([0.05, 0.1, 0.2, 0.5])
        t0_arr = np.linspace(0, 20, n_t0)
        like_arr = np.random.rand(n_dur, n_t0).astype(np.float32)
        depth_arr = (np.random.rand(n_dur, n_t0) * 0.01).astype(np.float32)
        var_arr = np.ones((n_dur, n_t0), dtype=np.float32) * 1e-6
        mock_lc = SimpleNamespace(input_time_start=0.0)
        lr = LinearResult(
            light_curve=mock_lc,
            transit_model=None,
            duration_array=dur_arr,
            t0_array=t0_arr,
            like_ratio_array=like_arr,
            depth_array=depth_arr,
            depth_variance_array=var_arr,
        )
        self.pr = PeriodicResult(
            linear_result=lr,
            period_array=np.linspace(2.0, 15.0, n_per),
            like_ratio_array=np.random.rand(n_per).astype(np.float32),
            depth_array=(np.random.rand(n_per) * 0.01).astype(np.float32),
            depth_variance_array=np.ones(n_per, dtype=np.float32) * 1e-6,
            t0_index_array=np.random.randint(0, n_t0, n_per),
            duration_index_array=np.random.randint(0, n_dur, n_per),
        )

    def test_get_params_period(self):
        idx = 10
        tr = self.pr.get_params(idx)
        self.assertAlmostEqual(tr.period, float(self.pr.period_array[idx]))

    def test_get_params_returns_transit_with_period(self):
        tr = self.pr.get_params(5)
        self.assertIsInstance(tr, Transit)
        self.assertIsNotNone(tr.period)

    def test_deprecation_get_max_likelihood_parameters(self):
        with self.assertWarns(DeprecationWarning):
            _ = self.pr.get_max_likelihood_parameters()

    def test_deprecation_get_max_snr_parameters(self):
        with self.assertWarns(DeprecationWarning):
            _ = self.pr.get_max_snr_parameters()


class TestTransitDetectorConstructor(unittest.TestCase):

    def setUp(self):
        self.lc = _flat_lc()

    def test_invalid_light_curve_type_raises(self):
        with self.assertRaises(RuntimeError):
            TransitDetector("not a LightCurve", verbose=False)

    def test_invalid_transit_model_type_raises(self):
        with self.assertRaises(RuntimeError):
            TransitDetector(self.lc, transit_model="invalid", verbose=False)

    def test_default_transit_model_is_created(self):
        tced = TransitDetector(self.lc, verbose=False)
        self.assertIsInstance(tced.transit_model, TransitModel)

    def test_custom_transit_model_accepted(self):
        model = np.ones(2000)
        model[900:1100] = 0.9
        tm = TransitModel(model, verbose=False)
        tced = TransitDetector(self.lc, transit_model=tm, verbose=False)
        self.assertIs(tced.transit_model, tm)

    def test_durations_attribute_populated(self):
        tced = TransitDetector(self.lc, verbose=False)
        self.assertIsNotNone(tced.durations)
        self.assertGreater(len(tced.durations), 0)

    def test_t0_array_attribute_populated(self):
        tced = TransitDetector(self.lc, verbose=False)
        self.assertIsNotNone(tced.t0_array)
        self.assertGreater(len(tced.t0_array), 0)

    def test_lc_is_padded(self):
        original_size = self.lc.size
        tced = TransitDetector(self.lc, verbose=False)
        self.assertGreater(tced.lc.size, original_size)

    def test_custom_durations_respected(self):
        custom = np.array([0.05, 0.1, 0.5])
        tced = TransitDetector(self.lc, durations=custom, verbose=False)
        np.testing.assert_array_equal(tced.durations, custom)


class TestTransitDetectorPreSearchWarnings(unittest.TestCase):
    """Methods called before the required search should warn and return None."""

    def setUp(self):
        self.tced = TransitDetector(_flat_lc(), verbose=False)

    def test_get_max_likelihood_single_before_linear_search(self):
        with self.assertWarns(UserWarning):
            result = self.tced.get_max_likelihood_single_transit()
        self.assertIsNone(result)

    def test_get_max_snr_single_before_linear_search(self):
        with self.assertWarns(UserWarning):
            result = self.tced.get_max_snr_single_transit()
        self.assertIsNone(result)

    def test_period_search_before_linear_search(self):
        with self.assertWarns(UserWarning):
            result = self.tced.period_search(verbose=False)
        self.assertIsNone(result)

    def test_get_max_likelihood_periodic_before_period_search(self):
        with self.assertWarns(UserWarning):
            result = self.tced.get_max_likelihood_periodic_transit()
        self.assertIsNone(result)

    def test_get_max_snr_periodic_before_period_search(self):
        with self.assertWarns(UserWarning):
            result = self.tced.get_max_snr_periodic_transit()
        self.assertIsNone(result)

    def test_get_single_transits_threshold_before_linear_search(self):
        with self.assertWarns(UserWarning):
            result = self.tced.get_single_transits_above_snr_threshold(snr_threshold=5.0)
        self.assertIsNone(result)

    def test_get_periodic_transits_threshold_before_period_search(self):
        with self.assertWarns(UserWarning):
            result = self.tced.get_periodic_transits_above_snr_threshold(snr_threshold=5.0)
        self.assertIsNone(result)


class TestConcatenateLightCurves(unittest.TestCase):

    def test_invalid_element_type_raises(self):
        with self.assertRaises(TypeError):
            concatenate_lightcurves([1, 2, 3])

    def test_empty_list_raises(self):
        with self.assertRaises(IndexError):
            concatenate_lightcurves([])

    def test_two_lightcurves_combined(self):
        lc1 = _flat_lc(n=300)
        cadence = lc1.cadence
        times2 = lc1.input_time[-1] + cadence + np.arange(300) * cadence
        lc2 = LightCurve(times2, np.ones(300), np.full(300, 1e-4), verbose=False)
        combined = concatenate_lightcurves([lc1, lc2])
        self.assertGreater(combined.input_epoch_baseline, lc1.input_epoch_baseline)

    def test_single_lc_roundtrip_baseline(self):
        lc = _flat_lc()
        combined = concatenate_lightcurves([lc])
        self.assertAlmostEqual(
            combined.input_epoch_baseline, lc.input_epoch_baseline, places=5
        )

    def test_transit_mask_warning(self):
        lc1 = _flat_lc()
        tr = Transit(t0=lc1.time[100], duration=0.1, depth=0.001, depth_error=1e-4)
        lc1.mask_transit(tr)
        lc2 = _flat_lc()
        with self.assertWarns(UserWarning):
            concatenate_lightcurves([lc1, lc2])


# ---------------------------------------------------------------------------
# Integration tests using the synthetic test package
# ---------------------------------------------------------------------------

class TransitDetectorIntegrationTestCase(unittest.TestCase):

    def setUp(self):
        self.test_data = np.load(os.path.join(_here, 'test_package.npz'))

    def _make_lc(self, verbose=False):
        return LightCurve(
            times=self.test_data["times"],
            fluxes=self.test_data["fluxes"],
            flux_errors=self.test_data["errors"],
            verbose=verbose
        )

    def _make_detector(self, verbose=False):
        return TransitDetector(self._make_lc(), verbose=verbose)

    def test_lightcurve_cadence(self):
        lc = self._make_lc(verbose=True)
        self.assertEqual(lc.cadence, self.test_data["cadence"])

    def test_linear_search(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=True)
        transit = tced.get_max_likelihood_single_transit()

        t0s = np.arange(self.test_data["t0"], np.max(self.test_data["times"]),
                        self.test_data["period"])
        idx_found = np.argmin(np.abs(transit.t0 - t0s))
        time_diff = np.abs(t0s[idx_found] - transit.t0)
        self.assertLess(time_diff, tced.t0_stride_length)

        idx_best_true_dur = np.argmin(np.abs(tced.durations - self.test_data["duration"]))
        best_duration = tced.durations[idx_best_true_dur]
        self.assertEqual(best_duration, transit.duration)

        self.assertAlmostEqual(transit.depth, self.test_data["depth"], places=3)

    def test_periodic_search(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=False)
        _ = tced.period_search(verbose=True)
        transit = tced.get_max_likelihood_periodic_transit()

        idx_best_true_per = np.argmin(np.abs(tced.periods - self.test_data["period"]))
        best_period = tced.periods[idx_best_true_per]
        self.assertEqual(best_period, transit.period)

        self.assertAlmostEqual(transit.t0, self.test_data["t0"], places=2)
        self.assertAlmostEqual(transit.duration, self.test_data["duration"], places=2)
        self.assertAlmostEqual(transit.depth, self.test_data["depth"], places=3)

    def test_linear_result_array_shapes(self):
        tced = self._make_detector()
        lr = tced.linear_search(verbose=False)
        expected = (tced.duration_count, tced.num_t0_strides)
        self.assertEqual(lr.like_ratio_array.shape, expected)
        self.assertEqual(lr.depth_array.shape, expected)
        self.assertEqual(lr.depth_variance_array.shape, expected)

    def test_max_snr_single_transit_finds_injection(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=False)
        transit = tced.get_max_snr_single_transit()
        t0s = np.arange(self.test_data["t0"], np.max(self.test_data["times"]),
                        self.test_data["period"])
        time_diff = np.min(np.abs(transit.t0 - t0s))
        self.assertLess(time_diff, tced.t0_stride_length)

    def test_get_single_transits_above_threshold_returns_list(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=False)
        transits = tced.get_single_transits_above_snr_threshold(snr_threshold=5.0)
        self.assertIsInstance(transits, list)
        self.assertGreater(len(transits), 0)
        self.assertIsInstance(transits[0], Transit)

    def test_get_single_transits_impossible_threshold_empty(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=False)
        transits = tced.get_single_transits_above_snr_threshold(snr_threshold=1e10)
        self.assertEqual(len(transits), 0)

    def test_periodic_result_array_shapes(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=False)
        ptr = tced.period_search(verbose=False)
        n = len(tced.periods)
        self.assertEqual(len(ptr.like_ratio_array), n)
        self.assertEqual(len(ptr.depth_array), n)
        self.assertEqual(len(ptr.duration_index_array), n)
        self.assertEqual(len(ptr.t0_index_array), n)

    def test_get_periodic_transits_above_threshold_returns_list(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=False)
        _ = tced.period_search(verbose=False)
        transits = tced.get_periodic_transits_above_snr_threshold(snr_threshold=5.0)
        self.assertIsInstance(transits, list)
        self.assertGreater(len(transits), 0)
        self.assertIsInstance(transits[0], Transit)
        self.assertIsNotNone(transits[0].period)

    def test_max_snr_periodic_transit_finds_injection(self):
        tced = self._make_detector()
        _ = tced.linear_search(verbose=False)
        _ = tced.period_search(verbose=False)
        transit = tced.get_max_snr_periodic_transit()
        idx_best = np.argmin(np.abs(tced.periods - self.test_data["period"]))
        self.assertEqual(transit.period, tced.periods[idx_best])


if __name__ == '__main__':
    unittest.main()
