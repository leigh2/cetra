#!/usr/bin/env python3
"""
Benchmark the CETRA pipeline stages (resampling, linear search, period
search, and the check_period() kernel call in isolation) against a
synthetic light curve representative of in-operation use: a 1-year
baseline at 600s cadence, with an injected transit. Run after each
optimization change and compare timings against a previous run.
"""
import numpy as np
from time import perf_counter
import pycuda.driver as drv

from cetra.cetra import LightCurve, TransitDetector, TransitModel, Transit, Constants

CADENCE_S = 600.0
BASELINE_DAYS = 365.25
INJECTED_TRANSIT = Transit(t0=50.0, duration=0.12, depth=0.001, depth_error=np.nan, period=8.3)
FLUX_ERROR = 1e-4

N_REPEATS = 3
N_CHECK_PERIOD_CALLS = 500


def _timed(fn, *args, **kwargs):
    drv.Context.synchronize()
    t0 = perf_counter()
    result = fn(*args, **kwargs)
    drv.Context.synchronize()
    t1 = perf_counter()
    return result, t1 - t0


def make_one_year_lightcurve(seed=0):
    """1-year baseline, 600s cadence light curve with an injected transit,
    representative of an in-operation single-target light curve."""
    cadence = CADENCE_S / Constants.seconds_per_day
    n = int(round(BASELINE_DAYS / cadence))
    times = np.arange(n, dtype=float) * cadence

    rng = np.random.default_rng(seed)
    fluxes = np.ones(n) + rng.normal(0, FLUX_ERROR, size=n)
    flux_errors = np.full(n, FLUX_ERROR)

    model = TransitModel('b32', verbose=False)
    _, model_flux = model.get_model_lc(times, INJECTED_TRANSIT)
    fluxes *= model_flux

    return times, fluxes, flux_errors


def bench_resample(times, fluxes, errors):
    durations = []
    lc = None
    for _ in range(N_REPEATS):
        lc, dt = _timed(LightCurve, times, fluxes, errors, verbose=False)
        durations.append(dt)
    print(f"LightCurve resample ({lc.input_num_points} -> {lc.size} points): "
          f"{1000 * np.median(durations):.3f} ms (median of {N_REPEATS})")
    return lc


def bench_linear_search(lc):
    durations = []
    det = None
    for _ in range(N_REPEATS):
        det = TransitDetector(lc, verbose=False)
        _, dt = _timed(det.linear_search, verbose=False)
        durations.append(dt)
    print(f"linear_search ({det.duration_count} durations x "
          f"{det.num_t0_strides} t0 strides): "
          f"{1000 * np.median(durations):.3f} ms (median of {N_REPEATS})")
    return det


def bench_period_search(det):
    _, dt = _timed(det.period_search, verbose=False)
    print(f"period_search ({det.period_count} periods): "
          f"{1000 * dt:.3f} ms ({1000 * dt / det.period_count:.4f} ms/period)")


def bench_check_period(det):
    # sample periods/durations representative of a real period_search() call
    rng = np.random.default_rng(0)
    sample_idx = rng.integers(0, det.period_count, size=N_CHECK_PERIOD_CALLS)
    periods = det.periods[sample_idx]
    min_durations = det.min_durations[sample_idx]
    max_durations = det.max_durations[sample_idx]

    # warm up
    for p, dmin, dmax in zip(periods[:10], min_durations[:10], max_durations[:10]):
        det.check_period(float(p), dmin, dmax)

    drv.Context.synchronize()
    t0 = perf_counter()
    for p, dmin, dmax in zip(periods, min_durations, max_durations):
        det.check_period(float(p), dmin, dmax)
    drv.Context.synchronize()
    t1 = perf_counter()

    dt = t1 - t0
    print(f"check_period() inner loop ({N_CHECK_PERIOD_CALLS} calls): "
          f"{dt:.3f}s ({1000 * dt / N_CHECK_PERIOD_CALLS:.4f} ms/call)")


if __name__ == "__main__":
    times, fluxes, errors = make_one_year_lightcurve()
    lc = bench_resample(times, fluxes, errors)
    det = bench_linear_search(lc)
    bench_period_search(det)
    bench_check_period(det)
