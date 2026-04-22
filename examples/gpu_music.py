#!/usr/bin/env python3
"""
gpu_music.py — Play a tune via GPU coil whine.

The period search in CETRA iterates over periods one by one in a Python
for-loop, firing a CUDA kernel each iteration. The GPU's coil whine
frequency tracks the kernel-launch rate. By sleeping between kernel
launches we throttle the rate to match musical note frequencies.
This works on the RTX A5000 in my home desktop, ymmv.

Usage:
    python gpu_music.py
    python gpu_music.py twinkle
    python gpu_music.py imperial --bpm 80
"""

import time
import numpy as np
import pycuda.driver as drv
from cetra import LightCurve, TransitModel, TransitDetector

SECONDS_PER_DAY = 86400.0

# ── note frequencies (equal temperament, A4 = 440 Hz) ────────────
#
# PITCH_SCALE controls the mapping:
#   kernel_launches_per_second = target_audio_hz * PITCH_SCALE
#
# At PITCH_SCALE=1 the kernel launch rate equals the audio frequency in Hz
# (e.g. 440 launches/s → A4 = 440 Hz).
#
# If you hear a different pitch to what's expected, adjust PITCH_SCALE.
# e.g. if 1000 launches/s sounds like 1 Hz, set PITCH_SCALE = 1000.
PITCH_SCALE = 1

NOTE_FREQ = {
    # octave 3
    'C3': 130.81, 'Cs3': 138.59, 'D3': 146.83, 'Ds3': 155.56,
    'E3': 164.81, 'F3': 174.61, 'Fs3': 185.00,
    'G3': 196.00, 'Gs3': 207.65, 'A3': 220.00, 'As3': 233.08,
    'B3': 246.94,
    # octave 4 (middle)
    'C4': 261.63, 'Cs4': 277.18, 'D4': 293.66, 'Ds4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'Fs4': 369.99,
    'G4': 392.00, 'Gs4': 415.30, 'A4': 440.00, 'As4': 466.16,
    'B4': 493.88,
    # octave 5
    'C5': 523.25, 'Cs5': 554.37, 'D5': 587.33, 'Ds5': 622.25,
    'E5': 659.26, 'F5': 698.46, 'Fs5': 739.99,
    'G5': 783.99, 'Gs5': 830.61, 'A5': 880.00, 'As5': 932.33,
    'B5': 987.77,
    # octave 6
    'C6': 1046.50, 'Cs6': 1108.73, 'D6': 1174.66, 'Ds6': 1244.51,
    'E6': 1318.51, 'F6': 1396.91, 'Fs6': 1479.98,
    'G6': 1567.98, 'Gs6': 1661.22, 'A6': 1760.00, 'As6': 1864.66,
    'B6': 1975.53,
    # silence
    '_': 0.0,
}


# ── setup ─────────────────────────────────────────────────────────

def make_detector(n=5000, cadence_s=1800.0):
    """Flat LightCurve → TransitDetector with linear_search done.

    n controls the light curve length and therefore the kernel workload:
    larger n → heavier GPU kernel per check_period call → louder coil whine.
    """
    cadence = cadence_s / SECONDS_PER_DAY
    times = np.arange(n, dtype=float) * cadence
    lc = LightCurve(times, np.ones(n), np.full(n, 1e-4), verbose=False)
    # box model (all-zero array) — simplest possible transit shape
    tm = TransitModel(np.zeros(4096), verbose=False)
    tced = TransitDetector(lc, transit_model=tm, verbose=False)
    tced.linear_search(verbose=False)
    return tced


def _check_args(tced):
    """Return (period, min_d, max_d) safe for repeated check_period calls."""
    period = 2.0
    min_d = float(tced.durations[0])
    max_d = float(min(tced.durations[-1], period * 0.12))
    return period, min_d, max_d


# ── calibration ───────────────────────────────────────────────────

def calibrate(tced, n_calls=100):
    """
    Measure the natural (unthrottled) check_period call rate.

    Returns
    -------
    float
        Kernel launches per second at full GPU speed.
    """
    period, min_d, max_d = _check_args(tced)

    # warm-up
    for _ in range(10):
        tced.check_period(period, min_d, max_d)
    drv.Context.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_calls):
        tced.check_period(period, min_d, max_d)
    drv.Context.synchronize()

    elapsed = time.perf_counter() - t0
    return n_calls / elapsed


# ── playback ──────────────────────────────────────────────────────

def play_note(tced, audio_hz, duration_s):
    """
    Produce coil whine at audio_hz for duration_s seconds.

    One kernel launch per cycle: fires a single check_period call then
    sleeps for the remainder of the cycle.  This gives a clean discrete
    pulse at exactly the target frequency.

    audio_hz = 0 → silence (plain sleep).
    """
    if audio_hz == 0.0 or duration_s <= 0.0:
        time.sleep(duration_s)
        return

    period, min_d, max_d = _check_args(tced)

    launch_hz = audio_hz * PITCH_SCALE
    target_interval = 1.0 / launch_hz
    n_launches = max(1, round(duration_s * launch_hz))
    t_note_end = time.perf_counter() + duration_s

    for _ in range(n_launches):
        t_launch = time.perf_counter()
        tced.check_period(period, min_d, max_d)
        drv.Context.synchronize()
        sleep_t = target_interval - (time.perf_counter() - t_launch)
        if sleep_t > 1e-4:
            time.sleep(sleep_t)
        if time.perf_counter() >= t_note_end:
            break


def play_tune(tced, score, bpm=120, beat_value=1.0, gap_s=0.02):
    """
    Play a sequence of (note_name, beats) pairs.

    Parameters
    ----------
    tced : TransitDetector
    score : list of (str, float)
        Each element is a note name (see NOTE_FREQ) and its duration in beats.
    bpm : float
        Beats per minute.
    beat_value : float
        Duration of one 'beat' unit in quarter-note equivalents (default 1.0).
    gap_s : float
        Silent gap between notes to articulate them (seconds).
    """
    beat_s = 60.0 / bpm * beat_value

    max_rate = calibrate(tced, n_calls=50)
    max_audio_hz = max_rate / PITCH_SCALE
    kernel_ms = 1000.0 / max_rate
    print(f"\nNatural GPU rate:  {max_rate:.1f} launches/s  ({kernel_ms:.3f} ms/kernel)")
    print(f"Max playable note: {max_audio_hz:.1f} Hz  (PITCH_SCALE={PITCH_SCALE})\n")

    print(f"{'Note':<6} {'Freq':>8}  {'Dur':>6}  {'Achievable'}")
    print("-" * 40)

    for note, beats in score:
        freq = NOTE_FREQ.get(note, 0.0)
        dur = beats * beat_s
        ok = '✓' if (freq == 0 or freq <= max_audio_hz) else f'✗ (max {max_audio_hz:.0f} Hz)'
        print(f"{note:<6} {freq:>7.1f} Hz  {dur:>5.3f}s  {ok}")

        play_note(tced, freq, max(0.0, dur - gap_s))
        if gap_s > 0:
            time.sleep(gap_s)

    print("\nDone.")


# ── tunes ─────────────────────────────────────────────────────────

# Ode to Joy — Beethoven's 9th, 4th movement (first theme, bars 1–8)
ODE_TO_JOY = [
    # bar 1
    ('E5', 1), ('E5', 1), ('F5', 1), ('G5', 1),
    # bar 2
    ('G5', 1), ('F5', 1), ('E5', 1), ('D5', 1),
    # bar 3
    ('C5', 1), ('C5', 1), ('D5', 1), ('E5', 1),
    # bar 4
    ('E5', 1.5), ('D5', 0.5), ('D5', 2),
    # bar 5
    ('E5', 1), ('E5', 1), ('F5', 1), ('G5', 1),
    # bar 6
    ('G5', 1), ('F5', 1), ('E5', 1), ('D5', 1),
    # bar 7
    ('C5', 1), ('C5', 1), ('D5', 1), ('E5', 1),
    # bar 8
    ('D5', 1.5), ('C5', 0.5), ('C5', 2),
]

# Twinkle Twinkle Little Star
TWINKLE = [
    ('C5', 1), ('C5', 1), ('G5', 1), ('G5', 1),
    ('A5', 1), ('A5', 1), ('G5', 2),
    ('F5', 1), ('F5', 1), ('E5', 1), ('E5', 1),
    ('D5', 1), ('D5', 1), ('C5', 2),
    ('G5', 1), ('G5', 1), ('F5', 1), ('F5', 1),
    ('E5', 1), ('E5', 1), ('D5', 2),
    ('G5', 1), ('G5', 1), ('F5', 1), ('F5', 1),
    ('E5', 1), ('E5', 1), ('D5', 2),
    ('C5', 1), ('C5', 1), ('G5', 1), ('G5', 1),
    ('A5', 1), ('A5', 1), ('G5', 2),
    ('F5', 1), ('F5', 1), ('E5', 1), ('E5', 1),
    ('D5', 1), ('D5', 1), ('C5', 2),
]

# Imperial March — John Williams, Star Wars
IMPERIAL_MARCH = [
    ('G5', 1), ('G5', 1), ('G5', 1),
    ('Ds5', 0.75), ('As5', 0.25), ('G5', 1),
    ('Ds5', 0.75), ('As5', 0.25), ('G5', 2),
    ('D6', 1), ('D6', 1), ('D6', 1),
    ('Ds6', 0.75), ('As5', 0.25), ('Fs5', 1),
    ('Ds5', 0.75), ('As5', 0.25), ('G5', 2),
]


# ── main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    tunes = {
        'ode':      (ODE_TO_JOY,      120, 'Ode to Joy — Beethoven'),
        'twinkle':  (TWINKLE,         100, 'Twinkle Twinkle Little Star'),
        'imperial': (IMPERIAL_MARCH,  100, 'Imperial March — John Williams'),
    }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'tune', nargs='?', default='ode',
        choices=list(tunes), help='Which tune to play (default: ode)'
    )
    parser.add_argument(
        '--bpm', type=float, default=None,
        help='Override tempo in BPM'
    )
    parser.add_argument(
        '--scale', type=float, default=PITCH_SCALE,
        help=f'PITCH_SCALE: launches/s per Hz of audio (default {PITCH_SCALE}). '
             'Increase if the pitch sounds too high.'
    )
    args = parser.parse_args()

    PITCH_SCALE = args.scale
    score, default_bpm, title = tunes[args.tune]
    bpm = args.bpm if args.bpm is not None else default_bpm

    print("Setting up CETRA detector...")
    tced = make_detector()
    print(f"Linear search done.  Playing: {title}  ({bpm:.0f} BPM)")

    play_tune(tced, score, bpm=bpm)
