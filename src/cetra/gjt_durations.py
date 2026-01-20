#!/usr/bin/env python3

# This code was written by Geert Jan Talens by provided by private
# communication. I'm including it here with his permission. It
# implements the extended duration limits of
# Talens G.J., et al., 2025, RNAAS, 9, 319
# https://ui.adsabs.harvard.edu/abs/2025RNAAS...9..319T
# It was slightly modified to fit into the cetra package structure, and
# to improve performance when called with scalar period values.


from typing import Union
from collections import namedtuple
import numpy as np
from astropy import constants

_G = constants.G.value  # m^3 kg^-1 s^-2
seconds_per_day = 24. * 60. * 60.  # s/day

StableOrbit = namedtuple('stable_orbit', ['axis', 'ecc'])
DurationLimits = namedtuple('duration_limits', ['short', 'long'])


def get_period_min(density_star: float,
                   min_separation: float = 3.
                   ) -> float:
    """ Compute the orbital period at which the planet-star separation is
        N times the stellar radius.

    Parameters
    ----------
    density_star: float
        The density of the star in g/cm^3.
    min_separation: float
        The minimum orbital separation in stellar radii (default: 3).

    Returns
    -------
    period_min: float
        The minimum period at which the separation requirement is met.

    """

    density_star = density_star / 1000. * 100.**3  # to kg/m^3

    period_min = np.sqrt(min_separation**3 * 3 * np.pi / _G / density_star)
    period_min /= seconds_per_day

    return period_min


def get_axis(density_star: float,
             period: Union[float, np.ndarray]
             ) -> Union[float, np.ndarray]:
    """ Compute the scaled semi-major axis using Kepler's 3rd law.

    Parameters
    ----------
    density_star: float
        The density of the star in g/cm^3.
    period: float or np.ndarray
        The orbital period in days.

    Returns
    -------
    axis: float or np.ndarray
        The semi-major axis in units of stellar radii.

    """

    density_star = density_star / 1000. * 100.**3  # to kg/m^3
    period_s = period * seconds_per_day

    factor = 3 * np.pi / (_G * period_s ** 2)
    axis = (density_star / factor) ** (1 / 3)

    return axis


def get_stable_orbits(axis: np.ndarray,
                      min_separation: float = 3.
                      ) -> StableOrbit:
    """ Compute the limiting values for the semi-major axis and eccentricty
        that still result in stable orbits.

    Parameters
    ----------
    axis: np.ndarray
        Semi-major axis values in units of stellar radii.
    min_separation: float
        The minimum orbital separation in stellar radii (default: 3).

    Returns
    -------
    stable_orbit: StableOrbit
        Named tuple containing the semi-major axis and eccentricty of the
        most eccentric stable orbit.

    """

    # Not all densities produce stable orbits at low periods.
    axis_stable = np.maximum(axis, min_separation)

    # Not all eccentricties produce stable orbits.
    # Stable orbits asymptotically approach ecc -> 1 as a/R -> inf.
    ecc_stable = (axis_stable - min_separation)/axis_stable

    stable_orbit = StableOrbit(axis=axis_stable, ecc=ecc_stable)

    return stable_orbit


def get_orbit_bounds(period_grid: np.ndarray,
                     density_min: float,
                     density_max: float,
                     min_separation: float = 3.
                     ) -> tuple[StableOrbit, StableOrbit]:
    """ Given an array of period values and a stellar density interval, compute
        the bounding semi-major axis and eccentricity values of the inner and
        outer stable orbits represented by the density interval.

    Parameters
    ----------
    period_grid: np.ndarray
        An array of orbital periods in days.
    density_min: float
        The lower bound on the density interval in g/cm^3.
    density_max: float
        The upper bound on the density interval in g/cm^3.
    min_separation: float
        The minimum orbital separation in stellar radii (default: 3).

    Returns
    -------
    inner_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        innermost stable orbit.
    outer_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        outermost stable orbit.

    """

    # Compute semi-major axis values from the stellar density bounds.
    axis_inner = get_axis(density_min, period_grid)
    axis_outer = get_axis(density_max, period_grid)

    # Check the range of semi-major axes and eccentricties that produce stable orbits.
    inner_orbit = get_stable_orbits(axis_inner, min_separation=min_separation)
    outer_orbit = get_stable_orbits(axis_outer, min_separation=min_separation)

    return inner_orbit, outer_orbit


def get_transit_duration(period: Union[float, np.ndarray],
                         axis: Union[float, np.ndarray],
                         planet_radius: Union[float, np.ndarray],
                         impact_param: Union[float, np.ndarray],
                         eccentricty: Union[float, np.ndarray],
                         arg_periastron: Union[float, np.ndarray]
                         ) -> Union[float, np.ndarray]:
    """ Compute the full transit duration (T14) for an eccentric orbit, using
        Equations 7, 14 and 16 from Winn (2010).

    Parameters
    ----------
    period: float or np.ndarray
        The orbital period value(s) in days.
    axis: float or np.ndarray
        The semi-major axis value(s) in stellar radii.
    planet_radius: float or np.ndarray
        The planet radius value(s) in stellar radii.
    impact_param: float or np.ndarray
        The impact parameter value(s).
    eccentricty: float or np.ndarray
        The orbital eccentricty value(s).
    arg_periastron: float or np.ndarray
        The argument of periastron in degrees.

    Returns
    -------
    transit_duration: float or np.ndarray
        The transit duration in days.

    """
    arg_periastron = np.deg2rad(arg_periastron)

    # Compute the eccentricity terms.
    x = np.sqrt(1 - eccentricty ** 2)
    y = 1 + eccentricty * np.sin(arg_periastron)

    alpha = x/y
    beta = x**2/y

    # Compute the transit duration.
    sin_sq = beta ** 2 * ((1 + planet_radius) ** 2 - impact_param ** 2) / (beta ** 2 * axis ** 2 - impact_param ** 2)
    transit_duration = alpha * period / np.pi * np.arcsin(np.sqrt(sin_sq))

    return transit_duration


def get_transit_duration_limits(period_grid: np.ndarray,
                                density_bounds: tuple[float, float],
                                stellar_radius_bounds: tuple[float, float],
                                planet_radius_bounds: tuple[float, float] = (0.01, 0.20),
                                impact_param_bounds: tuple[float, float] = (0.0, 0.9),
                                min_separation: float = 3.,
                                circular_orbits: bool = False,
                                ) -> tuple[DurationLimits, StableOrbit, StableOrbit]:
    """ Compute the minimum and maximum transit duration as a function of the
        orbital period, given possible bounds on the stellar density and
        valid stellar radii.

    Parameters
    ----------
    period_grid: np.ndarray
        An array of orbital periods in days.
    density_bounds: tuple
        The minimum and maximum density in g/cm^3.
    stellar_radius_bounds: tuple
        Representative stellar radii corresponding to the density bounds in
        solar radii.
    planet_radius_bounds: tuple
        The minumum and maximum planet radius in solar radii. Used when
        computing the shortest and longest transit duration respectively
        (default: (0.01, 0.20)).
    impact_param_bounds: tuple
        The minimum and maximum impact parameter. Used when computing the
        longest and shortest transit duration respectively (default:
        (0.0, 0.9)).
    min_separation: float
        The minimum orbital separation in stellar radii (default: 3).
    circular_orbits: bool
        If True compute the shortest and longest duration on circular orbits,
        i.e. forces the eccentricty to zero (default: False).

    Returns
    -------
    duration_limits: DurationLimits
        A named tuple containing the short and long duration limits.
    inner_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        innermost stable orbit.
    outer_orbit: StableOrbit
        A named tuple containing the semi-major axis and eccentricty of the
        outermost stable orbit.

    """

    # Compute the minimum period where the density range produces stable orbits.
    density_min, density_max = density_bounds
    stellar_radius_mindens, stellar_radius_maxdens = stellar_radius_bounds
    planet_radius_min, planet_radius_max = planet_radius_bounds
    impact_param_min, impact_param_max = impact_param_bounds

    # Compute the period limits.
    period_min = get_period_min(density_max)
    period_break = get_period_min(density_min)

    # The minimum stellar density goes from Pbreak to Pmin.
    # The corresponding stellar radius should decrease.
    # Since there is no exact way to evolve this, we simply jump to the other radius extreme.
    stellar_radius_mindens = np.where(period_grid > period_break, stellar_radius_mindens, stellar_radius_maxdens)

    # At fixed period, the maximum density produces the shortest transit.
    # It follows that the small planet radius needs to be scaled by the
    # corresponding stellar radius.
    radius_ratio_min = planet_radius_min/stellar_radius_maxdens
    radius_ratio_max = planet_radius_max/stellar_radius_mindens

    # Compute the scaled semi-major axis and eccentricty limits for the density values.
    result = get_orbit_bounds(period_grid, density_min, density_max, min_separation=min_separation)
    inner_orbit, outer_orbit = result

    if circular_orbits:
        outer_orbit = outer_orbit._replace(ecc=np.zeros_like(period_grid))
        inner_orbit = inner_orbit._replace(ecc=np.zeros_like(period_grid))

    # Compute the duration limits for the given parameter bounds.
    duration_short = get_transit_duration(period_grid, outer_orbit.axis, radius_ratio_min, impact_param_max, outer_orbit.ecc, 90.)
    duration_long = get_transit_duration(period_grid, inner_orbit.axis, radius_ratio_max, impact_param_min, inner_orbit.ecc, 270.)

    # Apply the minimum period threshold.
    mask = period_grid < period_min
    duration_short[mask] = np.nan
    duration_long[mask] = np.nan
    outer_orbit = outer_orbit._replace(axis=np.where(mask, np.nan, outer_orbit.axis))
    outer_orbit = outer_orbit._replace(ecc=np.where(mask, np.nan, outer_orbit.ecc))
    inner_orbit = inner_orbit._replace(axis=np.where(mask, np.nan, inner_orbit.axis))
    inner_orbit = inner_orbit._replace(ecc=np.where(mask, np.nan, inner_orbit.ecc))

    duration_limits = DurationLimits(short=duration_short, long=duration_long)

    return duration_limits, inner_orbit, outer_orbit
