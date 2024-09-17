import numpy as np
from pycuda import gpuarray
from scipy.interpolate import interp1d


class Constants(object):
    seconds_per_day = 86400  # s
    G = 6.67430e-11  # m**3 * kg**-1 * s**-2
    solar_radius = 6.957e+8  # m, IAU
    solar_mass = 1.9891e+30  # kg, IAU
    jupiter_radius = 7.1492e+7  # m, IAU, equatorial


def interpolate_model(input_model, target_samples, kind='linear'):
    """
    CHECKED
    Resample an input model using interpolation

    Parameters
    ----------
    input_model : array-like
        The input model
    target_samples : int
        The required number of samples
    kind : str, optional
        A scipy.interpolate.interp1d interpolation method.
        'linear' is the default.

    Returns
    -------
    Model with the required number of samples
    """
    i1d = interp1d(
        np.linspace(0, 1, len(input_model)), input_model,
        kind=kind, copy=True, bounds_error=False, fill_value=1.0,
        assume_sorted=True
    )
    return i1d, i1d(np.linspace(0, 1, target_samples))


def nn_model_error(samples, input_model):
    """
    CHECKED
    Return the maximum nearest-neighbour error in the model for a given
    model size.

    Parameters
    ----------
    samples : int
        The number of samples in the model
    input_model : array-like
        The input model

    Returns
    -------
    The maximum fractional error on a transit model of the input size
    """
    s_inter = (samples-1)*2 + 1
    _, m_all = interpolate_model(input_model, s_inter)  # todo check if should be '1 - '
    m_orig = m_all[0::2]
    m_inter = m_all[1::2]
    frac_diff = m_orig[1:] - m_inter
    return np.max(np.abs(frac_diff)), np.mean(np.abs(frac_diff))


def max_t14(star_radius, star_mass, period, upper_limit=0.12, small_planet=False):
    """
    Compute the maximum transit duration.
    No need to reinvent the wheel here, thanks Hippke and Heller!
    https://github.com/hippke/tls/blob/master/transitleastsquares/grid.py#L10

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
        to the stellar radius is negligible). Otherwise uses a 2* Jupiter radius
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


def get_bounds(array, index, value, index_offset=1):
    """
    Get bounds from a parameter array, being robust to edge cases

    Parameters
    ----------
    array : array-like
        Parameter array
    index : int
        Index of best fit value
    value : numeric
        Best fit value
    index_offset : int, optional
        How many array elements to offset to find the gap

    Returns
    -------
    Tuple of upper and lower bounds on value
    """

    # make sure array is sorted, but need to find the corresponding index in
    # sorted array
    sort_idx = np.argsort(array)
    new_idx = np.where(sort_idx == index)[0][0]
    _array = array[sort_idx]

    val_pm = []
    if new_idx >= index_offset:
        val_pm.append(_array[new_idx] - _array[new_idx - index_offset])
    if new_idx < (_array.size - index_offset):
        val_pm.append(_array[new_idx + index_offset] - _array[new_idx])
    # using the 0 and -1 indices means if one is missing it uses the other
    # instead, which will generally be fine
    return value - val_pm[0], value + val_pm[-1]


def duration_grid(min_duration=0.02, max_duration=1.0, log_step=1.1):
    """
    CHECKED
    Generate the duration grid.

    Parameters
    ----------
    min_duration : float, optional
        Minimum transit duration in days, default 0.02.
    max_duration : float
        Maximum transit duration in days, default 1.0.
    log_step : float, optional
        Log spacing of duration grid points, default 1.1.

    Returns
    -------
    Grids of durations in days.
    """
    # could maybe be made more elegant but not really necessary
    durations = [min_duration]
    while durations[-1] < max_duration:
        durations.append(durations[-1] * log_step)
    durations_days = np.asarray(durations)

    return durations_days


def period_grid(
        epoch_baseline, min_period=0.0, max_period=np.inf,
        n_transits_min=2, R_star=1.0, M_star=1.0, oversampling_factor=3
):
    """
    Generates the optimal period grid.
    Grabbed this nice code from TLS. Thanks again Hippke and Heller!

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


def to_gpu(arr, dtype):
    """
    Convenience function for sending an array to the gpu with a specific data
    type.

    Parameters
    ----------
    arr : array-like
        The array to send to the gpu
    dtype : dtype
        The numpy data type to use

    Returns
    -------
    A gpuarray
    """
    return gpuarray.to_gpu(np.ascontiguousarray(arr, dtype=dtype))


