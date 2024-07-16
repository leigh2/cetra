# CUDA Exoplanet Transit Recovery Algorithm - CETRA
Exoplanet transit detection with Nvidia CUDA GPU architecture.

## Overview

Transit detection tools are inspired in a large part by Transit Least Squares (TLS):<br>
https://github.com/hippke/tls<br>
https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract<br>
Though this implementation is robust against gaps in data, irregular cadence, and 
non-uniform uncertainties on light curve flux. It will also detect transits that 
aren't periodic in the input light curve (AKA monotransits).

The transit detection algorithm has three setup stages, followed by two main stages:

* * The first setup step resamples the input light curve such that the cadence
    is regularised and gaps are eliminated. New data points with no contributing 
    observed data are null, with infinite uncertainty. This resampling means the 
    light curve point roughly corresponding to any point in time can be determined 
    quickly and efficiently.
  * The second setup step produces a transit model for comparison to the resampled light 
    curve. Model points are evenly spaced in time, and the model is scaled such that the 
    maximum transit depth is 1. By default, the model has 1024 data points, though this 
    is configurable. The default number of points correspond to a maximum error of ~1% in
    flux, when taking the nearest model point for any point in time. The maximum error
    occurs where the flux gradient is largest (i.e. during ingress and egress).
  * The final setup step produces grids of durations and start times, and (if applicable) periods.
    From user inputs (or default values), or uses user-provided grids. The start time grid stride
    length is computed as some fraction of the minimum duration (1% by default but 
    user-configurable). Unless provided by the user, the duration and period grids are 
    computed using the same algorithms as the TLS module.
* The first main stage traverses the grid of durations and transit start times, and for 
each it calculates the maximum-likelihood transit depth, its variance, and its 
likelihood ratio relative to a constant flux model. The grids of results can be 
interrogated to obtain likely singular transits, and tools are provided for doing
this.
* The second main stages traverses a grid of periods, and for each it phase-folds the
grid of likelihood ratios from the previous stage, computing the maximum 
joint-likelihood ratio (again versus the constant flux model) for each. The corresponding
joint-likelihood, depth, depth variance, start time and duration are recorded for each period
grid point, from which a periodogram can be produced and periodic signals identified.
Again, tools are provided to perform this identification.

Further refinement of the fitted transit can be performed using bounded least squares
fitting, the tools to do so are provided.

It's worth noting that this is at its core a template matching algorithm. The 
`get_transit_model()` function can be used to generate a suitable exoplanet transit 
model with any number of samples, parameterised by a planet radius, an impact parameter,
and a set of quadratic limb darkening coefficients. Stricly speaking though, any model 
could be used as a template, and there's no restriction that the 'transit depth'
must be positive. The transit detector will find the best fitting start time, duration 
and depth. The shape of the model will remain constant, but multiple models can be checked
cheaply, particularly if a periodic signal is not sought. For example, a stellar flare model
(or a suite of them) could be used. A model can be supplied to the `TransitDetector` 
instance on initialisation with the `transit_model=` kwarg.

There's the obvious potential for a further optimisation here. Instead of the start time
grid spacing being that of some fraction of the minimum duration, it could vary by
duration. In the grid of duration index vs start time index this change would empty the 
high duration high start time number portion. Warp divergence should be avoided where possible, but 
where grid blocks are totally empty their computation could be skipped and hence some
time saved. This is how TLS operates, but from a parallelisation perspective the
algorithm is simpler as currently implemented.

## Installation

### Requirements

Requirements are the `numpy`, `scipy`, `pycuda` and `tqdm` python packages. The CUDA 
toolkit must also be installed, with nvcc available on the system path.<br>
See https://developer.nvidia.com/cuda-toolkit<br>
As long as the CUDA toolkit is installed, pip will take care of the installation of the 
python module dependencies.

### Installation

With the CUDA toolkit installed, module installation is simply a matter of cloning the 
bitbucket repository, and installing via pip:

```shell
git clone https://github.com/leigh2/cetra.git
cd cetra
pip install -e .
```

### Running tests

Unit tests can be run through unittest or by navigating to the `tests` directory
and running `test_cetra.py` directly from the command line. 

## Basic usage

Example notebooks showing usage of the module for transit detection have been 
provided in the `examples` directory.

The transit detector is initialised with:

```python
from cetra import TransitDetector

td = TransitDetector(time, flux, flux_error)
```
where `time`, `flux`, and `flux_error` are 1D arrays containing light curve 
times, fluxes and flux errors.

The monotransit detector can then be run with:
```python
monotransit_result = td.monotransit_search()
```
Which will scan 2d grid of durations and start times and compute their likelihood ratios
vs. a constant flux model and then return a `MonotransitResult` instance. 
The `MonotransitResult` can be queried to obtain the grid, and provides methods 
for extracting the maximum likelihood or maximum SNR transit parameters:<br>
`monotransit_result.get_max_likelihood_parameters()` and <br>
`monotransit_result.get_max_snr_parameters()`<br>
return a `Transit` dataclass containing the parameters as instance variables.
Note that the `Transit.period` parameter will be `None` at this point.
`MonotransitResult` instances also contain the light curves, duration and start time grids,
and the log-likelihood of the constant flux model.

Once the monotransit result has been computed, the grid of likelihood ratios can
be phase-folded to produce a periodogram. This is done with:
```python
periodic_result = td.period_search()
```
Which returns a `PeriodogramResult` instance, which can be queried to obtain the
light curve, the duration, start time, and period grids, and the `MonotransitResult` from
which it was produced.
The returned `PeriodogramResult` instance also contains the likelihood ratio, 
depth, depth variance, start time index and duration index arrays containing the values
of the maximum likelihood model at each period. i.e. if the period grid contains N
elements, these arrays also contain N elements.
`PeriodogramResult` instances also provide methods for extracting transit parameters 
of the maximum likelihood or maximum SNR period:<br>
`periodic_result.get_max_likelihood_parameters()` <br>
and `periodic_result.get_max_snr_parameters()`<br>
return a `Transit` dataclass containing the parameters as instance variables.

The `.get_max_likelihood_parameters()` and `.get_max_snr_parameters()` can optionally 
be called with `lsq_refine=True` to perform a bounded least squares refinement of the
transit parameters. In this case a `Transit` object is returned containing the fitted 
parameters in addition to the one containing the grid search parameters.


## Acknowledgements
Leigh Smith acknowledges support from PLATO grant UKSA ST/R004838/1
