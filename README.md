# Cambridge Exoplanet Transit Recovery Algorithm - CETRA
Exoplanet transit detection within the NVIDIA CUDA GPU framework.

## Overview

CETRA separates the transit detection task into a linear transit search followed 
by a phase-folding of the former into a periodic signal search, using a 
physically motivated transit model to improve detection sensitivity. Implemented 
with NVIDIA’s CUDA platform, it outperforms traditional methods like Box Least 
Squares and Transit Least Squares in both sensitivity and speed. It can also 
be used to identify transits that aren't periodic in the input light curve (AKA 
monotransits). CETRA is designed to be run on detrended light curves.

The transit detection algorithm has three setup stages, followed by two main stages:

* * The first setup step resamples the input light curve such that the cadence
    is regularised and data gaps are eliminated. New data points with no contributing 
    observations are null, with infinite uncertainty. This resampling means the 
    light curve point roughly corresponding to any point in time can be determined 
    efficiently.
  * The second setup step prepares a transit model for comparison to the resampled light 
    curve. Models with three values of impact parameter are provided, but the user is also
    free to supply their own. Input model points are evenly spaced in time, and the model is 
    scaled such that the maximum transit depth is 1. By default, the model has 1024 data 
    points, though this is configurable. The default number of points correspond to a maximum 
    error of ~1% in flux (mean ~0.1%), when taking the nearest model point for any point in time.
    The maximum error occurs where the flux gradient is largest (i.e. during ingress and egress).
  * The final setup step produces grids of durations and start times, and (when required) periods
    from user inputs (or default values). The user is also free to supply their own grids. The 
    reference time (t0) grid stride length is computed as a fraction of the minimum duration 
    (1% by default but user-configurable). Unless provided by the user, the duration and period 
    grids are computed using the same algorithms as TLS.
* The first main stage is the linear search. It traverses the grid of durations and t0, and 
  for each it calculates the maximum-likelihood transit depth, its variance, and its likelihood 
  ratio relative to a constant flux model. The grids of results can be interrogated to obtain 
  likely singular transits.
* The second main stage is the periodic signal search. It traverses a grid of periods, and for 
  each it phase-folds the arrays of depths, variances and likelihood ratios from the previous 
  stage, finding the maximum joint-likelihood ratio (again versus the constant flux model). 
  The corresponding joint-likelihood ratio, depth, depth variance, start time and duration are 
  recorded for each period grid point, from which a periodogram can be produced and periodic 
  signals identified.

A more thorough description of the algorithm can be found in the paper
🔴insert DOI/ADS/arXiv link when published🔴.

## Installation

### Requirements

Requirements are the `numpy`, `scipy`, `pycuda` and `tqdm` python packages. The CUDA 
toolkit must also be installed, with nvcc available on the system path.<br>
See https://developer.nvidia.com/cuda-toolkit<br>
As long as the CUDA toolkit is installed, pip will take care of the installation of the 
python module dependencies.

### Installation

With the CUDA toolkit installed, module installation is simply a matter of cloning the 
git repository, and installing via pip:

```shell
git clone https://github.com/leigh2/cetra.git
cd cetra
pip install -e .
```

### Running tests


Unit tests can be run through unittest or by navigating to the `tests` directory
and running `test_cetra.py` directly from the command line. 

## Basic usage

An example notebook demonstrating the usage of CETRA to identify the three planets of 
the HD 101581 system in TESS short cadence data is provided in the `examples` directory.

The most basic usage will involve creating a light curve instance with:

```python
from cetra import LightCurve
light_curve = LightCurve(times, fluxes, flux_errors)
```
where `times`, `fluxes`, and `flux_errors` are 1D arrays containing observation times 
(days), and the observed fluxes (normalised to baseline) and their errors, respectively.
This runs the resampling setup step mentioned above, by default it uses the median input
cadence, but the user can specify an alternative cadence if desired.

The transit detector can then be initialised with default parameters (incl. transit 
model) with:
```python
from cetra import TransitDetector
transit_detector = TransitDetector(light_curve)
```
This reads and resamples the default transit model, and generates the duration and t0 
grids as mentioned above. The user can specify or supply an alternative transit model, 
duration grid or t0 spacing here if they wish.

The linear search can then be run with:
```python
linear_result = transit_detector.linear_search()
```
`linear_result` is a `cetra.LinearResult` instance, and can be probed for single transits.
For example, the maximum likelihood single transit can be obtained as a `cetra.Transit` instance
with the method:
```python
monotransit_ml = linear_result.get_max_likelihood_parameters()
print(monotransit_ml)

# Transit(
#         t0=3017.123841601484, 
#         duration=0.0835449633883131, 
#         depth=0.00027097395, 
#         depth_error=2.6651789e-05, 
#         period=None
#        )
```

The periodic signal search can then be run over the default period grid specification with:
```python
periodic_result = transit_detector.period_search()
```
The user is free to specify or supply an alternative period grid at this stage. `periodic_result`
is a `cetra.PeriodicResult` instance and can be probed for periodic signals. For example, the 
maximum likelihood signal, as a `cetra.Transit` instance, can be obtained with the method:
```python
periodic_ml = periodic_result.get_max_likelihood_parameters()
print(periodic_ml)

# Transit(
#         t0=3014.8518416020543, 
#         duration=0.06904542428786206, 
#         depth=0.00019935601, 
#         depth_error=8.626047e-06, 
#         period=4.465103568398068
#        )
```

## Acknowledgements
Leigh Smith acknowledges support from UKRI-STFC grants ST/X001628/1 and ST/X001571/1.
