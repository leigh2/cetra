Quickstart
==========

This page shows the minimal steps needed to detect a transit signal.  For
a worked example on real TESS data see the notebooks in the ``examples/``
directory of the repository.

Loading a light curve
---------------------

Pass time (days), flux, and flux error arrays to :class:`~cetra.LightCurve`.
Flux values should be normalised to a baseline of 1.0.

.. code-block:: python

    import numpy as np
    from cetra import LightCurve, TransitDetector

    lc = LightCurve(times, fluxes, flux_errors)

CETRA resamples the input data onto a uniform cadence using a GPU kernel.
Any cadence gaps are filled with null points (infinite error) so that array
indexing can be used instead of time lookups during the search.

Running the linear search
--------------------------

Wrap the light curve in a :class:`~cetra.TransitDetector` and call
:meth:`~cetra.TransitDetector.linear_search`:

.. code-block:: python

    detector = TransitDetector(lc)
    linear_result = detector.linear_search()

This performs a 2-D grid search over transit duration and mid-transit time
(t0), returning a :class:`~cetra.LinearResult` containing likelihood, depth
and depth-variance arrays.

To retrieve the highest-SNR single transit:

.. code-block:: python

    transit = detector.get_max_snr_single_transit()
    print(transit)

Running the period search
--------------------------

The period search phase-folds the linear search result over a grid of
periods.  It must follow the linear search:

.. code-block:: python

    periodic_result = detector.period_search()
    transit = detector.get_max_likelihood_periodic_transit()
    print(transit)

Extracting multiple signals
----------------------------

Use the iterative extraction methods to find all signals above a given SNR
threshold.  Each signal is masked out before the next search, then the
original light curve is restored.

.. code-block:: python

    # single transits above SNR 7
    transits = detector.get_single_transits_above_snr_threshold(snr=7.0)

    # periodic transits above SNR 7
    transits = detector.get_periodic_transits_above_snr_threshold(snr=7.0)

Using a different transit model
--------------------------------

Three built-in limb-darkened models are available at impact parameters
b = 0.32, 0.93 and 0.99:

.. code-block:: python

    from cetra import TransitModel

    model = TransitModel('b93')
    detector = TransitDetector(lc, transit_model=model)

A custom 1-D array can also be passed directly to :class:`~cetra.TransitModel`.
