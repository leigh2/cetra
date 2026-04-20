CETRA documentation
===================

**CETRA** (Cambridge Exoplanet Transit Recovery Algorithm) is a GPU-accelerated
exoplanet transit detection library.  It identifies transit signals in stellar
light curves using a physically-motivated transit model and CUDA kernels.

Published in `Smith et al., 2025, MNRAS, 539, 297
<https://ui.adsabs.harvard.edu/abs/2025MNRAS.539..297S>`_.

The two-stage detection pipeline:

.. code-block:: text

    LightCurve(times, fluxes, errors)     # resamples to regular cadence via GPU
        ↓
    TransitDetector(lc)                   # prepares duration/t0 grids
        ↓
    .linear_search()  →  LinearResult     # 2D grid (duration × t0)
        ↓
    .period_search()  →  PeriodicResult   # phase-folds LinearResult over periods
        ↓
    .get_max_likelihood_periodic_transit() → Transit

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   cetra

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
