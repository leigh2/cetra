Installation
============

Requirements
------------

CETRA requires a CUDA-capable GPU.  The ``nvcc`` compiler must be on the
system ``PATH`` and ``pycuda`` must be installed before CETRA can be used.

The Python dependencies are:

- Python ≥ 3.8
- numpy ≥ 1.15
- pycuda ≥ 2022.1
- scipy ≥ 1.7
- tqdm ≥ 4.0

Installing from PyPI
--------------------

.. code-block:: bash

    pip install cetra

Installing from source
----------------------

.. code-block:: bash

    git clone https://github.com/leigh2/cetra.git
    cd cetra
    pip install -e .

The CUDA kernels in ``src/cetra/cetra.cu`` are compiled at runtime by
pycuda on the first call that requires GPU access, so no separate
compilation step is needed.
