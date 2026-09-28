Installation
============

phiPICT ships a compiled CUDA extension (``phipict._C``). It requires Linux, an
NVIDIA GPU, and Python 3.11–3.14. We recommend a dedicated virtual environment.

Which wheel do I need?
----------------------

The extension is bound to two versions it was built against:

- the **PyTorch minor version** (e.g. 2.9), because libtorch has no stable C++ ABI;
- the **CUDA major version** (cu12x or cu13x).

A single wheel can therefore not serve every setup. Wheels are built for each
combination of Python × PyTorch minor × CUDA major, and the combination is part of
the version as a local tag, e.g. ``phipict-0.1.0+pt29cu128``. Check your
installed PyTorch with:

.. code-block:: bash

    python -c "import torch; print(torch.__version__, torch.version.cuda)"

1. Prebuilt wheels
------------------

PyPI does not accept local version tags, so the wheels are hosted on GitHub
Releases and listed in a wheel index with one page per PyTorch/CUDA combination.
Install PyTorch first, then point ``pip`` to the matching page:

.. code-block:: bash

    pip install torch==2.9.* --index-url https://download.pytorch.org/whl/cu128
    pip install phipict -f https://safe-autonomous-systems.github.io/phiPICT/whl/torch-2.9.0+cu128.html

Optional extras: ``amg`` (``pyamg``, needed for ``potential_use_preconditioner``),
``plot`` (grid plots and image output), ``utils``, or ``all``.

2. Building from source
-----------------------

For combinations without a prebuilt wheel, phiPICT builds against the PyTorch
already installed in your environment. This requires a CUDA toolkit whose major
version matches ``torch.version.cuda`` (``nvcc`` on the ``PATH`` or
``CUDA_HOME`` set).

.. code-block:: bash

    git clone https://github.com/safe-autonomous-systems/phiPICT.git
    cd phiPICT
    make install        # builds a wheel into dist/ and installs it
    make install-dev    # editable install with development tools

The build honours the usual PyTorch extension variables:

- ``TORCH_CUDA_ARCH_LIST``: target GPU architectures, ``"8.0;8.6;9.0+PTX"`` in
  the Makefile. Add ``12.0`` for Blackwell GPUs (needs CUDA ≥ 12.8).
- ``MAX_JOBS``: number of parallel compile jobs. Compiling is memory hungry, so
  the Makefile uses ``1``.
- ``PHIPICT_BUILD_NOISE_EXT=1``: additionally build the simplex noise extension
  ``phipict._noise``.

``make build-manylinux`` builds manylinux wheels with ``cibuildwheel``, as done in
CI.

Troubleshooting
---------------

An ``ImportError: ... undefined symbol`` when importing ``phipict`` means that the
installed wheel was built for a different PyTorch version. Reinstall the wheel
matching your PyTorch/CUDA combination, or build from source.
