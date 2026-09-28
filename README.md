<p align="center">
    <a href="./docs/source/_static/img/logo_lm.png#gh-light-mode-only">
        <img src="./docs/source/_static/img/logo_lm.png#gh-light-mode-only" alt="phiPICT Logo" width="40%"/>
    </a>
    <a href="./docs/source/_static/img/logo_dm.png#gh-dark-mode-only">
        <img src="./docs/source/_static/img/logo_dm.png#gh-dark-mode-only" alt="phiPICT Logo" width="40%"/>
    </a>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-%2376B900)
![License](https://img.shields.io/badge/License-Apache--2.0-orange)

</div>

# $\phi$-PICT: Differentiable multi-block PISO MHD fluid solver, based on PICT.

---

## Installation

### 🧱 Build from Source

Here is an example for Python 3.12 and CUDA 12.8.

1. Create a new conda environment and activate it:
```bash
conda create -n phipict python=3.12
conda activate phipict
```

2. Install the CUDA toolkit from conda-forge. This also installs a gcc/g++ that
   the chosen nvcc supports, so no separate compiler install is needed. The CUDA
   version must match the one your PyTorch build uses (12.8 here):
```bash
conda install -c conda-forge pip cuda-toolkit cuda-version=12.8
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
conda activate phipict  # re-activate so CUDA_HOME takes effect
```

3. Install the latest PyTorch for the same CUDA version via pip:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

4. Clone the repository and enter the directory, then compile the custom CUDA kernels and install the package (this might take several minutes):
```bash
make install
```

For development, use `make install-dev` instead, which installs the package in
editable mode together with the development dependencies.

Optional extras can be installed via `pip install ".[amg]"`, `".[plot]"`,
`".[utils]"`, or `".[all]"`.

## Getting Started

See the [`examples`](examples) directory for example simulations.

## License

This repository is published under the Apache-2.0 license.
