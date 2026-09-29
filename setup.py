# Copyright 2026 Jannis Becktepe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for the phipict package with CUDA extensions."""

import os

import torch
from setuptools import find_packages, setup
from torch.utils import cpp_extension

# include root of the C++/CUDA sources (absolute: the compiler does not run in the repo root)
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "phipict")

VERSION = "0.1.0"


def _get_version() -> str:
    """Return the package version, optionally with a torch/CUDA local tag.

    With PHIPICT_LOCAL_VERSION=1 the version gets a local tag such as
    ``+pt29cu128`` (see CI.md), because the extension is ABI-bound to the torch
    minor version and CUDA version it was built against.
    """
    if os.environ.get("PHIPICT_LOCAL_VERSION", "0") != "1":
        return VERSION
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    if torch.version.cuda is None:
        raise RuntimeError("PHIPICT_LOCAL_VERSION=1 requires a CUDA build of torch")
    cuda = torch.version.cuda.replace(".", "")
    return f"{VERSION}+pt{major}{minor}cu{cuda}"


def _get_install_requires() -> list[str]:
    """Return the runtime dependencies.

    Binary wheels only work with the torch minor version they were built
    against, so with PHIPICT_PIN_TORCH=1 (implied by PHIPICT_LOCAL_VERSION=1)
    they pin it (e.g. ``torch==2.14.*``). The PyPI wheels use PHIPICT_PIN_TORCH=1
    alone, since PyPI rejects local version tags.
    """
    torch_req = "torch>=2.9"
    if "1" in (
        os.environ.get("PHIPICT_LOCAL_VERSION", "0"),
        os.environ.get("PHIPICT_PIN_TORCH", "0"),
    ):
        major, minor = torch.__version__.split("+")[0].split(".")[:2]
        torch_req = f"torch=={major}.{minor}.*"
    # pyamg: the AMG pressure solver is used automatically for large systems
    return [torch_req, "numpy", "scipy", "pyamg>=4.0"]


def _get_extensions():
    # Main solver extension
    core_sources = [
        "csrc/phipict/bindings.cpp",
        # domain data structures (blocks, boundaries, domain, device atlas)
        "csrc/phipict/domain/csr_matrix.cpp",
        "csrc/phipict/domain/boundaries.cpp",
        "csrc/phipict/domain/block.cpp",
        "csrc/phipict/domain/domain.cpp",
        # PISO simulation kernels
        "csrc/phipict/piso/launch.cu",
        "csrc/phipict/piso/advection.cu",
        "csrc/phipict/piso/pressure.cu",
        "csrc/phipict/piso/velocity_correction.cu",
        "csrc/phipict/piso/analysis.cu",
        "csrc/phipict/piso/result_copy.cu",
        "csrc/phipict/piso/sgs.cu",
        # inductionless MHD
        "csrc/phipict/mhd/potential.cu",
        # linear solvers
        "csrc/phipict/solvers/linear_solve.cu",
        "csrc/phipict/solvers/krylov/krylov.cu",
        "csrc/phipict/solvers/legacy_cg.cu",
        "csrc/phipict/solvers/legacy_bicgstab.cu",
        # grids and math utilities
        "csrc/phipict/grid/grid_gen.cu",
        "csrc/phipict/grid/resampling.cu",
        "csrc/phipict/grid/transform_vectors.cu",
        "csrc/phipict/math/eigenvalue.cu",
        "csrc/phipict/math/ortho_basis.cu",
        "csrc/phipict/math/matrix_vector_ops.cu",
        "csrc/phipict/math/matrix_vector_ops_grads.cu",
    ]
    core_macros = [("PYTHON_EXTENSION_BUILD", "1")]

    core_ext = cpp_extension.CUDAExtension(
        name="phipict._C",
        sources=core_sources,
        include_dirs=[CSRC],
        extra_compile_args={
            "cxx": [
                "-fvisibility=hidden",
                "-Ofast",
                "-w",  # Suppress all warnings for GCC/Clang
            ],
            "nvcc": [
                "--threads=2",
                "-O3",
                "--use_fast_math",
                "--compiler-options=-w",  # Pass -w to the host compiler (GCC/Clang)
            ],
        },
        extra_link_args=[],
        define_macros=core_macros,
    )

    # SimplexNoiseVariations extension
    noise_sources = [
        "csrc/phipict/noise/simplex_noise.cu",
        "csrc/phipict/noise/SimplexNoiseVariations.cpp",
    ]

    noise_ext = cpp_extension.CUDAExtension(
        name="phipict._noise",
        sources=noise_sources,
        include_dirs=[
            os.path.join(CSRC, "noise"),
            CSRC,
        ],
        extra_compile_args={"cxx": ["-fvisibility=hidden"]},
    )

    # Only build the noise extension if the environment variable is set
    if os.environ.get("PHIPICT_BUILD_NOISE_EXT", "0") == "1":
        return [core_ext, noise_ext]
    else:
        return [core_ext]


setup(
    version=_get_version(),
    install_requires=_get_install_requires(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=_get_extensions(),
    cmdclass={
        "build_ext": cpp_extension.BuildExtension,
    },
)
