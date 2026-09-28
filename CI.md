# Wheel build strategy

The compiled extension (`phipict._C`) is ABI-bound to the **torch minor version**
it was built against (libtorch has no stable C++ ABI) and to the **CUDA major
version** (cu12 vs. cu13). A single wheel therefore cannot serve all setups.

## Approach: build matrix + own wheel index

Same scheme as PyG (`torch-scatter`) and flash-attn.

1. **Build matrix in CI** (cibuildwheel): one wheel per
   Python (3.11–3.14) × torch minor × CUDA major, each built with the lowest
   CUDA minor of its major. CUDA minor-version compatibility makes a cu130
   build work with torch cu130 and cu132, so no per-minor builds are needed.
   `release.yml` builds:

   | torch | CUDA | Notes |
   |---|---|---|
   | 2.10 | 12.8 | default, also on (Test)PyPI; runs with drivers < 580 |
   | 2.11, 2.12, 2.13, 2.14 | 13.0 | serves torch cu130 and cu132; driver >= 580 |

   torch >= 2.11 with CUDA 12 (cu126/cu129 builds, for drivers < 580) is not
   covered; such setups use torch 2.10 or build from source.
   - Select torch/CUDA per job via environment variables instead of pinning
     `torch==2.9.1+cu128` in `[build-system]`.
   - Install the matching CUDA toolkit in the build image, build with
     `--no-build-isolation` against the installed torch.
   - Set `TORCH_CUDA_ARCH_LIST` per CUDA version (e.g. add `12.0` for
     Blackwell with CUDA ≥ 12.8).

2. **Local version tag**: `setup.py` appends the build combination to the
   version, e.g. `phipict-0.1.0+pt210cu128`, derived from `torch.__version__`
   and `torch.version.cuda` at build time.

3. **Own wheel index**: PyPI rejects local version tags, so wheels are
   uploaded to GitHub Releases and listed on a GitHub Pages index, one page
   per torch minor × CUDA major:

   ```bash
   pip install phipict -f https://safe-autonomous-systems.github.io/phiPICT/whl/torch-2.12+cu13.html
   ```

## Implementation

Wheels are built by `.github/workflows/release.yml` on GitHub Actions, one
cibuildwheel job per Python × torch/CUDA combination, only when started
manually (*Actions → Release → Run workflow*, pick the branch or tag to build).
The index wheels (`index` input) are the `build-index` job's matrix; `DEFAULT_ARCH_LIST`
sets the GPU archs (`8.0;9.0;10.0;12.0+PTX`) for all wheels.

- The cibuildwheel configuration lives in `[tool.cibuildwheel]` in
  `pyproject.toml`; the combination is passed as `TORCH_VERSION`,
  `CUDA_VERSION`, `CUDA_TAG`. `PHIPICT_LOCAL_VERSION=1` makes `setup.py`
  append the `+ptXYcuXYZ` tag and pin `torch==X.Y.*` in the wheel metadata.
- PyTorch publishes no cu128 wheels for torch >= 2.12; cu126 is the CUDA 12
  build for drivers < 580 (CUDA 13 requires driver >= 580).
- CUDA < 12.8 does not accept the manylinux image's GCC 14, so those builds
  install and use `gcc-toolset-13`.
- `.github/scripts/make_wheel_index.py` generates the find-links pages from
  release assets.

### Release procedure

Same procedure as fluidgym (released together; fluidgym pins `phipict==X.Y.*`,
publish phipict first):

1. Set `VERSION` in `setup.py` to the release version, commit, then
   `git tag v0.1.0 && git push origin v0.1.0` (pushing the tag does not publish).
2. *Actions → Release → Run workflow* on the tag with target `testpypi`: uploads
   the default wheels of the exact release version to TestPyPI. TestPyPI and
   PyPI are separate registries, so no `.dev` version is needed.
3. Run it again on the tag with target `pypi` (and `index` for the wheel
   index): uploads to PyPI and, with `index`, creates a **draft** release with
   the index wheels (only visible to repo collaborators).
4. Publishing the draft triggers `wheel-index.yml`.

The `check` job fails early if the run is not on a `vX.Y.Z` tag matching
`VERSION`. A file name can only be uploaded once, even after deleting it; to
retry on TestPyPI, set `build_number` (wheel `phipict-0.1.0-1-cp312-...whl`,
pip prefers the highest build number). With target `none`, the wheels are only
built (downloadable as run artifacts).

### Default wheels on (Test)PyPI

`release.yml` also builds one *default* combination without local version tag
(`PHIPICT_LOCAL_VERSION=0 PHIPICT_PIN_TORCH=1`, so `phipict-0.1.0` requiring
`torch==2.10.*`) for all Python versions and uploads it with target
`testpypi` or `pypi`.
The combination is set by `DEFAULT_TORCH` / `DEFAULT_CUDA` / `DEFAULT_CUDA_TAG`
/ `DEFAULT_ARCH_LIST` at the top of the workflow. Its CUDA version must match
the torch wheel on PyPI (CUDA 12.8 for torch <= 2.10, CUDA 13.0 for
torch >= 2.11), since `pip install phipict` pulls torch from PyPI. The default
is torch 2.10 + cu128 so that it runs with drivers < 580 (CUDA 12 only).

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ phipict
```

The upload uses trusted publishing (no API token). One-time setup:

1. On test.pypi.org and pypi.org: *Your projects → Publishing → Add a new
   pending publisher* with project `phipict`, owner/repository of this repo,
   workflow `release.yml`, environment `testpypi` (TestPyPI) / `pypi` (PyPI).
2. On GitHub: *Settings → Environments* `testpypi` and `pypi` (optionally
   restricted to `v*` tags).
3. `workflow_dispatch` workflows can only be started when the workflow file is
   on the default branch; the run then uses the file of the selected ref.

PyPI limits files to 100 MB by default; check the wheel size when adding archs.

Prerequisites for the public index: the repository must be public (release
assets of private repos need authentication, so pip cannot download them),
and GitHub Pages must be enabled with source `gh-pages` / root.

## Complements

- **sdist fallback** on PyPI for unsupported combinations (builds against the
  user's installed torch; requires a CUDA toolkit).
- **Import-time check**: record the build's torch/CUDA versions (e.g. in a
  generated `phipict/_build_info.py`) and raise a clear error on mismatch
  instead of an `undefined symbol` import failure.
- **Long term**: PyTorch's stable ABI (torch ≥ 2.9) could allow one wheel per
  CUDA major, but requires rewriting the pybind11 class bindings.