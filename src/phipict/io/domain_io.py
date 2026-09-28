# Original work Copyright 2025 Aleksandra Franz, Nils Thuerey
# Modified work Copyright 2026 Jannis Becktepe
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
#
# Modifications:
# Save/load of MHD electric potential settings and open boundaries.
# Moved into the phipict package, formatted, linted and typed.

"""Saving and loading of multi-block domains to/from ``.json`` + ``.npz`` files.

A domain is stored as a pair of files next to each other, ``<name>.json`` (the
structure: blocks, boundaries and settings) and ``<name>.npz`` (the tensors,
referenced by index from the JSON). Both functions take the path of the pair
without the extension, as a :class:`pathlib.Path`::

    save_domain(domain, Path("runs/case/domain"))  # domain.json + domain.npz
    domain = load_domain(Path("runs/case/domain"), dtype=torch.float64)

Saving is atomic per file and the two files carry a shared id, so a save that
was interrupted (e.g. a killed job checkpointing) is detected on load instead of
silently mixing an old structure with new tensors.
"""

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from phipict import _C

__all__ = ["load_domain", "save_domain"]

#: Version of the file layout written by :func:`save_domain`. Version 1 (no
#: ``format_version`` key) is still read.
FORMAT_VERSION = 2

_SUFFIXES = (".json", ".npz")
_SAVE_ID_KEY = "save_id"


def dtype_to_string(dtype: torch.dtype) -> str:
    """Convert a saved tensor's dtype to its name.

    Parameters
    ----------
    dtype : torch.dtype
        ``torch.float32``, ``torch.float64`` or ``torch.int8`` (potential BC types).

    Returns
    -------
    str
        ``"float32"``, ``"float64"`` or ``"int8"``.

    Raises
    ------
    TypeError
        If the dtype is not supported.
    """
    if dtype == torch.float32:
        return "float32"
    elif dtype == torch.float64:
        return "float64"
    elif dtype == torch.int8:
        return "int8"
    else:
        raise TypeError(f"Unsupported dtype {dtype}.")


def boundary_condition_type_to_string(bound_type: _C.BoundaryConditionType) -> str:
    """Convert a boundary condition type to its name.

    Parameters
    ----------
    bound_type : phipict._C.BoundaryConditionType
        Boundary condition type.

    Returns
    -------
    str
        ``"DIRICHLET"`` or ``"NEUMANN"``.

    Raises
    ------
    TypeError
        If the type is not supported.
    """
    if bound_type == _C.BoundaryConditionType.DIRICHLET:
        return "DIRICHLET"
    elif bound_type == _C.BoundaryConditionType.NEUMANN:
        return "NEUMANN"
    else:
        raise TypeError("Unsupported boundary condition type.")


def boundary_condition_string_to_type(bound_type: str) -> _C.BoundaryConditionType:
    """Convert a boundary condition name to its type.

    Parameters
    ----------
    bound_type : str
        ``"DIRICHLET"`` or ``"NEUMANN"``.

    Returns
    -------
    phipict._C.BoundaryConditionType
        Boundary condition type.

    Raises
    ------
    TypeError
        If the name is not supported.
    """
    if bound_type == "DIRICHLET":
        return _C.BoundaryConditionType.DIRICHLET
    elif bound_type == "NEUMANN":
        return _C.BoundaryConditionType.NEUMANN
    else:
        raise TypeError("Unsupported boundary condition type.")


def bound_idx_to_str(idx: int) -> str:
    """Convert a boundary index to its face string.

    Parameters
    ----------
    idx : int
        Boundary index in ``[0, 5]``.

    Returns
    -------
    str
        Face string, one of ``"-x"``, ``"+x"``, ``"-y"``, ``"+y"``, ``"-z"``,
        ``"+z"``.

    Raises
    ------
    ValueError
        If the index is out of range.
    """
    if idx == 0:
        return "-x"
    elif idx == 1:
        return "+x"
    elif idx == 2:
        return "-y"
    elif idx == 3:
        return "+y"
    elif idx == 4:
        return "-z"
    elif idx == 5:
        return "+z"
    raise ValueError(f"Invalid boundary index: {idx}")


def _load_potential_bc(
    bound: _C.FixedBoundary,
    bound_dict: dict[str, Any],
    types: torch.Tensor | None,
    values: torch.Tensor | None,
) -> None:
    """Restore the electric potential BC of a FIXED boundary.

    Parameters
    ----------
    bound : phipict._C.FixedBoundary
        Boundary to configure.
    bound_dict : dict[str, Any]
        The boundary's saved settings.
    types : torch.Tensor or None
        The saved per-cell types, if the face had a mask. Files written before
        integer tensors kept their dtype on load hold them as floats.
    values : torch.Tensor or None
        The saved Dirichlet values, if the face had any.
    """
    if values is not None:
        bound.setPotentialValues(values)
    cw = bound_dict.get("potentialCw")
    if types is not None:
        if types.is_floating_point():
            types = types.round()
        bound.setPotentialTypes(types.to(torch.int8), cw=cw)
    elif "potentialBC" in bound_dict:
        bound.setPotentialBC(
            _C.PotentialBC.__members__[bound_dict["potentialBC"]], cw=cw
        )
    else:
        # files written before PotentialBC existed stored independent flags
        if "epotCw" in bound_dict:
            bound.setEpotCw(bound_dict["epotCw"])
        if "epotInsulating" in bound_dict:
            bound.setEpotInsulating(bound_dict["epotInsulating"])
        if bound_dict.get("epotDirichlet", False):
            bound.setEpotDirichlet(True)


def _fixed_boundary(block: _C.Block, idx: int) -> _C.FixedBoundary:
    """Get a boundary of a block that is known to be a FixedBoundary.

    Parameters
    ----------
    block : phipict._C.Block
        Block to read the boundary from.
    idx : int
        Boundary index.

    Returns
    -------
    phipict._C.FixedBoundary
        The boundary, narrowed to ``FixedBoundary`` for the type checker.
    """
    return cast(_C.FixedBoundary, block.getBoundary(idx))


def _pair_paths(path: Path) -> tuple[Path, Path]:
    """The ``.json`` and ``.npz`` paths of the file pair at ``path``.

    Parameters
    ----------
    path : pathlib.Path
        Path of the pair without extension. A ``.json`` or ``.npz`` extension is
        accepted too and dropped, so either file of a pair can be passed.

    Returns
    -------
    tuple of pathlib.Path
        The ``.json`` and the ``.npz`` path.

    Raises
    ------
    TypeError
        If ``path`` is not a :class:`pathlib.Path`.
    """
    if not isinstance(path, Path):
        raise TypeError(
            f"path must be a pathlib.Path, got {type(path).__name__}: {path!r}."
        )
    if path.suffix in _SUFFIXES:
        path = path.with_suffix("")
    # with_suffix would replace a dot in the name itself ("Re2.5" -> "Re2.json")
    return path.with_name(path.name + ".json"), path.with_name(path.name + ".npz")


def _atomic_write(target: Path, write: Any) -> None:
    """Write ``target`` via a temporary file in its directory and a rename.

    Parameters
    ----------
    target : pathlib.Path
        File to (over)write.
    write : callable
        Called with the open binary temporary file.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _tensor_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identity of a tensor's data, to store tensors shared between objects once.

    The bindings return a new Python wrapper on every access, so object identity
    cannot detect sharing; the storage location and layout can.
    """
    return (
        tensor.device,
        tensor.dtype,
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def save_domain(domain: _C.Domain, path: Path) -> None:
    """Save a domain to ``<path>.json`` and ``<path>.npz``.

    Tensors are stored in the ``.npz`` file and referenced by index from the
    ``.json`` description; tensors shared between objects are stored once. Missing
    parent directories are created, and each file is written atomically.

    Parameters
    ----------
    domain : phipict._C.Domain
        Domain to save.
    path : pathlib.Path
        Path of the file pair without extension (a ``.json``/``.npz`` extension
        is dropped).

    Raises
    ------
    TypeError
        If ``path`` is not a :class:`pathlib.Path`, or the domain contains an
        unknown boundary type or tensor dtype.
    NotImplementedError
        If the domain contains a Neumann boundary.
    """
    json_path, npz_path = _pair_paths(path)

    data: list[torch.Tensor] = []
    data_index: dict[tuple[Any, ...], int] = {}

    # The bindings type most tensors as optional; callers check presence first.
    def add_data(tensor: Any, dict: dict[str, Any], name: str) -> None:
        key = _tensor_key(tensor)
        if key not in data_index:
            data_index[key] = len(data)
            data.append(tensor)
        dict[name] = str(data_index[key])

    domain_dict: dict[str, Any] = {"format_version": FORMAT_VERSION}
    domain_dict["name"] = domain.name
    domain_dict["spatialDims"] = domain.getSpatialDims()
    add_data(domain.viscosity, domain_dict, "viscosity")
    domain_dict["passiveScalarChannels"] = domain.getPassiveScalarChannels()
    if domain.hasPassiveScalarViscosity():
        add_data(domain.passiveScalarViscosity, domain_dict, "passiveScalarViscosity")
    domain_dict["blocks"] = []

    blocks = domain.getBlocks()
    for block in blocks:
        block_dict: dict[str, Any] = {}
        block_dict["name"] = block.name
        if block.hasViscosity():
            add_data(block.viscosity, block_dict, "viscosity")
        add_data(block.velocity, block_dict, "velocity")
        add_data(block.pressure, block_dict, "pressure")
        if block.hasPassiveScalar():
            add_data(block.passiveScalar, block_dict, "scalar")

        if block.hasEpot():
            add_data(block.epot, block_dict, "epot")

        if block.hasVelocitySource():
            add_data(block.velocitySource, block_dict, "velocitySource")

        if block.hasVertexCoordinates():
            add_data(block.vertexCoordinates, block_dict, "vertexCoordinates")
        elif block.hasTransform():
            add_data(block.transform, block_dict, "transform")
            if block.hasFaceTransform():
                add_data(block.faceTransform, block_dict, "faceTransform")

        block_dict["boundaries"] = []
        for bound_idx in range(block.getSpatialDims() * 2):
            bound: Any = block.getBoundary(bound_idx)
            bound_dict: dict[str, Any] = {}
            if bound.type == _C.DIRICHLET:
                bound_dict["type"] = "DIRICHLET"
                add_data(bound.slip, bound_dict, "slip")
                add_data(bound.boundaryVelocity, bound_dict, "velocity")
                add_data(bound.boundaryScalar, bound_dict, "scalar")
            elif bound.type == _C.DIRICHLET_VARYING:
                bound_dict["type"] = "DIRICHLET_VARYING"
                add_data(bound.slip, bound_dict, "slip")
                add_data(bound.boundaryVelocity, bound_dict, "velocity")
                add_data(bound.boundaryScalar, bound_dict, "scalar")
                if bound.hasTransform:
                    add_data(bound.transform, bound_dict, "transform")
            elif bound.type == _C.FIXED:
                bound_dict["type"] = "FIXED"
                bound_dict["velocityType"] = boundary_condition_type_to_string(
                    bound.velocityType
                )
                if bound.velocity is not None:
                    add_data(bound.velocity, bound_dict, "velocity")
                if bound.hasPassiveScalar():
                    bound_dict["passiveScalarType"] = [
                        boundary_condition_type_to_string(_)
                        for _ in bound.passiveScalarTypes
                    ]
                    add_data(bound.passiveScalar, bound_dict, "scalar")
                # Only written when non-default (INSULATING, no mask), so a domain saved
                # before these keys existed loads back as all-insulating.
                if bound.hasPotentialTypes():
                    add_data(bound.potentialTypes, bound_dict, "potentialTypes")
                if (
                    bound.hasPotentialTypes()
                    or bound.getPotentialBC() != _C.PotentialBC.INSULATING
                ):
                    bound_dict["potentialBC"] = bound.getPotentialBC().name
                if bound.hasPotentialThinWall():
                    bound_dict["potentialCw"] = bound.getPotentialCw()
                if bound.hasPotentialValues():
                    add_data(bound.potentialValues, bound_dict, "potentialValues")
                if bound.hasTransform():
                    add_data(bound.transform, bound_dict, "transform")
            elif bound.type == _C.NEUMANN:
                raise NotImplementedError(
                    f"Block '{block.name}' face {bound_idx_to_str(bound_idx)}: "
                    "NEUMANN boundaries cannot be saved."
                )
            elif bound.type == _C.CONNECTED:
                bound_dict["type"] = "CONNECTED"
                bound_dict["connectedBlock"] = blocks.index(bound.getConnectedBlock())
                bound_dict["axes"] = bound.axes
            elif bound.type == _C.PERIODIC:
                bound_dict["type"] = "PERIODIC"
            else:
                raise TypeError(
                    f"Block '{block.name}' face {bound_idx_to_str(bound_idx)}: "
                    f"unknown boundary type {bound.type}."
                )
            block_dict["boundaries"].append(bound_dict)

        domain_dict["blocks"].append(block_dict)

    save_id = uuid.uuid4().hex
    domain_dict[_SAVE_ID_KEY] = save_id
    domain_dict["data_info"] = {
        str(i): {
            "shape": list(d.shape),
            "dtype": dtype_to_string(d.dtype),
            "device": str(d.device.type),
        }
        for i, d in enumerate(data)
    }
    arrays = {str(i): d.detach().cpu().numpy() for i, d in enumerate(data)}
    arrays[_SAVE_ID_KEY] = np.array(save_id)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    # the tensors first: the JSON names them, so it is written last
    _atomic_write(
        npz_path,
        lambda file: np.savez_compressed(file, **arrays),  # type: ignore[arg-type]
    )
    _atomic_write(
        json_path, lambda file: file.write(json.dumps(domain_dict, indent=1).encode())
    )


def _read_pair(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read and cross-check the JSON structure and the tensors of a saved domain.

    Parameters
    ----------
    path : pathlib.Path
        Path of the file pair without extension.

    Returns
    -------
    tuple
        The JSON description and the arrays by data index.

    Raises
    ------
    FileNotFoundError
        If either file is missing.
    ValueError
        If the files are unreadable, from different saves, or inconsistent.
    """
    json_path, npz_path = _pair_paths(path)
    missing = [str(p) for p in (json_path, npz_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Saved domain incomplete, missing: {', '.join(missing)}."
        )

    try:
        domain_dict = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{json_path} is not valid JSON: {e}.") from e
    version = int(domain_dict.get("format_version", 1))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{json_path} has format version {version}; this phipict reads up to "
            f"{FORMAT_VERSION}. Update phipict."
        )
    for key in ("spatialDims", "name", "blocks", "data_info", "viscosity"):
        if key not in domain_dict:
            raise ValueError(f"{json_path} is missing the '{key}' entry.")

    data_info: dict[str, Any] = domain_dict["data_info"]
    try:
        with np.load(npz_path) as npz:
            if _SAVE_ID_KEY in domain_dict:
                npz_id = str(npz[_SAVE_ID_KEY]) if _SAVE_ID_KEY in npz else None
                if npz_id != domain_dict[_SAVE_ID_KEY]:
                    raise ValueError(
                        f"{json_path} and {npz_path} belong to different saves "
                        "(interrupted save?)."
                    )
            missing_keys = [k for k in data_info if k not in npz]
            if missing_keys:
                raise ValueError(
                    f"{npz_path} lacks tensors {missing_keys} listed in {json_path}."
                )
            arrays = {k: npz[k] for k in data_info}
    except (OSError, EOFError) as e:
        raise ValueError(f"{npz_path} is not a readable .npz file: {e}.") from e

    for key, info in data_info.items():
        if "shape" in info and list(arrays[key].shape) != list(info["shape"]):
            raise ValueError(
                f"Tensor {key} in {npz_path} has shape {list(arrays[key].shape)}, "
                f"{json_path} expects {info['shape']}."
            )
    return domain_dict, arrays


def load_domain(
    path: Path,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    with_scalar: bool = True,
) -> _C.Domain:
    """Load a domain saved with :func:`save_domain`.

    Parameters
    ----------
    path : pathlib.Path
        Path of the file pair without extension (a ``.json``/``.npz`` extension
        is dropped).
    dtype : torch.dtype or None, optional
        Floating point dtype of the loaded domain. Default is None, the saved one.
    device : torch.device, str or None, optional
        Compute device of the loaded domain. Default is None, the saved one.
        Tensors that live on the host (e.g. the domain viscosity) stay there.
    with_scalar : bool, optional
        Whether to load passive scalars and their boundary conditions.
        Default is True.

    Returns
    -------
    phipict._C.Domain
        The loaded domain. Call ``PrepareSolve()`` before simulating it.

    Raises
    ------
    TypeError
        If ``path`` is not a :class:`pathlib.Path`, or the file contains an
        unknown boundary type.
    FileNotFoundError
        If a file of the pair is missing.
    ValueError
        If the files are corrupt, inconsistent or from a newer phipict.
    NotImplementedError
        If the file contains a Neumann boundary.
    """
    domain_dict, arrays = _read_pair(path)
    data_info: dict[str, Any] = domain_dict["data_info"]

    if dtype is None:
        dtype = getattr(
            torch, data_info[domain_dict["viscosity"]].get("dtype", "float32")
        )
    if device is None:
        # the compute device, i.e. that of the block fields
        blocks_dicts = domain_dict["blocks"]
        velocity_key = blocks_dicts[0].get("velocity") if blocks_dicts else None
        device = (
            data_info[velocity_key].get("device", "cuda") if velocity_key else "cuda"
        )
    target_device = torch.device(device)

    tensors: dict[str, torch.Tensor] = {}
    for key, array in arrays.items():
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        # host tensors (e.g. the domain viscosity) stay on the host, the rest moves
        # to the target device; integer tensors (potential BC types) keep their dtype
        on_host = data_info[key].get("device", "cuda") == "cpu"
        tensors[key] = tensor.to(
            device="cpu" if on_host else target_device,
            dtype=dtype if tensor.is_floating_point() else None,
        )

    # Returns Any: callers either check presence first or accept None.
    def get_data(dict: dict[str, Any], name: str) -> Any:
        return tensors[dict[name]] if name in dict else None

    domain = _C.Domain(
        domain_dict["spatialDims"],
        get_data(domain_dict, "viscosity"),
        domain_dict["name"],
        dtype=dtype,
        device=target_device,  # type: ignore[arg-type]
        passiveScalarChannels=(
            domain_dict.get("passiveScalarChannels", 1) if with_scalar else 0
        ),
        scalarViscosity=(
            get_data(domain_dict, "passiveScalarViscosity") if with_scalar else None
        ),
    )

    n_bounds = 2 * domain_dict["spatialDims"]
    for block_dict in domain_dict["blocks"]:
        if len(block_dict.get("boundaries", [])) != n_bounds:
            raise ValueError(
                f"Block '{block_dict.get('name')}' lists "
                f"{len(block_dict.get('boundaries', []))} boundaries, expected {n_bounds}."
            )
        vertex_coordinates = get_data(block_dict, "vertexCoordinates")
        block = domain.CreateBlock(
            get_data(block_dict, "velocity"),
            get_data(block_dict, "pressure"),
            get_data(block_dict, "scalar") if with_scalar else None,
            vertexCoordinates=vertex_coordinates,
            name=block_dict["name"],
        )

        if "viscosity" in block_dict:
            block.setViscosity(get_data(block_dict, "viscosity"))

        if "velocitySource" in block_dict:
            block.setVelocitySource(get_data(block_dict, "velocitySource"))

        if "epot" in block_dict:
            block.setEpot(get_data(block_dict, "epot"))

        if vertex_coordinates is None and "transform" in block_dict:
            block.setTransform(
                get_data(block_dict, "transform"),
                get_data(block_dict, "faceTransform"),
            )

    # all blocks exist now, so connections can refer to them by index
    blocks = domain.getBlocks()
    for block_dict, block in zip(domain_dict["blocks"], blocks, strict=True):
        for bound_idx, bound_dict in enumerate(block_dict["boundaries"]):
            bound_type = bound_dict["type"]
            if bound_type == "FIXED":
                _load_fixed_boundary(
                    block, bound_idx, bound_dict, get_data, with_scalar
                )
            elif bound_type == "NEUMANN":
                raise NotImplementedError(
                    f"Block '{block.name}' face {bound_idx_to_str(bound_idx)}: "
                    "NEUMANN boundaries cannot be loaded."
                )
            elif bound_type == "CONNECTED":
                block.ConnectBlock(
                    bound_idx_to_str(bound_idx),
                    blocks[bound_dict["connectedBlock"]],
                    *[bound_idx_to_str(x) for x in bound_dict["axes"]],
                )
            elif bound_type == "PERIODIC":
                block.MakePeriodic(bound_idx // 2)
            elif bound_type in ("DIRICHLET", "DIRICHLET_VARYING"):
                # the pre-FixedBoundary formats; Block no longer creates them
                raise NotImplementedError(
                    f"Block '{block.name}' face {bound_idx_to_str(bound_idx)}: "
                    f"{bound_type} boundaries (old format) cannot be loaded."
                )
            else:
                raise TypeError(
                    f"Block '{block.name}' face {bound_idx_to_str(bound_idx)}: "
                    f"unknown boundary type {bound_type!r}."
                )

    return domain


def _load_fixed_boundary(
    block: _C.Block,
    bound_idx: int,
    bound_dict: dict[str, Any],
    get_data: Any,
    with_scalar: bool,
) -> None:
    """Recreate a FIXED boundary, closed or open, with its field conditions.

    Parameters
    ----------
    block : phipict._C.Block
        Block of the boundary.
    bound_idx : int
        Boundary index.
    bound_dict : dict[str, Any]
        The boundary's saved settings.
    get_data : callable
        Looks up a saved tensor of a settings dict by name.
    with_scalar : bool
        Whether passive scalars are loaded.
    """
    has_scalar = with_scalar and ("scalar" in bound_dict)
    if bound_dict.get("velocityType", "DIRICHLET") == "NEUMANN":
        # OpenBoundary: Neumann velocity BC (zero-gradient outflow). It takes the
        # scalar only as a Dirichlet value; a Neumann scalar is passed as None.
        scalar_type = bound_dict.get("passiveScalarType", "NEUMANN")
        if isinstance(scalar_type, list):
            is_scalar_dirichlet = any(t == "DIRICHLET" for t in scalar_type)
        else:
            is_scalar_dirichlet = scalar_type == "DIRICHLET"
        block.OpenBoundary(
            bound_idx,
            get_data(bound_dict, "scalar")
            if has_scalar and is_scalar_dirichlet
            else None,
        )
    else:
        block.CloseBoundary(
            bound_idx,
            get_data(bound_dict, "velocity"),
            get_data(bound_dict, "scalar") if has_scalar else None,
        )
        if has_scalar:
            scalar_types = bound_dict["passiveScalarType"]
            if isinstance(scalar_types, list):
                _fixed_boundary(block, bound_idx).setPassiveScalarType(
                    [boundary_condition_string_to_type(t) for t in scalar_types]
                )
            else:
                _fixed_boundary(block, bound_idx).setPassiveScalarType(
                    boundary_condition_string_to_type(scalar_types)
                )
    _load_potential_bc(
        _fixed_boundary(block, bound_idx),
        bound_dict,
        get_data(bound_dict, "potentialTypes"),
        get_data(bound_dict, "potentialValues"),
    )
