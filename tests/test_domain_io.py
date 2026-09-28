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

"""Robustness of ``save_domain``/``load_domain``."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import phipict
from phipict import _C
from phipict.grid import shapes
from phipict.io.domain_io import load_domain, save_domain

DEVICE = torch.device("cuda")


def _domain(dtype: torch.dtype = torch.float32) -> _C.Domain:
    """A small curvilinear 2D channel, closed in y, periodic in x."""
    grid = shapes.generate_grid_vertices_2D(
        [7, 9],
        [(0.0, -1.0), (2.0, -1.0), (0.0, 1.0), (2.0, 1.0)],
        x_weights=shapes.make_weights("simple", res=6, grading=3, refinement="BOTH"),
        dtype=dtype,
    ).to(DEVICE)
    domain = phipict.Domain(
        2, torch.tensor([0.1], dtype=dtype), name="D", device=DEVICE, dtype=dtype
    )
    block = domain.CreateBlock(vertexCoordinates=grid.contiguous(), name="B")
    block.CloseBoundary("-y")
    block.CloseBoundary("+y")
    block.MakePeriodic("x")
    torch.manual_seed(0)
    block.setVelocity(torch.randn_like(block.velocity).contiguous())
    domain.PrepareSolve()
    return domain


def _assert_same(a: _C.Domain, b: _C.Domain) -> None:
    block_a, block_b = a.getBlock(0), b.getBlock(0)
    assert block_a.velocity.dtype == block_b.velocity.dtype
    assert block_a.velocity.device == block_b.velocity.device
    assert torch.equal(block_a.velocity, block_b.velocity)
    assert torch.equal(block_a.vertexCoordinates, block_b.vertexCoordinates)


def test_roundtrip_keeps_saved_dtype_and_device(tmp_path: Path) -> None:
    domain = _domain(torch.float32)
    save_domain(domain, tmp_path / "domain")
    loaded = load_domain(tmp_path / "domain")
    _assert_same(domain, loaded)
    loaded.PrepareSolve()


def test_load_converts_dtype(tmp_path: Path) -> None:
    save_domain(_domain(torch.float32), tmp_path / "domain")
    loaded = load_domain(tmp_path / "domain", dtype=torch.float64, device=DEVICE)
    assert loaded.getBlock(0).velocity.dtype == torch.float64


def test_rejects_str_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        save_domain(_domain(), str(tmp_path / "domain"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_domain(str(tmp_path / "domain"))  # type: ignore[arg-type]


def test_paths_with_extension_and_dots(tmp_path: Path) -> None:
    """Either file of the pair can be named, and dots in names are kept."""
    domain = _domain()
    base = tmp_path / "new" / "dirs" / "Re2.5_domain"
    save_domain(domain, base.with_name(base.name + ".json"))
    assert sorted(p.name for p in base.parent.iterdir()) == [
        "Re2.5_domain.json",
        "Re2.5_domain.npz",
    ]
    _assert_same(domain, load_domain(base.with_name(base.name + ".npz")))
    _assert_same(domain, load_domain(base))


def test_overwrite_leaves_no_temporary_files(tmp_path: Path) -> None:
    domain = _domain()
    for _ in range(2):
        save_domain(domain, tmp_path / "domain")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["domain.json", "domain.npz"]


def test_missing_file(tmp_path: Path) -> None:
    save_domain(_domain(), tmp_path / "domain")
    (tmp_path / "domain.npz").unlink()
    with pytest.raises(FileNotFoundError, match=r"domain\.npz"):
        load_domain(tmp_path / "domain")


def test_detects_files_of_different_saves(tmp_path: Path) -> None:
    """A JSON next to the tensors of another save (interrupted save) is refused."""
    domain = _domain()
    save_domain(domain, tmp_path / "a")
    save_domain(domain, tmp_path / "b")
    shutil.copy(tmp_path / "b.npz", tmp_path / "a.npz")
    with pytest.raises(ValueError, match="different saves"):
        load_domain(tmp_path / "a")


def test_detects_inconsistent_tensors(tmp_path: Path) -> None:
    save_domain(_domain(), tmp_path / "domain")
    json_path = tmp_path / "domain.json"
    saved = json.loads(json_path.read_text())
    saved["data_info"]["0"]["shape"] = [123]
    json_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="shape"):
        load_domain(tmp_path / "domain")


def test_refuses_newer_format(tmp_path: Path) -> None:
    save_domain(_domain(), tmp_path / "domain")
    json_path = tmp_path / "domain.json"
    saved = json.loads(json_path.read_text())
    saved["format_version"] = 99
    json_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="format version 99"):
        load_domain(tmp_path / "domain")


def test_reads_version_1_files(tmp_path: Path) -> None:
    """Files without format version and save id, all tensors as floats, still load."""
    domain = _domain()
    save_domain(domain, tmp_path / "domain")
    json_path = tmp_path / "domain.json"
    saved = json.loads(json_path.read_text())
    del saved["format_version"], saved["save_id"]
    json_path.write_text(json.dumps(saved))
    with np.load(tmp_path / "domain.npz") as npz:
        arrays = {k: npz[k] for k in npz.files if k != "save_id"}
    np.savez_compressed(tmp_path / "domain.npz", **arrays)
    _assert_same(domain, load_domain(tmp_path / "domain"))


def test_shared_tensors_are_stored_once(tmp_path: Path) -> None:
    domain = _domain()
    block = domain.getBlock(0)
    block.setVelocitySource(block.velocity)  # the same storage twice
    save_domain(domain, tmp_path / "domain")
    saved = json.loads((tmp_path / "domain.json").read_text())
    block_dict = saved["blocks"][0]
    assert block_dict["velocity"] == block_dict["velocitySource"]
