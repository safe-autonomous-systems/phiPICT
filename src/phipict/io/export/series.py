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

"""Time series of VTK snapshots, as a ParaView ``.pvd`` collection.

For example::

    series = export.VTKSeries("out/run", mesh=mesh)
    hooks = Hooks().append(Hook.POST, series.hook(every=50))
    sim = phipict.Simulation(domain=domain, hooks=hooks)

or by hand with ``series.write(domain, time)``. The collection file is rewritten
after every snapshot, so it is valid while the simulation runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import quoteattr

import torch

from phipict import _C

from .vtk import Fields, write_vtk

if TYPE_CHECKING:
    from phipict.meshing import Mesh

__all__ = ["VTKSeries"]


class VTKSeries:
    """Snapshots of a domain over time.

    Parameters
    ----------
    path : str or os.PathLike
        Collection file; ``.pvd`` is added if missing. The snapshots go to a
        directory of the same name without the extension.
    fields : Fields, optional
        Cell fields to write. Default is all present fields.
    mesh : Mesh or None, optional
        Mesh of the domain, to also write its boundary patches. Default is None.
    batch_index : int, optional
        Environment to write from a batched domain. Default is 0.
    compress : bool, optional
        Whether to compress the data. Default is True.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fields: Fields = Fields.ALL,
        mesh: Mesh | None = None,
        batch_index: int = 0,
        compress: bool = True,
    ) -> None:
        pvd = Path(path)
        if pvd.suffix != ".pvd":
            pvd = pvd.with_name(pvd.name + ".pvd")
        self.path = pvd
        self.folder = pvd.with_suffix("")
        self.fields = fields
        self.mesh = mesh
        self.batch_index = batch_index
        self.compress = compress
        self.entries: list[tuple[float, Path]] = []

    def write(self, domain: _C.Domain, time: float) -> Path:
        """Write a snapshot and add it to the collection.

        Parameters
        ----------
        domain : phipict.Domain
            The domain.
        time : float
            Simulation time of the snapshot.

        Returns
        -------
        pathlib.Path
            The snapshot's ``.vtm`` file.
        """
        index = len(self.entries)
        vtm = write_vtk(
            domain,
            self.folder / f"step_{index:06d}",
            fields=self.fields,
            mesh=self.mesh,
            batch_index=self.batch_index,
            compress=self.compress,
        )
        self.entries.append((float(time), vtm))
        rows = "".join(
            f'<DataSet timestep="{t!r}" part="0" '
            f"file={quoteattr(os.path.relpath(f, self.path.parent))}/>\n"
            for t, f in self.entries
        )
        self.path.write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0">\n'
            f"<Collection>\n{rows}</Collection>\n</VTKFile>\n"
        )
        return vtm

    def hook(self, every: int = 1) -> Callable[..., None]:
        """A callback for :attr:`Hook.POST <phipict.Hook.POST>` that writes every
        ``every``-th step.

        Parameters
        ----------
        every : int, optional
            Write interval in steps. Default is 1.

        Returns
        -------
        Callable
            The callback.
        """
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}.")
        count = 0

        def callback(domain: _C.Domain, total_time: Any = 0.0, **_: Any) -> None:
            nonlocal count
            if count % every == 0:
                t = total_time
                if isinstance(t, torch.Tensor):  # one time per batched environment
                    t = t.flatten()[min(self.batch_index, t.numel() - 1)]
                self.write(domain, float(t))
            count += 1

        return callback
