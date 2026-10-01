Reading and writing meshes
==========================

.. currentmodule:: phipict.meshing

Use the readers to run on meshes made with other tools, and the VTK export to
look at meshes and results in ParaView or pyvista.

OpenFOAM blockMeshDict
----------------------

:func:`read_blockmeshdict` reads an OpenFOAM ``blockMeshDict`` and returns the
:class:`Mesh` blockMesh would create, as phiPICT blocks: one block per ``hex``,
with the same vertices, cell counts, gradings and curved edges. Patches keep their
names; set the conditions of those whose type does not determine one:

.. code-block:: python

    import torch
    import phipict.meshing as pm

    mesh = pm.read_blockmeshdict("examples/meshes/flow_past_cylinder/blockMeshDict")
    print(mesh)
    # Mesh(2D, 9 blocks, 78400 cells, 12 connections, patches=[inlet, side1, side2, outlet])
    for patch in mesh.patches:
        print(patch.name, patch.kind, len(mesh.faces(patch)))

    mesh.set_bc("inlet", pm.Inflow((1.0, 0.0)))
    mesh.set_bc("outlet", pm.Outflow())
    mesh.set_bc("side1", pm.FreeSlip())
    mesh.set_bc("side2", pm.FreeSlip())
    assert mesh.check().ok

What is read, and how it maps:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - blockMeshDict
     - phiPICT
   * - ``vertices``, ``convertToMeters``/``scale``, ``$variables``
     - block corners
   * - ``hex (...) name (nx ny nz)``
     - one block; the optional name becomes the block name
   * - ``simpleGrading``, ``edgeGrading``, multi-grading
     - :class:`Simple` / :class:`MultiGrading` per edge
   * - ``arc`` (point or ``origin``), ``polyLine``, ``spline``, ``BSpline``
     - :class:`Arc`, :class:`Polyline`, :class:`Spline`, :class:`BSpline`
   * - ``wall`` patch
     - :class:`Wall`
   * - ``symmetry``/``symmetryPlane`` patch
     - :class:`FreeSlip`
   * - ``patch`` and other types
     - named patch without a condition, set it with :meth:`Mesh.set_bc`
   * - ``cyclic`` with ``neighbourPatch``
     - periodic connection
   * - ``empty`` front and back, one cell between them
     - a 2D mesh
   * - ``defaultPatch``
     - patch for faces not listed in ``boundary``

Not supported, and reported as an error: ``#calc``, ``#codeStream``, ``#include``
and other directives, projected vertices, edges and faces, ``mergePatchPairs``
with entries (non-conforming interfaces), and collapsed hexes. A general OpenFOAM
``polyMesh`` cannot be read: it is unstructured, and the solver needs structured
blocks.

:func:`load_blockmeshdict <phipict.meshing.formats.load_blockmeshdict>` returns
the unbuilt :class:`BlockMesh`, e.g. to change cell counts before building.
:func:`write_blockmeshdict` writes a :class:`BlockMesh` as a ``blockMeshDict``,
e.g. to run the same blocks in OpenFOAM:

.. code-block:: python

    walls = pm.Patch("walls", pm.Wall())
    bm = pm.BlockMesh()
    bm.add(pm.Quad([(0, 0), (4, 0), (0, 1), (4, 1)], cells=(32, 16),
                   grading=(None, pm.Symmetric(4.0)),
                   patches=pm.FacePatches(y_minus=walls, y_plus=walls,
                                          x_minus=pm.Patch("inlet"),
                                          x_plus=pm.Patch("outlet"))))
    pm.write_blockmeshdict(bm, "blockMeshDict")

VTK
---

:func:`read_vtk` reads VTK structured grids: ``.vts`` files and ``.vtm``
multi-block files of them, as written by :func:`phipict.io.export.write_vtk` or
ParaView. Each grid becomes a block, coinciding faces are connected, and the
patches written by :func:`~phipict.io.export.write_vtk` are restored (without
conditions). Other VTK formats are read with pyvista, if installed
(``pip install phipict[mesh]``).

.. code-block:: python

    from phipict.io import export

    export.write_vtk(mesh, "mesh")          # writes mesh.vtm and mesh/
    again = pm.read_vtk("mesh.vtm")
    print(again)

Export to VTK
-------------

:mod:`phipict.io.export` writes meshes and simulation results as VTK XML files:
one ``.vts`` structured grid per block, collected in a ``.vtm`` multi-block file
that ParaView opens directly. Only numpy and the standard library are used.

.. code-block:: python

    from phipict.io import export

    export.write_vtk(mesh, "out/mesh")      # cell volume and scaled Jacobian as cell data

    domain = mesh.get_domain(viscosity=0.01, dtype=torch.float64, device="cuda",
                             passive_scalar_channels=0)
    export.write_vtk(domain, "out/solution", mesh=mesh)   # fields and named patches

For a domain, the cell fields are chosen with :class:`~phipict.io.export.Fields`
(velocity, pressure, passive scalar, velocity source, electric potential,
viscosity; ``Fields.QUALITY`` adds the quality measures). Passing the ``mesh``
also writes its boundary patches, one grid per patch face, so they can be shown
separately in ParaView. 2D meshes are written as grids of zero thickness in z.

Time series
~~~~~~~~~~~

:class:`~phipict.io.export.VTKSeries` writes snapshots into a ``.pvd`` collection
that ParaView plays as a time series. Its :meth:`~phipict.io.export.VTKSeries.hook`
writes every n-th step during a simulation:

.. code-block:: python

    import phipict
    from phipict import Hook, Hooks

    series = export.VTKSeries("out/run", mesh=mesh)
    hooks = (
        Hooks()
        .append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0))
        .append(Hook.POST, series.hook(every=5))
    )
    sim = phipict.Simulation(domain=domain, dt=0.05, hooks=hooks, pressure_use_amg=True)
    for _ in range(10):
        sim.single_step()

API
---

.. autofunction:: phipict.meshing.read_blockmeshdict
   :no-index:

.. autofunction:: phipict.meshing.write_blockmeshdict
   :no-index:

.. autofunction:: phipict.meshing.read_vtk
   :no-index:

.. autofunction:: phipict.io.export.write_vtk
   :no-index:

.. autoclass:: phipict.io.export.Fields
   :no-index:

.. autoclass:: phipict.io.export.VTKSeries
   :members:
   :no-index:
