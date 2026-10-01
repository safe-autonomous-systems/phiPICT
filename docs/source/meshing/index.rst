Meshing
=======

:mod:`phipict.meshing` creates the structured multi-block meshes that phiPICT
solves on, reads them from files, and turns them into a :class:`~phipict.Domain`.
You describe the geometry, the resolution and the boundary conditions; block
connections, face orientations and the solver setup follow from that.

.. code-block:: python

    import torch
    import phipict.meshing as pm

    walls = pm.Patch("walls", pm.Wall())
    channel = pm.make_box(
        (0.0, -1.0), (10.0, 1.0),
        cells=(128, 64),
        grading=(None, pm.Symmetric(20.0)),
        patches=pm.FacePatches(y_minus=walls, y_plus=walls),
    )
    mesh = pm.Mesh([channel])
    mesh.make_periodic("x")
    print(mesh.check())
    domain = mesh.get_domain(viscosity=1e-2, dtype=torch.float64, device="cuda")

Blocks, connections and patches
-------------------------------

A mesh consists of structured blocks, logically rectangular grids of cells. Each
face of a block (``-x``, ``+x``, ``-y``, ``+y``, and ``-z``, ``+z`` in 3D) is
either **connected** to a face of another block, **periodic**, or part of a
boundary **patch** with a condition (wall, inflow, outflow, ...). Connections are
found from the geometry, so what you specify are the blocks and the patches.

How the parts fit together
--------------------------

.. code-block:: text

    Grading, EdgeShape            how vertices are spaced along an edge, and its shape
          |
          v
    box / quad / hexa / annulus   one block from corners, cells, gradings, edges
    BlockMesh (Quad, Hex)         many blocks sharing corners, resolution inferred
          |
          v
    MeshBlock                     one discretised block: vertices + face patches
          |
          v
    Mesh                          blocks + connections + patches with conditions
      |   |   |                   (connections are found from the geometry)
      |   |   +--> get_domain()   the solver Domain
      |   +------> check()        cell quality, open faces, missing conditions
      +----------> io.export      VTK files for ParaView
    read_blockmeshdict, read_vtk  meshes from files

From the top, and where you need each part:

:doc:`mesh`
    :class:`~phipict.meshing.Mesh`: always. It holds the finished mesh; you set
    boundary conditions on it, make it periodic, check it and create the domain.

:doc:`boundaries`
    :class:`~phipict.meshing.Patch` and the conditions
    (:class:`~phipict.meshing.Wall`, :class:`~phipict.meshing.Inflow`, ...):
    whenever the domain has walls, inflows or outflows.

:doc:`connections`
    Periodic boundaries, faces that do not connect, explicit connections. Plain
    connections between neighbouring blocks need nothing.

:doc:`blocks`
    :class:`~phipict.meshing.MeshBlock` and its builders: to mesh a geometry in
    code, one block at a time, or to use existing vertex tensors.

:doc:`blockmesh`
    :class:`~phipict.meshing.BlockMesh`: for geometries of many blocks (e.g. an
    O-grid around a cylinder), where corners are shared and cell counts are
    passed on between neighbouring blocks.

:doc:`grading` and :doc:`edges`
    To refine cells towards walls, and to give blocks curved or measured edges.

:doc:`quality`
    To judge a mesh before running it.

:doc:`io`
    To read OpenFOAM ``blockMeshDict`` and VTK files, and to look at meshes and
    results in ParaView.

:doc:`migration`
    If you have code using :mod:`phipict.grid.shapes`.

Three ways to a mesh
--------------------

1. **From existing vertex tensors**, e.g. grids made with
   :mod:`phipict.grid.shapes`:
   ``pm.Mesh([pm.MeshBlock.from_coords(coords)])``.
2. **Built in code**: single blocks with :func:`~phipict.meshing.make_box`,
   :func:`~phipict.meshing.make_quad`, :func:`~phipict.meshing.make_hexa` and
   :func:`~phipict.meshing.make_annulus`, or many blocks with
   :class:`~phipict.meshing.BlockMesh`.
3. **Read from a file**: :func:`~phipict.meshing.read_blockmeshdict` and
   :func:`~phipict.meshing.read_vtk`.

All three give a :class:`~phipict.meshing.Mesh`, which is used the same way from
there on.

Conventions
-----------

* Mesh geometry is kept in float64 on the CPU; :meth:`Mesh.get_domain
  <phipict.meshing.Mesh.get_domain>` converts it to the precision and device of
  the domain.
* Vertices are stored like the solver stores them, ``[dims, (nz+1,) ny+1,
  nx+1]``: coordinate channels x, y, z first, then the grid in z, y, x order.
* Corners of a block are listed in the solver's order, x fastest: ``(-x-y, +x-y,
  -x+y, +x+y)`` in 2D, followed by the same four at ``+z`` in 3D.
* Blocks should be right-handed (positive cell volumes);
  :meth:`Mesh.check <phipict.meshing.Mesh.check>` reports blocks that are not.
* Faces are addressed with :class:`phipict.Face` (``Face.X_MINUS`` ...) or, on
  :class:`~phipict.meshing.FacePatches`, by name (``x_minus`` ...).

.. toctree::
   :maxdepth: 1
   :hidden:

   mesh
   boundaries
   connections
   blocks
   blockmesh
   grading
   edges
   quality
   io
   migration
