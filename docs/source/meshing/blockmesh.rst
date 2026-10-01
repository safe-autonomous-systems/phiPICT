Multi-block meshes with BlockMesh
=================================

.. currentmodule:: phipict.meshing

Use :class:`BlockMesh` when a geometry needs several blocks that share corners and
edges, e.g. an O-grid around a cylinder inside a channel. You describe each block
by its corners (:class:`Quad` in 2D, :class:`Hex` in 3D) and give cell counts and
gradings only where they matter; :meth:`BlockMesh.build` fills in the rest from
the neighbouring blocks and returns a connected :class:`Mesh`. It works like
OpenFOAM's blockMesh, and :func:`read_blockmeshdict` uses it to read blockMesh
files.

For a single block, or for blocks that do not share edges, the builders on
:doc:`blocks` are simpler.

Building a mesh
---------------

.. code-block:: python

    import phipict.meshing as pm

    walls = pm.Patch("walls", pm.Wall())
    bm = pm.BlockMesh()
    bm.add(pm.Quad(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        cells=(16, 12),
        grading=(None, pm.Symmetric(4.0)),
        patches=pm.FacePatches(y_minus=walls, y_plus=walls),
        name="a",
    ))
    bm.add(pm.Quad(
        [(1, 0), (3, 0), (1, 1), (3, 1)],
        cells=(24, None),              # 12 cells in y, from block a
        patches=pm.FacePatches(y_minus=walls, y_plus=walls),
        name="b",
    ))
    mesh = bm.build()
    print([b.cells for b in mesh])     # [(16, 12), (24, 12)]

What is passed on between blocks
--------------------------------

Corners closer than the merge tolerance (``BlockMesh(merge_tol=1e-9)``, relative
to the mesh size) are the same vertex, and edges between the same two vertices are
the same edge. Along chains of shared edges:

* **Cell counts** must agree. A block may leave a count at ``None`` if a
  neighbour sharing an edge along that axis gives it. Conflicting counts raise an
  error that lists every source.
* **Gradings**: an axis with grading ``None`` takes the grading of a neighbour's
  shared edge, mirrored if that edge runs the other way. Without such a
  neighbour it is uniform. This keeps the faces between blocks conforming.
* **Curved edges** given by one block, or with :meth:`BlockMesh.add_edge`, apply
  to every block with that edge.

Counts are passed along edges, not corners: blocks that only touch at a corner
(like the four blocks of an O-grid, along the angle) need their own counts.

Shared curved edges
-------------------

:meth:`BlockMesh.add_edge` shapes the edge between two corners for all blocks that
have it, in the right direction for each:

.. code-block:: python

    bm = pm.BlockMesh()
    bm.add(pm.Quad([(0, 0), (1, 0), (0, 1), (1, 1)], cells=(8, 8), name="left"))
    bm.add(pm.Quad([(1, 0), (2, 0), (1, 1), (2, 1)], cells=(8, 8), name="right"))
    bm.add_edge((1, 0), (1, 1), pm.Arc(through=(1.2, 0.5)))
    mesh = bm.build()

Example: cylinder in a channel
------------------------------

Four blocks form an O-grid between the cylinder and a square; eight channel
blocks surround it. Only a few counts are given; the patches name the boundaries:

.. code-block:: python

    import math

    r, a, h = 0.5, 1.5, 2.0          # cylinder radius, O-grid half-width, half-height
    arc = pm.Arc(center=(0.0, 0.0))

    def on_circle(deg):
        return (r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg)))

    cyl = pm.Patch("cylinder", pm.Wall())
    walls = pm.Patch("walls", pm.FreeSlip())
    inlet = pm.Patch("inlet", pm.Inflow((1.0, 0.0)))
    outlet = pm.Patch("outlet", pm.Outflow())
    radial = pm.Simple(4.0)            # cells grow 4x away from the cylinder

    bm = pm.BlockMesh()
    bm.add(pm.Quad([on_circle(-45), (a, -a), on_circle(45), (a, a)],       # right
                   cells=(16, 16), grading=(radial, None),
                   edges=pm.QuadEdges(x_minus=arc), patches=pm.FacePatches(x_minus=cyl)))
    bm.add(pm.Quad([on_circle(135), on_circle(45), (-a, a), (a, a)],       # top
                   cells=(16, None), grading=(None, radial),
                   edges=pm.QuadEdges(y_minus=arc), patches=pm.FacePatches(y_minus=cyl)))
    bm.add(pm.Quad([(-a, -a), on_circle(-135), (-a, a), on_circle(135)],   # left
                   cells=(None, 16), grading=(radial.reversed(), None),
                   edges=pm.QuadEdges(x_plus=arc), patches=pm.FacePatches(x_plus=cyl)))
    bm.add(pm.Quad([(-a, -a), (a, -a), on_circle(-135), on_circle(-45)],   # bottom
                   cells=(16, None), grading=(None, radial.reversed()),
                   edges=pm.QuadEdges(y_plus=arc), patches=pm.FacePatches(y_plus=cyl)))

    xs, ys = [-4.0, -a, a, 16.0], [-h, -a, a, h]
    for i in range(3):
        for j in range(3):
            if i == j == 1:
                continue                                  # the O-grid
            fp = pm.FacePatches(
                x_minus=inlet if i == 0 else None,
                x_plus=outlet if i == 2 else None,
                y_minus=walls if j == 0 else None,
                y_plus=walls if j == 2 else None,
            )
            bm.add(pm.Quad(
                [(xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i], ys[j + 1]), (xs[i + 1], ys[j + 1])],
                cells=((12, None, 64)[i], (8, None, 8)[j]),
                patches=fp,
            ))
    mesh = bm.build()
    print(mesh)     # Mesh(2D, 12 blocks, 3712 cells, 16 connections, ...)

``examples/cylinder.py`` is this mesh with graded channel blocks and a flow
simulation.

Periodic patch pairs
--------------------

:meth:`BlockMesh.add_periodic` makes the faces of two patches periodic partners,
like OpenFOAM's ``cyclic`` patches:

.. code-block:: python

    left, right = pm.Patch("left"), pm.Patch("right")
    walls = pm.Patch("walls", pm.Wall())
    bm = pm.BlockMesh()
    bm.add(pm.Quad([(0, 0), (4, 0), (0, 1), (4, 1)], cells=(16, 8),
                   patches=pm.FacePatches(x_minus=left, x_plus=right,
                                          y_minus=walls, y_plus=walls)))
    bm.add_periodic(left, right)
    mesh = bm.build()

3D blocks
---------

:class:`Hex` takes eight corners (solver order, index ``i + 2j + 4k`` for the
corner at ``(i, j, k)``) and curved edges as :class:`CornerEdge` between two corner
indices. Many 3D meshes are simpler as a 2D mesh extruded with
:meth:`Mesh.extrude`.

.. code-block:: python

    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    bm = pm.BlockMesh()
    bm.add(pm.Hex(corners, cells=(8, 8, 4), grading=(None, pm.Symmetric(3.0), None)))
    cube = bm.build()

Inspecting the result
---------------------

:meth:`BlockMesh.resolve` returns the blocks with merged corners and all counts,
gradings and edges filled in, without building them. :func:`write_blockmeshdict`
writes a :class:`BlockMesh` as an OpenFOAM file (see :doc:`io`).

API
---

.. autoclass:: phipict.meshing.BlockMesh
   :members:
   :no-index:

.. autoclass:: phipict.meshing.Quad
   :no-index:

.. autoclass:: phipict.meshing.Hex
   :no-index:
