Blocks
======

.. currentmodule:: phipict.meshing

A :class:`MeshBlock` is a single discretised block: its vertex coordinates, a
name and the patches of its faces. You create blocks with the builders below
(:func:`make_box`, :func:`make_quad`, :func:`make_hexa`, :func:`make_annulus`) when you mesh a
geometry in code, or wrap existing vertex tensors with
:meth:`MeshBlock.from_coords`. The block operations (transform, reindex, split,
join, extend, extrude) are what you use to assemble more complex geometries from
simple blocks; patches move with their faces through all of them.

All operations return new blocks; a block is not changed in place.

The vertices are stored as the solver stores them, ``[dims, (nz+1,) ny+1,
nx+1]`` in float64:

.. code-block:: python

    import torch
    import phipict.meshing as pm
    from phipict import Face

    b = pm.make_box((0, 0), (2, 1), cells=(4, 3))
    print(b.ndims, b.cells, b.n_cells, list(b.coords.shape))   # 2 (4, 3) 12 [2, 4, 5]
    print(b.face_coords(Face.X_PLUS).T)       # vertices of the +x face
    print(b.cell_centers().shape)             # [2, 3, 4], like a cell field

Existing vertex tensors
-----------------------

:meth:`MeshBlock.from_coords` wraps an existing tensor, with or without the batch
dimension of the solver layout. This is the way in for grids made with other
tools, e.g. :mod:`phipict.grid.shapes`:

.. code-block:: python

    x = torch.linspace(0, 1, 5, dtype=torch.float64)
    y = torch.linspace(0, 1, 3, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx, yy])[None]          # [1, 2, 3, 5]
    block = pm.MeshBlock.from_coords(coords, name="imported")

Builders
--------

The builders describe a block by its corners, cells and gradings. Each axis takes
a cell count and a :doc:`grading <grading>` (``None`` for uniform, a number for an
OpenFOAM expansion ratio).

:func:`make_box`
~~~~~~~~~~~~~~~~

An axis-aligned 2D or 3D block, the most common case: channels, ducts, cavities,
the outer blocks of multi-block meshes.

.. code-block:: python

    walls = pm.Patch("walls", pm.Wall())
    duct = pm.make_box(
        (0.0, -1.0, -1.0), (2.0, 1.0, 1.0),
        cells=(50, 40, 40),
        grading=(None, pm.Symmetric(200.0), pm.Symmetric(50.0)),
        patches=pm.FacePatches(y_minus=walls, y_plus=walls, z_minus=walls, z_plus=walls),
        name="duct",
    )

:func:`make_quad` and :func:`make_hexa`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

General quadrilateral (2D) and hexahedral (3D) blocks from their corners, with
optional curved edges (see :doc:`edges`). Corners are listed x fastest: ``(-x-y,
+x-y, -x+y, +x+y)``, and in 3D the same four again at ``+z``.

.. code-block:: python

    # a quadrilateral with a curved bottom edge
    bump = pm.make_quad(
        [(0, 0), (2, 0), (0, 1), (2, 1)],
        cells=(32, 16),
        grading=(None, 3.0),
        edges=pm.QuadEdges(y_minus=pm.Arc(through=(1.0, 0.2))),
    )

    # a hexahedron with one arc edge between corners 0 and 1
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    hexahedron = pm.make_hexa(corners, cells=(8, 8, 8),
                         edges=[pm.CornerEdge(0, 1, pm.Arc(through=(0.5, -0.1, 0.0)))])

The interior follows from the edges by transfinite interpolation. ``cells`` may be
``None`` along an axis whose count is fixed by a :class:`Points` edge or an
:class:`Explicit` grading.

:func:`make_annulus`
~~~~~~~~~~~~~~~~~~~~

A ring segment, e.g. the blocks of an O-grid around a cylinder. The x axis runs
along the angle, the y axis outwards. With ``None`` radial cells, the radial
cells grow with the radius so that they stay roughly square (PICT's
``make_torus_2D``); :func:`annulus_radial_weights` gives that distribution.

.. code-block:: python

    ring = pm.make_annulus((0, 0), r_inner=0.5, r_outer=1.0, start_angle=135.0,
                      angle=-90.0, cells=(24, None))
    print(ring.cells)          # (24, 8): radial cells from the square-cell rule

A clockwise (negative) angle gives a right-handed block; for a counter-clockwise
one, flip an axis afterwards.

Transformations
---------------

Geometric transformations move the vertices:

.. code-block:: python

    b = pm.make_box((0, 0), (2, 1), cells=(8, 4))
    moved = b.translate((1.0, 0.5))
    turned = b.rotate(30.0, center=(1.0, 0.5))     # degrees, counter-clockwise
    scaled = b.scale((2.0, 1.0))
    mirrored = b.mirror("y", at=0.0)               # stays right-handed
    b3 = pm.make_box((0, 0, 0), (1, 1, 1), cells=(4, 4, 4)).rotate(45.0, axis=(1, 0, 0))

:meth:`MeshBlock.mirror` also reverses the vertex order along x, so the mirrored
block keeps positive cell volumes; the patches move with their faces.

Reindexing
----------

:meth:`MeshBlock.flip` and :meth:`MeshBlock.permute` change the order of the
vertices without moving them. The solver does not care about the orientation of
blocks (connections are found in any orientation), so these are needed only to
make a block right-handed, or to match an expected index order:

.. code-block:: python

    walls = pm.Patch("walls")
    b = pm.make_box((0, 0), (2, 1), cells=(8, 4), patches=pm.FacePatches(y_minus=walls))
    f = b.flip("y")               # the wall patch is now on +y, where the face went
    print(f.patches.y_plus.name)  # walls
    p = b.permute("yx")           # x and y swapped: left-handed, flip one axis
    p = p.flip("x")

Splitting and joining
---------------------

:meth:`MeshBlock.split` cuts a block at a vertex index, e.g. to make faces match
a neighbour; :meth:`MeshBlock.concat` joins two blocks with a common face into one
(fewer, larger blocks are faster to solve):

.. code-block:: python

    long = pm.make_box((0, 0), (3, 1), cells=(12, 4))
    first, second = long.split("x", 4)             # 4 and 8 cells along x
    again = first.concat(second, "x")
    assert torch.equal(again.coords, long.coords)

Growing and extruding
---------------------

:meth:`MeshBlock.extend` adds cell layers beyond a face by continuing the grid
lines, e.g. an outflow buffer; :meth:`MeshBlock.extrude` makes a 3D block from a
2D one:

.. code-block:: python

    out = pm.Patch("outlet", pm.Outflow())
    b = pm.make_box((0, 0), (1, 1), cells=(8, 8), patches=pm.FacePatches(x_plus=out))
    longer = b.extend(Face.X_PLUS, length=2.0, cells=8, grading=pm.Simple(3.0))
    print(longer.cells, longer.patches.x_plus.name)     # (16, 8) outlet
    slab = b.extrude((0.0, 0.5), cells=4)                # [3, 5, 9, 9] vertices

For whole meshes, use :meth:`Mesh.extrude`, which also keeps the connections.

API
---

.. autoclass:: phipict.meshing.MeshBlock
   :members:
   :no-index:

.. autofunction:: phipict.meshing.make_box
   :no-index:

.. autofunction:: phipict.meshing.make_quad
   :no-index:

.. autofunction:: phipict.meshing.make_hexa
   :no-index:

.. autofunction:: phipict.meshing.make_annulus
   :no-index:
