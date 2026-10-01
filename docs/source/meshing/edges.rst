Curved edges
============

.. currentmodule:: phipict.meshing

Block edges are straight by default. Give an edge a shape when the geometry is
curved: the arc of a cylinder, a bump in a channel wall, an airfoil surface. The
shape describes the edge *between* its two corners, so it always ends exactly at
the corners, and the interior of the block follows the curved edges by
transfinite interpolation.

Edges are attached with :class:`QuadEdges` in :func:`make_quad` and :class:`Quad`
(named by the face they form), with :class:`CornerEdge` in :func:`make_hexa` and
:class:`Hex`, and with :meth:`BlockMesh.add_edge` for edges shared by several
blocks.

.. code-block:: python

    import math
    import torch
    import phipict.meshing as pm

    block = pm.make_quad(
        [(0, 0), (2, 0), (0, 1), (2, 1)],
        cells=(32, 16),
        edges=pm.QuadEdges(y_minus=pm.Arc(through=(1.0, 0.15))),   # a bump
    )

In 2D, ``y_minus``/``y_plus`` run along x (from ``-x`` to ``+x``), ``x_minus``/
``x_plus`` along y (from ``-y`` to ``+y``); points of a shape are given in that
direction.

Shapes
------

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Shape
     - Use it for
   * - :class:`Line`
     - straight edges (the default)
   * - :class:`Arc`
     - circular arcs, given by a point on the arc (``through``) or the centre
       (``center``)
   * - :class:`Polyline`
     - straight segments through interior points
   * - :class:`Spline`, :class:`BSpline`
     - smooth curves through (or, for B-splines, near) interior points
   * - :class:`Sampled`
     - a densely sampled curve (e.g. from CAD), resampled with the grading
   * - :class:`Points`
     - given vertex positions, used as they are
   * - :class:`Parametric`
     - any curve given as a function ``c(t)``, ``t`` in [0, 1]

.. code-block:: python

    around_origin = pm.Arc(center=(0.0, 0.0))
    through_point = pm.Arc(through=(0.0, 1.0))
    zigzag = pm.Polyline([(0.5, 0.1), (1.0, -0.1), (1.5, 0.1)])
    smooth = pm.Spline([(0.5, 0.1), (1.0, -0.1), (1.5, 0.1)])
    sine = pm.Parametric(
        lambda t: torch.stack([2.0 * t, 0.1 * torch.sin(2 * math.pi * t)], dim=1)
    )

Vertices are placed along the curve by arc length, so a grading distributes them
along a curved edge as it would along a straight one.

Measured and precomputed surfaces
---------------------------------

When the vertices of an edge are known, e.g. the points of an airfoil surface, use
:class:`Points`: they become the vertices of the edge exactly, and their number
fixes the cell count of that axis (leave it at ``None``). :class:`Sampled`
instead treats the points as a curve and places the vertices with the grading:

.. code-block:: python

    t = torch.linspace(0.0, math.pi, 41, dtype=torch.float64)
    surface = torch.stack([1.0 - torch.cos(t), 0.2 * torch.sin(t)], dim=1)   # 41 points

    over_surface = pm.make_quad(
        [surface[0], surface[-1], (0.0, 1.0), (2.0, 1.0)],
        cells=(None, 16),                               # 40 cells along x, from the points
        grading=(None, pm.Simple(8.0)),
        edges=pm.QuadEdges(y_minus=pm.Points(surface)),
    )
    resampled = pm.make_quad(
        [surface[0], surface[-1], (0.0, 1.0), (2.0, 1.0)],
        cells=(24, 16),
        grading=(pm.Symmetric(3.0), pm.Simple(8.0)),
        edges=pm.QuadEdges(y_minus=pm.Sampled(surface)),
    )

Per-edge gradings
-----------------

An :class:`Edge` combines a shape with a grading of its own, overriding the
grading of its axis for that edge. This is needed when opposite edges of a block
must be distributed differently, e.g. an arc that meets a uniformly spaced ring
while the opposite edge meets a graded block:

.. code-block:: python

    transition = pm.make_quad(
        [(0.35, -0.35), (1.5, -1.5), (0.35, 0.35), (1.5, 1.5)],
        cells=(8, 16),
        grading=(None, pm.Symmetric(2.0)),          # +x edge: graded
        edges=pm.QuadEdges(
            x_minus=pm.Edge(pm.Arc(center=(0.0, 0.0)), pm.Uniform()),   # arc: uniform
        ),
    )

3D edges
--------

In 3D, :class:`CornerEdge` names an edge by its two corner indices (solver order,
``i + 2j + 4k`` for the corner at ``(i, j, k)``); the shape runs from the first to
the second:

.. code-block:: python

    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    dome = pm.make_hexa(
        corners,
        cells=(8, 8, 8),
        edges=[
            pm.CornerEdge(4, 5, pm.Arc(through=(0.5, 0.0, 1.2))),
            pm.CornerEdge(6, 7, pm.Arc(through=(0.5, 1.0, 1.2))),
        ],
    )

Interior interpolation
----------------------

The interior vertices follow from the edges by transfinite (Coons)
interpolation, as in blockMesh. For reproducing grids made with PICT's
``generate_grid_vertices_2D``, 2D blocks can use its row interpolation instead:
``pm.make_quad(..., interpolation=pm.Interpolation.PICT_ROWS)``. For the common cases
(at most one curved edge per pair of opposite edges) both give the same vertices.

API
---

.. automodule:: phipict.meshing.curves
   :members:
   :no-index:

.. autoclass:: phipict.meshing.Edge
   :no-index:

.. autoclass:: phipict.meshing.QuadEdges
   :no-index:

.. autoclass:: phipict.meshing.CornerEdge
   :no-index:
