Grading
=======

.. currentmodule:: phipict.meshing

A grading sets how the vertices are distributed along an edge. You need one
whenever cells should not be equal: fine cells at walls to resolve boundary
layers (Hartmann layers, the viscous sublayer), fine cells around an obstacle,
coarse cells far away. Every block builder takes one grading per axis, and
individual edges can have their own (see :doc:`edges`).

Wherever a grading is expected, ``None`` means uniform and a number means an
OpenFOAM expansion ratio (:class:`Simple`):

.. code-block:: python

    import phipict.meshing as pm

    wall_refined = pm.make_box((0, -1), (10, 1), cells=(64, 48),
                          grading=(None, pm.Symmetric(20.0)))
    stretched = pm.make_box((0, 0), (5, 1), cells=(40, 10), grading=(4.0, None))

A grading computes the normalised vertex positions for a number of cells:

.. code-block:: python

    print(pm.Simple(4.0).weights(4))
    # tensor([0.0000, 0.1098, 0.2841, 0.5608, 1.0000], dtype=torch.float64)

Which grading to use
--------------------

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Grading
     - Use it for
   * - :class:`Uniform`
     - equal cells (the default)
   * - :class:`Simple`
     - geometric growth along the edge, given as last cell / first cell
       (OpenFOAM ``simpleGrading``); ``Simple(4)`` refines the start,
       ``Simple(0.25)`` the end
   * - :class:`Symmetric`
     - refinement at both ends, e.g. a channel between two walls; ratio of the
       centre cells to the end cells
   * - :class:`FirstCell`
     - a prescribed height of the first cell, e.g. for a target y+; the growth is
       solved for
   * - :class:`MultiGrading`
     - several geometric segments, e.g. fine in a shear layer in the middle of an
       edge (OpenFOAM multi-grading)
   * - :class:`Geometric`
     - a constant cell-to-cell ratio (PICT's ``make_weights_exp``)
   * - :class:`Tanh`, :class:`Cosine`, :class:`ChebyshevBlend`
     - smooth clustering laws common in spectral and DNS grids
   * - :class:`Explicit`
     - vertex positions computed elsewhere; fixes the cell count

Refinement at walls
-------------------

.. code-block:: python

    y = pm.Symmetric(10.0).weights(20)          # both ends 10x finer than centre
    wall = pm.FirstCell(1e-3).weights(40, length=2.0)
    print(float(wall[1] * 2.0))                  # 0.001: first cell height
    both = pm.FirstCell(1e-3, pm.Cluster.BOTH)   # the same at both ends

``FirstCell`` needs the length of the edge, which the builders pass in.

Segments
--------

:class:`MultiGrading` combines geometric segments given as (relative length,
relative cell count, expansion ratio), exactly as OpenFOAM does:

.. code-block:: python

    # 20% of the length with 30% of the cells refined towards the start,
    # 60% uniform, 20% refined towards the end
    shear = pm.MultiGrading([(0.2, 0.3, 4.0), (0.6, 0.4, 1.0), (0.2, 0.3, 0.25)])

Clustering laws
---------------

:class:`Tanh`, :class:`Cosine`, :class:`ChebyshevBlend` and :class:`FirstCell`
take a :class:`Cluster` that says where the small cells are: ``START``, ``END`` or
``BOTH``:

.. code-block:: python

    pm.Tanh(2.0)                          # both ends (default)
    pm.Tanh(2.0, pm.Cluster.START)        # the start only
    pm.Cosine()                           # Chebyshev-like spacing
    pm.ChebyshevBlend(0.8)                # 80% sine mapping, 20% uniform

Direction
---------

A grading refers to the direction of its edge or axis. :meth:`Grading.reversed`
gives the same distribution seen from the other end, e.g. for a block whose axis
runs the other way:

.. code-block:: python

    toward_end = pm.Simple(4.0).reversed()     # the same as Simple(0.25)

:class:`BlockMesh` mirrors gradings automatically when it passes them to a
neighbour whose edge runs the other way.

Cell counts from sizes
----------------------

:func:`cells_for_size` gives the number of uniform cells closest to a target size:

.. code-block:: python

    n = pm.cells_for_size(length=22.0, size=0.05)     # 440

API
---

.. automodule:: phipict.meshing.grading
   :members:
   :no-index:
