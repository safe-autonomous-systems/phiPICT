Connections and periodicity
===========================

.. currentmodule:: phipict.meshing

Blocks are connected wherever their faces coincide, in any orientation, so you
usually do nothing: place the blocks so that neighbouring faces share their
vertices and the :class:`Mesh` connects them. This page is for the cases where you
need more: periodic boundaries, faces that should connect but do not, and
explicit connections.

How faces are matched
---------------------

:meth:`Mesh.auto_connect` (called by the :class:`Mesh` constructor and
:meth:`BlockMesh.build`) looks at every face that has no patch and no connection:

1. Faces whose corners coincide are candidates.
2. For each candidate pair, every relative orientation of the in-face axes is
   tried (2 in 2D; 8 in 3D: swapped or not, each axis flipped or not). The pair is
   connected if all vertices coincide under one of them, within ``rel_tol`` times
   the smallest vertex spacing of the faces.
3. The result is a :class:`Connection`, from which
   :meth:`Connection.axis_codes` derives the face and axis codes of the solver's
   ``Block.ConnectBlock``, which you therefore never write yourself.

.. code-block:: python

    import phipict.meshing as pm
    from phipict import Face

    a = pm.make_box((0, 0), (1, 1), cells=(4, 4), name="a")
    # the same neighbour, but with its vertex order reversed along both axes
    b = pm.make_box((1, 0), (2, 1), cells=(4, 4), name="b").flip("x").flip("y")
    mesh = pm.Mesh([a, b])
    (c,) = mesh.connections
    print(c.face_a.name, c.face_b.name, c.axis_codes(2))
    # X_PLUS X_PLUS (3, -1): y of b, running the other way

Every relative orientation gives the same flow: the test suite checks the solver
result of a two-block channel for all 4 orientations in 2D and all 24 in 3D.

Faces that do not match
-----------------------

A face that shares its corners with another face but not its vertices is a
modelling error: the blocks have different resolutions or gradings along the
interface. The solver cannot connect such faces, so this raises
:class:`NonConformingInterfaceError` with the blocks and the reason:

.. code-block:: python

    coarse = pm.make_box((0, 0), (1, 1), cells=(4, 4), name="coarse")
    fine = pm.make_box((1, 0), (2, 1), cells=(4, 6), name="fine")
    try:
        pm.Mesh([coarse, fine])
    except pm.NonConformingInterfaceError as err:
        print(err)
    # Face X_PLUS of block 'coarse' and face X_MINUS of block 'fine' share their
    # corners but are not conforming: [4] vs [6] cells; match the resolutions ...

Faces that touch only partially are not connected and show up as free faces in
:meth:`Mesh.check`. Split a block (:meth:`MeshBlock.split`) so that the faces
match.

Periodicity
-----------

:meth:`Mesh.make_periodic` connects free faces that coincide after a shift:

.. code-block:: python

    walls = pm.Patch("walls", pm.Wall())
    fp = pm.FacePatches(y_minus=walls, y_plus=walls)
    channel = pm.Mesh([
        pm.make_box((0, -1), (2, 1), cells=(8, 8), patches=fp, name="upstream"),
        pm.make_box((2, -1), (6, 1), cells=(16, 8), patches=fp, name="downstream"),
    ])
    channel.make_periodic("x")               # over the full extent in x
    # the same: channel.make_periodic(translation=(6.0, 0.0))

A periodic connection of a block with itself along one axis becomes
``Block.MakePeriodic``; all others (as between ``upstream`` and ``downstream``
here) become ``ConnectBlock`` calls. :meth:`Mesh.extrude` makes the extruded mesh
periodic in z unless patches for the z faces are given, and
:meth:`BlockMesh.add_periodic` pairs two patches like OpenFOAM's ``cyclic``
patches.

Explicit connections
--------------------

:meth:`Mesh.connect` connects two given faces. The orientation is still found
from the geometry unless it is passed, so this is only needed for faces that do
not coincide:

.. code-block:: python

    left = pm.make_box((0, 0), (1, 1), cells=(4, 4), name="left")
    right = pm.make_box((1, 0), (2, 1), cells=(4, 4), name="right")
    mesh = pm.Mesh([left, right], auto_connect=False)
    mesh.connect("left", Face.X_PLUS, "right", Face.X_MINUS)

Tolerance
---------

``rel_tol`` (default ``1e-4``) is relative to the smallest vertex spacing of the
faces, so it adapts to the resolution. Meshes from files with few significant
digits may need a larger value: ``pm.Mesh(blocks, rel_tol=1e-3)``.

API
---

.. autoclass:: phipict.meshing.Connection
   :members:
   :no-index:

.. autoclass:: phipict.meshing.Orientation
   :no-index:

.. autoclass:: phipict.meshing.NonConformingInterfaceError
   :no-index:
