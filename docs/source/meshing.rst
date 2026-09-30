Meshing
=======

:mod:`phipict.meshing` builds the structured multi-block meshes the solver runs on,
reads them from files and turns them into a :class:`~phipict.Domain`. A
:class:`~phipict.meshing.Mesh` holds blocks of vertex coordinates, the connections
between the blocks and named boundary patches with their conditions:

.. code-block:: python

    import phipict.meshing as pm

    mesh = ...                        # built, or read from a file (see below)
    print(mesh.check())               # cell quality, open faces, missing conditions
    domain = mesh.get_domain(viscosity=1e-3, dtype=torch.float64, device="cuda")

Blocks are connected wherever their faces coincide, in whatever orientation, so
face indices and the axis codes of ``Block.ConnectBlock`` are never written by
hand. Every other face must belong to a :class:`~phipict.meshing.Patch`.

Single blocks
-------------

A block is described by its extent, the number of cells per axis and a grading per
axis. The builders return a :class:`~phipict.meshing.MeshBlock`:

.. code-block:: python

    walls = pm.Patch("walls", pm.Wall())
    channel = pm.box(
        (0.0, -1.0), (10.0, 1.0),
        cells=(128, 64),
        grading=(None, pm.Symmetric(20.0)),       # fine at both walls
        patches=pm.FacePatches(y_minus=walls, y_plus=walls),
    )
    mesh = pm.Mesh([channel])
    mesh.make_periodic("x")

    duct = pm.Mesh([pm.box(
        (0.0, -1.0, -1.0), (2.0, 1.0, 1.0),
        cells=(50, 40, 40),
        grading=(None, pm.Symmetric(200.0), pm.Symmetric(50.0)),
        patches=pm.FacePatches(y_minus=walls, y_plus=walls, z_minus=walls, z_plus=walls),
    )])
    duct.make_periodic("x")

Gradings (``None`` is uniform, a number is an OpenFOAM expansion ratio):

.. list-table::
   :widths: 30 70

   * - :class:`~phipict.meshing.Uniform`
     - equal cells
   * - :class:`~phipict.meshing.Simple`
     - OpenFOAM ``simpleGrading``: last cell / first cell
   * - :class:`~phipict.meshing.Symmetric`
     - fine at both ends, centre / end cell ratio
   * - :class:`~phipict.meshing.MultiGrading`
     - OpenFOAM multi-grading segments
   * - :class:`~phipict.meshing.Geometric`
     - constant cell-to-cell ratio (PICT ``make_weights_exp``)
   * - :class:`~phipict.meshing.FirstCell`
     - prescribed size of the first cell, e.g. for y+
   * - :class:`~phipict.meshing.Tanh`
     - tanh stretching
   * - :class:`~phipict.meshing.Cosine`
     - cosine (Chebyshev) spacing
   * - :class:`~phipict.meshing.ChebyshevBlend`
     - blend of sine and uniform spacing
   * - :class:`~phipict.meshing.Explicit`
     - given vertex weights

General quadrilaterals and hexahedra take their corners in the solver's order
(x fastest: ``-x-y, +x-y, -x+y, +x+y``, then the same at ``+z``) and optional
curved edges. An edge can also have its own grading, and
:class:`~phipict.meshing.Points` edges use given vertices as they are, e.g. the
points of an airfoil surface:

.. code-block:: python

    block = pm.quad(
        [(-0.5, -0.7), surface[0], (-0.5, 0.7), surface[-1]],
        cells=(95, None),                          # along y: from the points
        grading=(pm.Geometric(0.97), None),
        edges=pm.QuadEdges(x_plus=pm.Points(surface)),
    )
    ring = pm.annulus((0, 0), 0.5, 1.0, start_angle=135, angle=-90, cells=(24, None))

Edge shapes are :class:`~phipict.meshing.Arc` (through a point or around a
centre), :class:`~phipict.meshing.Polyline`, :class:`~phipict.meshing.Spline`,
:class:`~phipict.meshing.BSpline`, :class:`~phipict.meshing.Sampled` (resampled
by the grading), :class:`~phipict.meshing.Points` and
:class:`~phipict.meshing.Parametric`. The interior follows by transfinite
interpolation. Blocks can be transformed (``translate``, ``rotate``, ``scale``,
``mirror``), reindexed (``flip``, ``permute``), split, joined (``concat``),
extended beyond a face (``extend``) and extruded to 3D (``extrude``, or
:meth:`Mesh.extrude <phipict.meshing.Mesh.extrude>` for a whole mesh).

Existing vertex tensors, e.g. from :mod:`phipict.grid.shapes`, become blocks with
:meth:`MeshBlock.from_coords <phipict.meshing.MeshBlock.from_coords>`.

Multi-block meshes
------------------

:class:`~phipict.meshing.BlockMesh` works like OpenFOAM's blockMesh: blocks are
given by their corners, corners closer than a tolerance are shared, and along
chains of shared edges

* the number of cells must agree; an unknown count (``None``) is inferred,
* an axis without a grading takes the grading of a neighbour's shared edge,
* a curved edge applies to every block that has it.

.. code-block:: python

    bm = pm.BlockMesh()
    bm.add(pm.Quad([(0, 0), (1, 0), (0, 1), (1, 1)], cells=(16, 12),
                   grading=(None, pm.Symmetric(4.0)), name="a"))
    bm.add(pm.Quad([(1, 0), (3, 0), (1, 1), (3, 1)], cells=(24, None), name="b"))
    bm.add_edge((1, 0), (1, 1), pm.Arc(through=(1.1, 0.5)))   # shared by a and b
    mesh = bm.build()

The cylinder example below meshes a cylinder in a channel with twelve blocks in
this way. Blocks that do not conform (different resolution or grading along a
shared face) raise :class:`~phipict.meshing.NonConformingInterfaceError` with the
blocks and the reason.

Periodicity
-----------

:meth:`Mesh.make_periodic <phipict.meshing.Mesh.make_periodic>` connects faces
that coincide after a shift, along the full extent of an axis (``"x"``) or by a
given ``translation``. Periodicity of a single block along one axis becomes
``Block.MakePeriodic``, all other cases a connection between blocks.
:meth:`BlockMesh.add_periodic <phipict.meshing.BlockMesh.add_periodic>` pairs two
patches instead (OpenFOAM ``cyclic``).

Boundary conditions
-------------------

A :class:`~phipict.meshing.Patch` names faces and carries a condition:

.. list-table::
   :widths: 30 70

   * - :class:`~phipict.meshing.Wall`
     - no-slip, optionally moving
   * - :class:`~phipict.meshing.FreeSlip`
     - free-slip wall or symmetry plane
   * - :class:`~phipict.meshing.Inflow`
     - prescribed velocity: a vector, a field, or a profile ``fn(face_cell_centres) -> velocities``
   * - :class:`~phipict.meshing.Outflow`
     - advected outflow; balanced against the inflow when the domain is made, advanced by :meth:`Mesh.outflow_hook <phipict.meshing.Mesh.outflow_hook>`

Conditions of other fields go into ``extra`` as specs of :mod:`phipict.bc`, e.g.
``pm.Wall(extra=(bc.Potential.ThinWall(cw=0.1),))``. Patches of imported meshes
have names only; their conditions are set by name with
:meth:`Mesh.set_bc <phipict.meshing.Mesh.set_bc>`, which suggests close names on a
typo. :meth:`Mesh.boundaries <phipict.meshing.Mesh.boundaries>` gives the solver
boundaries of a patch.

.. code-block:: python

    hooks = Hooks().append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0))
    sim = phipict.Simulation(domain=domain, dt=0.01, hooks=hooks)

Importing meshes
----------------

:func:`~phipict.meshing.read_blockmeshdict` reads OpenFOAM ``blockMeshDict``
files: vertices (with ``convertToMeters``/``scale`` and ``$macros``), ``hex``
blocks with ``simpleGrading`` or ``edgeGrading`` including multi-grading,
``arc``/``polyLine``/``spline``/``BSpline`` edges, ``boundary`` patches and
``cyclic`` pairs. A mesh with one cell between two ``empty`` patches is read as
2D. ``wall`` patches become walls and ``symmetry``/``symmetryPlane`` free-slip;
the conditions of other patches are set by the user. Code directives
(``#calc``, ``#codeStream``, ...), projections and ``mergePatchPairs`` are not
supported and raise an error.
:func:`~phipict.meshing.write_blockmeshdict` writes a
:class:`~phipict.meshing.BlockMesh` back.

:func:`~phipict.meshing.read_vtk` reads VTK structured grids (``.vts`` and
``.vtm`` multi-block files, e.g. written by
:func:`~phipict.io.export.write_vtk` or ParaView), and other VTK files through
pyvista (``pip install phipict[mesh]``).

A general OpenFOAM ``polyMesh`` is unstructured and cannot be imported: the solver
needs structured blocks.

Export
------

:mod:`phipict.io.export` writes meshes and solutions as VTK files for ParaView,
without further dependencies:

.. code-block:: python

    from phipict.io import export

    export.write_vtk(mesh, "out/mesh")                  # cell volume, scaled Jacobian
    export.write_vtk(domain, "out/solution", mesh=mesh) # fields and named patches

    series = export.VTKSeries("out/run", mesh=mesh)     # .pvd time series
    hooks = Hooks().append(Hook.POST, series.hook(every=50))
