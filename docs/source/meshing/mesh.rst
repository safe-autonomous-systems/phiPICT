Mesh
====

.. currentmodule:: phipict.meshing

:class:`Mesh` is the object you work with once the geometry exists: it holds the
blocks, the connections between them and the boundary patches. You use it to set
boundary conditions, make the mesh periodic, check it, and finally create the
solver :class:`~phipict.Domain` with :meth:`Mesh.get_domain`. Every way of making
a mesh (builders, :class:`BlockMesh`, file readers) ends in a :class:`Mesh`.

A mesh is geometry only (float64, on the CPU), so the same mesh can create
domains of different precision, device or batch size.

Creating a mesh
---------------

A mesh is made from :class:`MeshBlock` objects (see :doc:`blocks`), by
:meth:`BlockMesh.build` (see :doc:`blockmesh`) or by a reader (see :doc:`io`).
When blocks are passed to the constructor, faces that coincide are connected:

.. code-block:: python

    import torch
    import phipict.meshing as pm
    from phipict import Face

    walls = pm.Patch("walls", pm.Wall())
    inlet = pm.Patch("inlet", pm.Inflow((1.0, 0.0)))
    outlet = pm.Patch("outlet", pm.Outflow())

    left = pm.make_box((0, 0), (1, 1), cells=(8, 8), name="left",
                  patches=pm.FacePatches(x_minus=inlet, y_minus=walls, y_plus=walls))
    right = pm.make_box((1, 0), (3, 1), cells=(16, 8), name="right",
                   patches=pm.FacePatches(x_plus=outlet, y_minus=walls, y_plus=walls))
    mesh = pm.Mesh([left, right])
    print(mesh)
    # Mesh(2D, 2 blocks, 192 cells, 1 connections, patches=[inlet, walls, outlet])

Blocks can also be added one by one with :meth:`Mesh.add` and connected later
with :meth:`Mesh.auto_connect`; ``pm.Mesh(blocks, auto_connect=False)`` skips the
automatic connection. Blocks with the same name are renamed (``block``,
``block_1``, ...), and :meth:`Mesh.block`/:meth:`Mesh.index` find blocks by name.

Periodicity
-----------

:meth:`Mesh.make_periodic` connects free faces that coincide after a shift, over
the full extent of an axis or by a given translation (see :doc:`connections`):

.. code-block:: python

    channel = pm.Mesh([pm.make_box((0, -1), (4, 1), cells=(32, 16),
                              patches=pm.FacePatches(y_minus=walls, y_plus=walls))])
    channel.make_periodic("x")

Boundary conditions
-------------------

Patches are attached to faces when the blocks are made. Their conditions can be
given with the patch or later by name, which is how conditions are set on meshes
read from files (see :doc:`boundaries`):

.. code-block:: python

    mesh.set_bc("outlet", pm.Outflow())      # replaces the condition of the patch
    print([p.name for p in mesh.patches])
    print(mesh.faces("walls"))               # [(block, Face), ...]

Checking
--------

:meth:`Mesh.check` returns a :class:`MeshReport` with quality measures and a list
of errors: inverted cells, faces that are neither connected nor in a patch, and
patches without a condition (see :doc:`quality`):

.. code-block:: python

    report = mesh.check()
    print(report)
    report.raise_if_invalid()

Creating the domain
-------------------

:meth:`Mesh.get_domain` creates the solver :class:`~phipict.Domain`: one solver
block per mesh block (in the same order and with the same names), the connections
and periodicity, and the boundary conditions of all patches. Outflow velocities
are scaled so that the boundary fluxes balance, and ``PrepareSolve`` is called:

.. code-block:: python

    domain = mesh.get_domain(
        viscosity=0.01,
        dtype=torch.float64,
        device="cuda",
        passive_scalar_channels=0,
    )

For outflow patches, :meth:`Mesh.outflow_hook` returns the callback that advects
the outflow velocity before every step, and :meth:`Mesh.boundaries` gives the
solver boundaries of any patch:

.. code-block:: python

    import phipict
    from phipict import Hook, Hooks

    hooks = Hooks().append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0))
    sim = phipict.Simulation(domain=domain, dt=0.01, hooks=hooks)
    sim.single_step()
    outlet_bounds = mesh.boundaries(domain, "outlet")

:meth:`Mesh.from_domain` goes the other way: it recovers the mesh of an existing
domain (e.g. one loaded with :func:`~phipict.io.load_domain`) with its
connections, for plotting or export.

Changing a mesh
---------------

:meth:`Mesh.extrude` turns a 2D mesh into a 3D one, keeping connections and
patches; by default the result is periodic in z:

.. code-block:: python

    mesh3d = mesh.extrude((0.0, 0.5), cells=4)
    print(mesh3d)

:meth:`Mesh.transform` applies a function to every block and connects the result
again, e.g. to rotate a whole mesh:

.. code-block:: python

    rotated = mesh.transform(lambda block: block.rotate(30.0))

Plotting
--------

:meth:`Mesh.plot` draws the grid lines of a 2D mesh (or of one z layer of a mesh
extruded along z) with matplotlib, one colour per block from the phiPICT palette:

.. code-block:: python

    import matplotlib.pyplot as plt

    ax = mesh.plot()
    plt.savefig("mesh.png")

API
---

.. autoclass:: phipict.meshing.Mesh
   :members:
   :no-index:
