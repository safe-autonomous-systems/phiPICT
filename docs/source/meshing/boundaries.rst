Boundary patches and conditions
===============================

.. currentmodule:: phipict.meshing

Use patches to say what happens at the boundary of the domain: every block face
that is not connected to another block belongs to a :class:`Patch`, which names
the boundary ("inlet", "walls", "cylinder") and carries its condition. You attach
patches to block faces with :class:`FacePatches` when building blocks, and set or
change conditions by patch name with :meth:`Mesh.set_bc`, which is how you give
conditions to meshes read from files. One patch can span many faces of many
blocks.

Patch
-----

A :class:`Patch` is a name, a condition (``bc``) and, for imported patches, the
type it had in the file (``kind``). A mesh keeps one patch per name; blocks that
use patches of the same name share the same patch.

.. code-block:: python

    import torch
    import phipict.meshing as pm

    walls = pm.Patch("walls", pm.Wall())
    inlet = pm.Patch("inlet")                 # condition set later
    print(walls, inlet)

FacePatches
-----------

:class:`FacePatches` assigns patches to the faces of one block. It is a small
typed record with one field per face (``x_minus``, ``x_plus``, ``y_minus``,
``y_plus``, ``z_minus``, ``z_plus``) rather than a dictionary, so a misspelled
face is an error. Faces left at ``None`` are to be connected or made periodic.

.. code-block:: python

    from phipict import Face

    fp = pm.FacePatches(x_minus=inlet, y_minus=walls, y_plus=walls)
    print(fp[Face.Y_MINUS].name)                     # walls
    fp = fp.set(Face.X_PLUS, pm.Patch("outlet", pm.Outflow()))
    all_walls = pm.FacePatches.uniform(walls, ndims=2)   # every face

Block operations keep the patches with their faces: flipping or permuting the axes
of a block moves the patches accordingly (see :doc:`blocks`).

Conditions
----------

The conditions are small frozen dataclasses:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Condition
     - Meaning
     - Solver setup
   * - :class:`Wall`
     - no-slip wall, optionally moving (``velocity``)
     - ``CloseBoundary``, Dirichlet velocity
   * - :class:`FreeSlip`
     - free-slip wall or symmetry plane
     - ``OpenBoundary`` (zero normal velocity, zero tangential gradient)
   * - :class:`Inflow`
     - prescribed velocity
     - ``CloseBoundary``, Dirichlet velocity
   * - :class:`Outflow`
     - advected outflow
     - ``CloseBoundary`` with a varying velocity, balanced against the inflow

A velocity can be a vector, a tensor with one value per face cell
(``[1, dims, *face]``), or a **profile**: a function that maps the centres of the
face cells ``[n, dims]`` to velocities ``[n, dims]``. Profiles are evaluated when
the domain is made, so they need no knowledge of the face resolution:

.. code-block:: python

    def parabola(p: torch.Tensor) -> torch.Tensor:
        u = 1.5 * (1.0 - p[:, 1] ** 2)           # channel between y = -1 and 1
        return torch.stack([u, torch.zeros_like(u)], dim=1)

    inlet = pm.Patch("inlet", pm.Inflow(parabola))
    lid = pm.Patch("lid", pm.Wall(velocity=(1.0, 0.0)))   # moving wall

Other fields
~~~~~~~~~~~~

Conditions for the passive scalar and the electric potential are passed as
``extra`` specs of :mod:`phipict.bc` and applied with :func:`phipict.bc.set_bc`
after the velocity condition:

.. code-block:: python

    from phipict import bc

    hartmann_walls = pm.Patch(
        "hartmann_walls", pm.Wall(extra=(bc.Potential.ThinWall(cw=0.1),))
    )
    heated = pm.Patch(
        "heated", pm.Wall(extra=(bc.Scalar.Dirichlet(torch.tensor([[1.0]])),))
    )

Setting conditions by name
--------------------------

:meth:`Mesh.set_bc` sets the condition of a patch. A misspelled name raises a
``KeyError`` that suggests the closest names:

.. code-block:: python

    block = pm.make_box((0, -1), (4, 1), cells=(16, 8),
                   patches=pm.FacePatches(x_minus=inlet, x_plus=pm.Patch("outlet"),
                                          y_minus=walls, y_plus=walls))
    mesh = pm.Mesh([block])
    mesh.set_bc("outlet", pm.Outflow())
    try:
        mesh.set_bc("outlt", pm.Outflow())
    except KeyError as err:
        print(err)          # No patch called 'outlt'. Did you mean 'outlet'? ...

:meth:`Mesh.get_domain` refuses meshes with patches that have no condition and
lists them. :meth:`Mesh.assign_patch` puts further faces into a patch, e.g. all
remaining free faces.

Outflow
-------

The solver needs the boundary fluxes of a domain to balance, and an outflow has to
move with the flow. :class:`Outflow` handles both:

* its initial velocity is the outward face normal (or a given ``velocity``),
  which :meth:`Mesh.get_domain` scales so that the outflow carries exactly the
  inflow,
* :meth:`Mesh.outflow_hook` returns the callback that advects the outflow
  velocity before every step.

.. code-block:: python

    import phipict
    from phipict import Hook, Hooks

    domain = mesh.get_domain(viscosity=0.01, dtype=torch.float64, device="cuda",
                             passive_scalar_channels=0)
    hooks = Hooks().append(Hook.PRE, mesh.outflow_hook(domain, velocity=1.0))
    sim = phipict.Simulation(domain=domain, dt=0.01, hooks=hooks)

API
---

.. autoclass:: phipict.meshing.Patch
   :no-index:

.. autoclass:: phipict.meshing.FacePatches
   :members:
   :no-index:

.. autoclass:: phipict.meshing.Wall
   :no-index:

.. autoclass:: phipict.meshing.FreeSlip
   :no-index:

.. autoclass:: phipict.meshing.Inflow
   :no-index:

.. autoclass:: phipict.meshing.Outflow
   :no-index:
