Magnetohydrodynamics
====================

:class:`~phipict.MHDSimulation` extends :class:`~phipict.Simulation` with the
inductionless MHD approximation for a static, imposed magnetic field. Each step
solves a Poisson equation for the electric potential :math:`\phi` and adds the
Lorentz force as a velocity source:

.. math::

    \nabla^2 \phi = \nabla \cdot (\mathbf{u} \times \mathbf{e}_B), \qquad
    \mathbf{j} = -\nabla \phi + \mathbf{u} \times \mathbf{e}_B, \qquad
    \mathbf{F}_L = N \, (\mathbf{j} \times \mathbf{e}_B),

with the Stuart number :math:`N = Ha^2 / Re`.

Setup
-----

.. code-block:: python

    sim = phipict.MHDSimulation(
        domain=domain,
        dt=5e-3,
        stuart_number=torch.tensor(Ha**2 / Re),
        e_b=torch.tensor([0.0, 1.0, 0.0], dtype=dtype),  # field along y
        substeps="ADAPTIVE",
        prep_fn={"PRE_VELOCITY_SETUP": apply_forcing},
        pressure_tol=phipict.SolverTolerance(rtol=1e-6, atol=1e-14),
        potential_tol=phipict.SolverTolerance(rtol=1e-8, atol=1e-14),
    )
    sim.make_divergence_free()

- ``e_b`` always has three components, also in 2D. It is either uniform, shape
  ``(3,)``, or spatially varying, shape ``(3, *spatial)`` with broadcastable
  spatial axes, e.g. ``(3, 1, 1, nx)``. A spatially varying field needs a
  single-block domain.
- ``sim.set_magnetic_field(e_b)`` changes the field between steps. The
  potential is solved again before the next Lorentz force.
- ``stuart_number`` and ``e_b`` are tensors, so gradients can flow into them
  when ``differentiable=True``.
- ``sim.get_current_density(block)`` returns :math:`\mathbf{j}`.

.. note::

    The Lorentz force is **added** to the block's velocity source. If you drive
    the flow with ``setVelocitySource``, reset the source in a
    ``"PRE_VELOCITY_SETUP"`` callback every step (as above). Otherwise the
    Lorentz force accumulates.

Electric boundary conditions
----------------------------

The boundary condition for :math:`\phi` is set per boundary face, independently of
the velocity boundary condition, with :func:`phipict.bc.set_bc`. The face must be
FIXED, i.e. made by ``CloseBoundary`` or ``OpenBoundary``:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Wall type
     - Condition
     - Spec
   * - Insulating (default)
     - :math:`j_n = 0`
     - ``bc.Potential.Insulating()``
   * - Thin conducting wall
     - :math:`\partial_n \phi = C_w \nabla^2_\tau \phi`
     - ``bc.Potential.ThinWall(cw=cw)``
   * - Grounded plane / electrode
     - :math:`\phi = g` (default :math:`g = 0`)
     - ``bc.Potential.Dirichlet(value=g)``
   * - Open in/outflow
     - current may leave the domain
     - ``bc.Potential.Open()``

Faces are addressed with :class:`phipict.Face`. In/outflows made by
``CloseBoundary`` are FIXED faces just like walls, so they must be marked
``Open()`` explicitly.

Conducting walls (:math:`C_w > 0`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wall conductance ratio :math:`C_w = \sigma_w t_w / (\sigma a)` is set per
face after the boundary is closed. For example, for Hunt's flow (conducting
Hartmann walls, insulating side walls):

.. code-block:: python

    from phipict import Face, bc

    for side in ("-y", "+y", "-z", "+z"):
        block.CloseBoundary(side)

    for face in (Face.Y_MINUS, Face.Y_PLUS):
        bc.set_bc(block, face, bc.Potential.ThinWall(cw=0.1))
    block.MakePeriodic("x")
    domain.PrepareSolve()

The thin-wall condition only changes the Poisson matrix of the fluid cells next
to the wall. No extra wall unknowns are added. Like with insulating walls, the
matrix stays singular, and ``potential_normalize=True`` (the default) removes
the mean of :math:`\phi`. A Dirichlet face cell fixes :math:`\phi` so that the
solution is unique, and no normalisation is applied then.

Mixed faces and electrodes
~~~~~~~~~~~~~~~~~~~~~~~~~~

``where`` restricts a condition to some cells of a face. It is a boolean mask
over the face cells, in the block's cell index order with the face normal
dropped (``[NZ, NX]`` for a ``±y`` face of a ``[NZ, NY, NX]`` block):

.. code-block:: python

    electrodes = torch.zeros(nz, nx, dtype=torch.bool)
    electrodes[:, 10:20] = True
    bc.set_bc(block, Face.Y_MINUS, bc.Potential.Dirichlet(), where=electrodes)
    # the other cells of the face stay insulating

The condition *types* set the Poisson matrix, which is assembled once when the
simulation is created and shared by all batched environments. The Dirichlet
*values* only enter the right-hand side, so electrodes can act as actuators and
be driven every step, with different values per environment:

.. code-block:: python

    for step in range(n_steps):
        # [NZ, NX] for all environments, or [B, NZ, NX] per environment
        bc.set_potential_values(block, Face.Y_MINUS, action, where=electrodes)
        sim.single_step()

With ``differentiable=True``, gradients flow into Dirichlet values that require
grad: per-environment values get one gradient slice per environment, shared
values the sum over the environments. See :doc:`batching` for batched
environments in general.

Potential solver
----------------

Most of the runtime of an MHD run is spent in the potential solve. The main
options are:

- ``potential_use_preconditioner=True``: AMG-preconditioned CG (requires the
  ``amg`` extra). The hierarchy is built once and reused for all steps.
  ``potential_amg_options`` is passed to
  :func:`phipict.solvers.amg.build_amg_hierarchy`.
- ``potential_reuse_result=True`` (default): start from the previous
  :math:`\phi`.
- ``potential_solve_dtype=torch.float64``: solve in double precision while the
  rest of the simulation runs in single precision.
- ``potential_return_best_result``: if the solve does not converge, return the
  best iterate (default) or raise an error.
