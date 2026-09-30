Basic Usage
===========

A simulation consists of a :class:`~phipict.Domain` with one or more blocks, and a
:class:`~phipict.Simulation` that advances it in time.

Domain and grid
---------------

A mesh is made with :mod:`phipict.meshing` (see :doc:`meshing`): blocks with a
number of cells and a grading per axis, and named patches with boundary
conditions on the faces that are not connected to other blocks:

.. code-block:: python

    import torch
    import phipict
    import phipict.meshing as pm

    dtype, device = torch.float64, torch.device("cuda")

    walls = pm.Patch("walls", pm.Wall())                  # no-slip
    channel = pm.box(
        (0.0, -1.0), (10.0, 1.0),
        cells=(100, 50),
        grading=(None, pm.Symmetric(10.0)),               # refined towards both walls
        patches=pm.FacePatches(y_minus=walls, y_plus=walls),
    )
    mesh = pm.Mesh([channel])
    mesh.make_periodic("x")
    # mesh = mesh.extrude((0.0, 2.0), cells=20)           # 3D, periodic in z

    domain = mesh.get_domain(viscosity=1.0 / 100.0, dtype=dtype, device=device)
    block = domain.getBlocks()[0]

Boundaries
----------

``get_domain`` sets up the solver blocks: coinciding block faces are connected,
periodic faces made periodic, and each patch gets its condition
(:class:`~phipict.meshing.Wall`, :class:`~phipict.meshing.FreeSlip`,
:class:`~phipict.meshing.Inflow`, :class:`~phipict.meshing.Outflow`). Underneath,
this is the block API of the extension, which can also be used directly; block
sides are addressed as ``"-x"``, ``"+x"``, ``"-y"``, ``"+y"``, ``"-z"``, ``"+z"``
and configured before calling ``PrepareSolve``:

.. code-block:: python

    block = domain.CreateBlock(vertexCoordinates=coords, name="Block")
    block.CloseBoundary("-y")   # no-slip wall
    block.CloseBoundary("+y")
    block.MakePeriodic("x")     # periodic in x
    # block.OpenBoundary("-x")  # free-slip wall or symmetry plane
    # block.ConnectBlock("+x", other_block, "-x", "-y")  # multi-block domains
    domain.PrepareSolve()

Simulation
----------

.. code-block:: python

    sim = phipict.Simulation(
        domain=domain,
        dt=1e-2,
        substeps="ADAPTIVE",  # sub-steps chosen from adaptive_CFL
        non_orthogonal=False,  # the grid above is orthogonal
        pressure_tol=phipict.SolverTolerance(rtol=1e-6, atol=1e-14),
    )
    sim.make_divergence_free()

    for step in range(1000):
        sim.single_step()

    u = block.velocity  # [1, dims, (nz,) ny, nx]

Callbacks (``prep_fn``)
~~~~~~~~~~~~~~~~~~~~~~~

``prep_fn`` maps hook names (e.g. ``"PRE"``, ``"PRE_VELOCITY_SETUP"``, ``"POST"``)
to functions ``fn(domain, time_step, **kwargs)``. A typical use is a body force,
e.g. a constant pressure gradient driving a periodic channel:

.. code-block:: python

    def apply_forcing(domain: phipict.Domain, **kwargs: object) -> None:
        source = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device)
        domain.getBlocks()[0].setVelocitySource(source)
        domain.UpdateDomainData()

    sim = phipict.Simulation(..., prep_fn={"PRE_VELOCITY_SETUP": apply_forcing})

Solver tolerances
~~~~~~~~~~~~~~~~~

``advection_tol``, ``pressure_tol`` (and ``potential_tol`` for MHD) take either a
float or a :class:`~phipict.SolverTolerance`:

- a **float** is an absolute tolerance on ``||r||_2 / sqrt(n)``;
- ``SolverTolerance(rtol, atol)`` stops at ``||r||_2 < rtol * ||b||_2``, with
  ``atol`` as a floor for a (near) zero right-hand side. Its meaning does not
  depend on the domain size, so a tolerance tuned on a small domain carries over
  to a large one.

``pressure_tol_intermediate`` sets a looser tolerance for all but the last
corrector step, and ``pressure_warm_start=True`` starts each pressure solve from
the previous result.

Other options
~~~~~~~~~~~~~

- Advection scheme: ``domain.setAdvectionScheme(phipict.AdvectionScheme.LINEAR_UPWIND)``
  (default ``CENTRAL``).
- Sub-grid-scale models: :func:`phipict.simulation.sgs.append_sgs_viscosity_prep_fn`
  adds Smagorinsky or WALE eddy viscosity.
- Batched environments: see :doc:`batching`.
- Gradients: ``differentiable=True`` uses the differentiable solver backend, so
  losses on the flow fields can be back-propagated with PyTorch.
- Saving/loading: :mod:`phipict.io.domain_io` (``save_domain`` / ``load_domain``).
- Logging: ``phipict.set_verbosity(...)``.
