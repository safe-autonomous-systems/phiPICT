Basic Usage
===========

A simulation consists of a :class:`~phipict.Domain` with one or more blocks, and a
:class:`~phipict.Simulation` that advances it in time.

Domain and grid
---------------

Blocks are defined by their vertex coordinates, a tensor of shape
``[1, dims, (nz+1,) ny+1, nx+1]``. :mod:`phipict.grid.shapes` has helpers to
build graded grids:

.. code-block:: python

    import torch
    import phipict
    from phipict.grid import shapes

    dtype, device = torch.float64, torch.device("cuda")
    nx, ny = 100, 50

    # Refine towards both walls; "simple" is OpenFOAM's simpleGrading
    # (other options: "tanh", "chebyshev_identity")
    y_weights = shapes.make_weights("simple", res=ny, grading=10, refinement="BOTH")
    grid = shapes.generate_grid_vertices_2D(
        [ny + 1, nx + 1],
        [(0.0, -1.0), (10.0, -1.0), (0.0, 1.0), (10.0, 1.0)],  # corners
        x_weights=y_weights,
        dtype=dtype,
    ).to(device)
    # shapes.extrude_grid_z(grid, res_z=..., weights_z=...) makes it 3D

    viscosity = torch.tensor([1.0 / 100.0], dtype=dtype)  # 1 / Re
    domain = phipict.Domain(2, viscosity, name="Channel", device=device, dtype=dtype)
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")

Boundaries
----------

Block sides are addressed as ``"-x"``, ``"+x"``, ``"-y"``, ``"+y"``, ``"-z"``,
``"+z"`` and are configured before calling ``PrepareSolve``:

.. code-block:: python

    block.CloseBoundary("-y")   # no-slip wall
    block.CloseBoundary("+y")
    block.MakePeriodic("x")     # periodic in x
    # block.OpenBoundary("-x")  # open boundary (e.g. symmetry plane)
    # block.ConnectBlock("+x", other_block, "-x")  # multi-block domains
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
