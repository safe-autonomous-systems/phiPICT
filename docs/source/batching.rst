Batched Simulations
===================

A batched domain holds ``B`` environments: copies of one setup that share the
grid, the block and boundary layout and the sparsity patterns, but each have
their own state. One simulation advances all of them together, so every kernel
launch and every solver iteration covers the whole batch. This pays off for small
environments, whose runtime is dominated by host overhead (e.g. many
reinforcement learning environments): a small 2D MHD duct runs at about 213
environment steps/s unbatched and about 1900 at ``B = 64``–``128``.

Batched results match those of separate simulations.

Creating a batched domain
-------------------------

Set up the domain as usual, then call ``setBatchSize`` and ``PrepareSolve`` again
before creating the simulation:

.. code-block:: python

    domain = make_domain()      # blocks and boundaries, PrepareSolve() done
    domain.setBatchSize(8)
    domain.PrepareSolve()

    block = domain.getBlocks()[0]
    block.velocity.shape        # [8, dims, (nz,) ny, nx]
    for b in range(8):
        block.velocity[b].copy_(initial_velocity(b))
    domain.UpdateDomainData()

    sim = phipict.Simulation(domain=domain, dt=1e-2, substeps="ADAPTIVE")
    sim.single_step()           # advances all 8 environments

``setBatchSize`` repeats the state fields of all blocks (velocity, pressure,
passive scalar and, for MHD, the electric potential) from batch size 1 to ``B``.
Static data stays shared until you give it per environment.

Alternatively, :func:`phipict.batching.stack_domains` combines single-environment
domains of the same setup, e.g. with different initial states or loaded from
files:

.. code-block:: python

    from phipict.batching import stack_domains

    domains = [make_domain() for _ in range(B)]
    ...  # set the state of each
    domain = stack_domains(domains)  # domains[0], now batched and prepared

The first domain becomes the batched domain in place; grid, boundaries and solver
settings are taken from it. :func:`phipict.batching.copy_env_state` writes a
single-environment domain into one environment of an existing batch, e.g. to
reset that environment:

.. code-block:: python

    from phipict.batching import copy_env_state

    copy_env_state(domain, env_idx=3, src=reset_domain)

Shared and per-environment data
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Data
     - Batching
   * - Grid, block and boundary layout, sparsity patterns
     - Always shared.
   * - Block fields (velocity, pressure, passive scalar, potential)
     - Always per environment, leading dimension ``B``.
   * - Velocity sources
     - Shared (``[1, dims]``) or per environment (``[B, dims]``).
   * - Fixed boundary data (velocity, passive scalar)
     - Shared (leading dimension 1) or per environment (leading dimension ``B``).
   * - Potential condition types (MHD)
     - Always shared: they set the potential matrix and its AMG hierarchy.
   * - Dirichlet potential values (MHD)
     - Shared or per environment, see :doc:`mhd`.

To give the boundary data different values per environment, set a tensor with
leading dimension ``B``. For example, a scaled inflow per environment:

.. code-block:: python

    inflow_bound = block.getBoundary("-x")
    inflow = inflow_bound.velocity.expand(B, *inflow_bound.velocity.shape[1:])
    scales = torch.tensor([0.5, 1.0, 2.0], dtype=dtype, device=device)
    inflow_bound.setVelocity((inflow * scales.view(B, 1, 1, 1)).contiguous())
    domain.UpdateDomainData()

Flux balancing of outflow boundaries and advective outflows work per
environment. :func:`~phipict.batching.copy_env_state` makes shared boundary data
per environment when the copied environment differs from the rest.

Solvers
-------

Batched linear solves iterate until every environment has converged. Relative
tolerances (:class:`~phipict.SolverTolerance`) are resolved per environment, so
each environment is solved to the same accuracy as in a separate run, even if
their right-hand sides differ by orders of magnitude. The potential matrix and
its AMG hierarchy are shared by all environments; the pressure AMG shares its
interpolation and refreshes the coarse operators of all environments together.

Adaptive substeps
-----------------

With ``substeps="ADAPTIVE"``, every environment splits the time step by its own
maximum velocity (``adaptive_CFL_per_env=True``, the default), so it takes the
same substeps as it would on its own:

- The batch runs as many substeps as the environment that needs the most. An
  environment that has already covered the time step runs the extra substeps
  too, and its state is restored afterwards
  (:class:`~phipict.batching.EnvStateSnapshot`). A batched substep costs about
  as much as with shared substeps.
- Hooks receive the substep sizes as a ``[B]`` tensor ``time_step``. A hook that
  uses ``time_step`` must broadcast it over the environments (as
  ``update_advective_boundaries`` does).
- ``sim.total_time`` advances by ``dt`` per step.
  ``Domain.getMaxVelocityPerEnv`` returns the ``[B]`` maximum velocities used for
  the CFL condition.

With ``adaptive_CFL_per_env=False`` all environments take the substeps of the
fastest one. The differentiable path always uses shared substeps.

Per-environment state outside the domain that a custom
:class:`~phipict.Simulation` subclass carries across substeps must be restored
too; override ``_env_state_extras`` to return getters of flat tensors with one
slice per environment (see :class:`~phipict.MHDSimulation`).

Gradients
---------

Batching works with ``differentiable=True``. Each environment gets its own
gradients w.r.t. its per-environment inputs (initial fields, per-environment
sources and boundary data, Dirichlet potential values), and parameters shared by
all environments (e.g. a shared velocity source or the Stuart number) get the
sum over the environments:

.. code-block:: python

    v0 = initial_velocities.requires_grad_(True)   # [B, dims, (nz,) ny, nx]
    block.setVelocity(v0.clone())
    domain.UpdateDomainData()
    sim = phipict.Simulation(domain=domain, dt=1e-2, differentiable=True)
    for _ in range(n_steps):
        sim.single_step()
    loss = (weights.view(B, 1, 1, 1) * block.velocity**2).sum()
    (grad,) = torch.autograd.grad(loss, v0)        # one gradient per environment
    domain.Detach()
