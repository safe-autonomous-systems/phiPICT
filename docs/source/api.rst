API Reference
=============

Simulation
----------

.. autosummary::
   :toctree: api/generated/
   :nosignatures:

   phipict.Simulation
   phipict.MHDSimulation
   phipict.SolverTolerance
   phipict.simulation.sgs.append_sgs_viscosity_prep_fn

Domain
------

Classes of the compiled extension ``phipict._C``.

.. autosummary::
   :toctree: api/generated/
   :nosignatures:

   phipict.Domain
   phipict.Block
   phipict.Boundary
   phipict.FixedBoundary
   phipict.AdvectionScheme

Grids and I/O
-------------

.. autosummary::
   :toctree: api/generated/

   phipict.grid.shapes
   phipict.grid.helpers
   phipict.solvers.amg
   phipict.io.domain_io

Batching
--------

.. autosummary::
   :toctree: api/generated/

   phipict.batching
