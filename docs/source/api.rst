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

Meshing
-------

.. autosummary::
   :toctree: api/generated/
   :nosignatures:

   phipict.meshing.Mesh
   phipict.meshing.MeshBlock
   phipict.meshing.BlockMesh
   phipict.meshing.Quad
   phipict.meshing.Hex
   phipict.meshing.Patch
   phipict.meshing.FacePatches
   phipict.meshing.read_blockmeshdict
   phipict.meshing.write_blockmeshdict
   phipict.meshing.read_vtk

.. autosummary::
   :toctree: api/generated/

   phipict.meshing.shapes
   phipict.meshing.grading
   phipict.meshing.curves
   phipict.meshing.boundary
   phipict.meshing.connect
   phipict.meshing.quality
   phipict.meshing.formats.blockmeshdict
   phipict.meshing.formats.vtk

Grids and I/O
-------------

.. autosummary::
   :toctree: api/generated/

   phipict.io.export.vtk
   phipict.io.export.series
   phipict.io.domain_io
   phipict.grid.shapes
   phipict.grid.helpers
   phipict.solvers.amg

Batching
--------

.. autosummary::
   :toctree: api/generated/

   phipict.batching
