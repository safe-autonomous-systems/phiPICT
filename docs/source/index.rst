phiPICT Documentation
=====================

phiPICT is a differentiable, GPU-accelerated multi-block PISO solver for PyTorch,
based on `PICT <https://github.com/tum-pbs/PICT>`_. On top of the incompressible
Navier-Stokes solver it adds inductionless magnetohydrodynamics (MHD) with
insulating and thin conducting walls, relative solver tolerances and an
AMG-preconditioned potential solve.

.. toctree::
   :maxdepth: 2

   installation
   basic_usage
   meshing
   mhd
   batching
   examples
   api
