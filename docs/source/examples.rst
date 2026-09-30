Examples
========

The first scripts in ``examples/`` run small versions of the validation cases in
``runscripts/validation`` (same grids as the coarsest validation runs) and
compare the result with the analytical solution. Both need a CUDA GPU.

Hartmann flow (2D)
------------------

A channel flow with insulating walls at :math:`y = \pm 1` and a transverse field
along :math:`y`. A constant pressure gradient :math:`G` drives the flow, and the
steady profile is

.. math::

    u(y) = \frac{G}{N} \left(1 - \frac{\cosh(Ha\, y)}{\cosh Ha}\right).

.. code-block:: bash

    python examples/hartmann.py --ha 10

.. literalinclude:: ../../examples/hartmann.py
   :language: python
   :start-at: import argparse

Shercliff and Hunt flow (3D)
----------------------------

A square duct with walls at :math:`y = \pm 1` (Hartmann walls) and
:math:`z = \pm 1` (side walls), field along :math:`y`. With ``--cw 0`` all walls
are insulating (Shercliff flow). With ``--cw 0.1`` the Hartmann walls are thin
conducting walls (Hunt flow). The analytical reference is Hunt (1965).

.. code-block:: bash

    python examples/duct.py --cw 0      # Shercliff
    python examples/duct.py --cw 0.1    # Hunt

.. literalinclude:: ../../examples/duct.py
   :language: python
   :start-at: import argparse

Flow past a cylinder
--------------------

A cylinder in a channel, meshed with :class:`~phipict.meshing.BlockMesh`: an
O-grid of four blocks around the cylinder inside a 3 x 3 arrangement of channel
blocks. Only some cell counts are given, the others are inferred from shared
edges, and the twelve block connections are found automatically. The inflow is a
parabolic profile, and the solution is written as a VTK time series. ``--three-d``
extrudes the mesh, periodic in z. See :doc:`meshing`.

.. code-block:: bash

    python examples/cylinder.py --steps 200 --out out/cylinder

.. literalinclude:: ../../examples/cylinder.py
   :language: python
   :start-at: def make_mesh

OpenFOAM blockMeshDict
----------------------

Runs the channel of ``examples/meshes/flow_past_cylinder/blockMeshDict`` (3 x 3
blocks, read as 2D from its ``empty`` patches) after giving its patches flow
conditions.

.. code-block:: bash

    python examples/blockmesh_import.py --steps 100

.. literalinclude:: ../../examples/blockmesh_import.py
   :language: python
   :start-at: def main
