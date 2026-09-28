Examples
========

The scripts in ``examples/`` run small versions of the validation cases in
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
