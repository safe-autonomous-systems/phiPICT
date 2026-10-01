Migrating from phipict.grid
===========================

.. currentmodule:: phipict.meshing

Code that builds grids with :mod:`phipict.grid.shapes` and sets up blocks by hand
can move to :mod:`phipict.meshing` step by step. The shortest step keeps the
existing grid and only wraps it:

.. code-block:: python

    import torch
    import phipict.meshing as pm
    from phipict.grid import shapes

    grid = shapes.make_wall_refined_ortho_grid(
        64, 32, corner_lower=(0, -1), corner_upper=(4, 1),
        wall_refinement=["-y", "+y"], base=1.05,
    )
    walls = pm.Patch("walls", pm.Wall())
    block = pm.MeshBlock.from_coords(grid, patches=pm.FacePatches(y_minus=walls, y_plus=walls))
    mesh = pm.Mesh([block])
    mesh.make_periodic("x")

The replacements below reproduce the legacy grids: the test suite rebuilds the
fluidgym cylinder, airfoil and Rayleigh-Bénard meshes with them and compares the
vertices.

Functions
---------

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - :mod:`phipict.grid.shapes`
     - :mod:`phipict.meshing`
   * - ``make_weights_linear(n)``
     - ``Uniform()``
   * - ``make_weights_exp(n, base, "START"/"END"/"BOTH")``
     - ``Geometric(base, Cluster.START/END/BOTH)``
   * - ``make_weights("simple", n, r, "START")``
     - ``Simple(r)``
   * - ``make_weights("simple", n, r, "END")``
     - ``Simple(1 / r)``
   * - ``make_weights("simple", n, r, "BOTH")``
     - ``Symmetric(r)``
   * - ``make_weights("tanh", n, a, side)``
     - ``Tanh(a, side)``
   * - ``make_weights("chebyshev_identity", n, g, side)``
     - ``ChebyshevBlend(g, side)``
   * - ``make_weights_cos(n, side)``
     - ``Cosine(side)``
   * - ``generate_grid_vertices_2D(res, corners, borders, x_weights, y_weights)``
     - ``make_quad(corners, cells, grading, edges=QuadEdges(...))``; borders become
       :class:`Points` edges
   * - ``make_wall_refined_ortho_grid(nx, ny, lo, hi, walls, base)``
     - ``make_box(lo, hi, (nx, ny), grading=(..., Geometric(base, Cluster.BOTH)))``
   * - ``make_torus_2D(res, r1, r2, start, angle)``
     - ``make_annulus((0, 0), r1, r2, start, angle, cells=(res, None))``
   * - ``extrude_grid_z(grid, nz, z0, z1, weights)``
     - ``MeshBlock.extrude((z0, z1), nz, grading)`` or ``Mesh.extrude``
   * - ``rotate_grid(grid, angle)``
     - ``MeshBlock.rotate(angle, center=...)``
   * - ``extrapolate_boundary_layers(grid, [(face, scale)])``
     - ``MeshBlock.extend(face, length, cells, grading)``
   * - ``torch.cat`` of grids, dropping the shared row
     - ``MeshBlock.concat(other, axis)``
   * - ``torch.movedim``/``torch.flip`` to reorient a grid
     - ``MeshBlock.permute``/``MeshBlock.flip``, or not needed: connections work
       in any orientation
   * - :func:`phipict.grid.helpers.get_cell_centers`
     - ``MeshBlock.cell_centers()``

Two differences in meaning:

* The legacy ``x_weights`` of ``generate_grid_vertices_2D`` distribute the
  vertices *along y* (they are the weights of the x borders); here every grading
  belongs to the axis it is given for.
* The legacy ``refinement`` of ``make_weights_exp`` names where the exponent
  starts, so ``base < 1`` refines the other end. :class:`Geometric` keeps that
  behaviour for exact reproduction; for new code, :class:`Simple` and
  :class:`Symmetric` state the refinement directly.

Domain setup
------------

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Before
     - With :mod:`phipict.meshing`
   * - ``domain = phipict.Domain(...)``, ``domain.CreateBlock(vertexCoordinates=...)``
     - ``mesh.get_domain(viscosity, dtype=..., device=...)``
   * - ``block.ConnectBlock("+y", other, "-x", "-z", "+y")``
     - nothing: coinciding faces are connected
   * - ``block.MakePeriodic("x")``
     - ``mesh.make_periodic("x")``
   * - ``block.CloseBoundary("-y")``
     - a patch with :class:`Wall`
   * - ``block.OpenBoundary("-y")``
     - a patch with :class:`FreeSlip`
   * - ``CloseBoundary`` + ``setVelocity(profile)``
     - a patch with ``Inflow(profile)``
   * - ``CloseBoundary`` + ``makeVelocityVarying()`` + ``balance_boundary_fluxes``
     - a patch with :class:`Outflow`; the balance is done by ``get_domain``
   * - ``update_advective_boundaries`` in a ``PRE`` hook
     - ``mesh.outflow_hook(domain, velocity)``
   * - ``bc.set_bc(block, face, spec)`` after creation
     - ``Wall(extra=(spec,))`` on the patch

``examples/hartmann.py`` and ``examples/duct.py`` show the result for two simple
cases, ``tests/meshing/test_references.py`` for the fluidgym geometries.
