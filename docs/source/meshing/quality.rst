Mesh quality
============

.. currentmodule:: phipict.meshing

Check a mesh before running it: :meth:`Mesh.check` finds the problems that stop
a simulation (inverted cells, faces without a patch or connection, patches without
a condition) and summarises the cell quality that affects accuracy and solver
convergence. :meth:`Mesh.get_domain` runs the same checks and refuses a mesh with
errors.

.. code-block:: python

    import phipict.meshing as pm

    walls = pm.Patch("walls", pm.Wall())
    mesh = pm.Mesh([pm.make_quad([(0, 0), (2, 0), (0.5, 1), (2.5, 1)], cells=(16, 8),
                            patches=pm.FacePatches(y_minus=walls, y_plus=walls))])
    report = mesh.check()
    print(report)
    # 1 blocks, 128 cells
    # min volume 0.01562, min scaled Jacobian 0.894
    # max aspect ratio 1.12, max non-orthogonality 26.6 deg
    # ERROR: faces without patch or connection: quad:X_MINUS, quad:X_PLUS

    mesh.make_periodic(translation=(2.0, 0.0))
    assert mesh.check().ok

The report
----------

:class:`MeshReport` has the size of the mesh, four quality measures, a list of
``errors`` and a list of ``warnings``. ``report.ok`` is true without errors, and
``report.raise_if_invalid()`` raises a ``ValueError`` listing them.

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Measure
     - Meaning
   * - ``min_volume``
     - smallest cell volume (area in 2D); negative for inverted cells
   * - ``min_scaled_jacobian``
     - smallest normalised corner Jacobian, 1 for rectangular cells; at or below 0
       a cell is degenerate or inverted
   * - ``max_aspect_ratio``
     - longest over shortest cell edge; large values are normal in boundary layers
   * - ``max_non_orthogonality``
     - largest angle between a face normal and the line between the two cell
       centres, in degrees; the pressure solves converge more slowly as it grows,
       above about 70 degrees a warning is issued

A block whose cells are all inverted is left-handed: its axes are ordered the
wrong way round. Flip one axis (:meth:`MeshBlock.flip`).

Per-cell values
---------------

The measures are also available per cell, e.g. to find where a mesh is poor or to
plot them (:func:`phipict.io.export.write_vtk` writes volume and scaled Jacobian
of a mesh as cell data):

.. code-block:: python

    from phipict.meshing.quality import non_orthogonality, scaled_jacobian

    block = mesh.blocks[0]
    angles = non_orthogonality(block.coords)    # [ny, nx], degrees
    print(float(angles.max()))

API
---

.. autoclass:: phipict.meshing.MeshReport
   :members:
   :no-index:

.. automodule:: phipict.meshing.quality
   :members: cell_volumes, scaled_jacobian, aspect_ratio, non_orthogonality
   :no-index:
