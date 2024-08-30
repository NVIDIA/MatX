.. _inv_func:

inv
===

Matrix inverse
--------------

Compute the inverse of a square matrix.

.. note::
   This function is currently not supported with host-based executors (CPU)

.. doxygenfunction:: inv(const OpA &a)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Inverse.cu
   :language: cpp
   :start-after: example-begin inv-test-1
   :end-before: example-end inv-test-1
   :dedent:


