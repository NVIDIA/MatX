.. _inv_func:

inv
===

Matrix inverse
--------------

Perform a matrix inverse on a square matrix using LU decomposition. The inverse API is currently using cuBLAS as a backend and uses
getri/getrf functions for LU decomposition.

.. note::
   This function is currently is not supported with host-based executors (CPU)

.. doxygenfunction:: inv(const OpA &a)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Inverse.cu
   :language: cpp
   :start-after: example-begin inv-test-1
   :end-before: example-end inv-test-1
   :dedent:


