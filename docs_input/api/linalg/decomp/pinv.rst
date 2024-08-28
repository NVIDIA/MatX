.. _pinv_func:

pinv
####

Compute the Moore-Penrose pseudo-inverse of a matrix.

.. doxygenfunction:: pinv(const OpA &a, float rcond = get_default_rcond<typename OpA::value_type>())

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_solver/Pinv.cu
   :language: cpp
   :start-after: example-begin pinv-test-1
   :end-before: example-end pinv-test-1
   :dedent:


