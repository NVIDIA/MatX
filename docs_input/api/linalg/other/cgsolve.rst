.. _cgsolve_func:

cgsolve
=======

Complex gradient solve on square matrix

.. doxygenfunction:: cgsolve(const AType &A, const BType &B, double tol=1e-6, int max_iters=4)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/Solve.cu
   :language: cpp
   :start-after: example-begin cgsolve-test-1
   :end-before: example-end cgsolve-test-1
   :dedent:

