.. _repmat_func:

repmat
======

Repeat an operator

.. doxygenfunction:: repmat(const T1 &t, index_t reps)    
.. doxygenfunction:: repmat(const T1 &t, const index_t(&reps)[N])
.. doxygenfunction:: repmat(const T1 &t, const index_t *reps)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/repmat_test.cu
   :language: cpp
   :start-after: example-begin repmat-test-1
   :end-before: example-end repmat-test-1
   :dedent:

