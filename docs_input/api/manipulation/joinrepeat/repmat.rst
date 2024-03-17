.. _repmat_func:

repmat
======

Repeat an operator

.. doxygenfunction:: repmat(T1 t, index_t reps)    
.. doxygenfunction:: repmat(T1 t, const index_t(&reps)[N])
.. doxygenfunction:: repmat(T1 t, const index_t *reps)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin repmat-test-1
   :end-before: example-end repmat-test-1
   :dedent:

