.. _rand_func:

Random Number Generation
########################

MatX provides the capability to generate random numbers on the host and device using the ``random()`` and ``randomi()`` 
operators. both host and device generation is supported through the cuRAND library.
 
 
- ``random()`` only generates random distribution for *float* data types
- ``randomi()`` only generates random disitrubtions for *integral* data types
 
Please see the documentation for each function for a full list of supported types

.. doxygenfunction:: matx::random(ShapeType &&s, Distribution_t dist, uint64_t seed = 0,LowerType alpha = 1, LowerType beta = 0)
.. doxygenfunction:: matx::random(const index_t (&s)[RANK], Distribution_t dist, uint64_t seed = 0,LowerType alpha = 1, LowerType beta = 0)
.. doxygenfunction:: matx::randomi(ShapeType &&s, uint64_t seed = 0, LowerType min = 0, LowerType max = 100)
.. doxygenfunction:: matx::randomi(const index_t (&s)[RANK], uint64_t seed = 0, LowerType min = 0, LowerType max = 100)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_tensor/ViewTests.cu
   :language: cpp
   :start-after: example-begin random-test-1
   :end-before: example-end random-test-1
   :dedent:
   

.. literalinclude:: ../../../test/00_tensor/ViewTests.cu
   :language: cpp
   :start-after: example-begin randomi-test-1
   :end-before: example-end randomi-test-1
   :dedent:
