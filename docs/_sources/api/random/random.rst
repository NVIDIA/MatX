.. _rand_func:

Random Number Generation
########################

MatX provides the capability to generate random numbers on the host and device using the ``random()`` operator. ``random()`` 
uses cuRAND on the device to generate random numbers from device code. 

.. note::
    randomGenerator_t has been deprecated after release 0.5.0. Please use the ``random()`` operator instead

.. doxygenfunction:: matx::random(ShapeType &&s, Distribution_t dist, uint64_t seed = 0, T alpha = 1, T beta = 0)
.. doxygenfunction:: matx::random(const index_t (&s)[RANK], Distribution_t dist, uint64_t seed = 0, T alpha = 1, T beta = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_tensor/ViewTests.cu
   :language: cpp
   :start-after: example-begin random-test-1
   :end-before: example-end random-test-1
   :dedent:
