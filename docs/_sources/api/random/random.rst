.. _rand_func:

Random Number Generation
########################

MatX provides the capability to generate random numbers and treat them as tensors that generate a new random
number each time they're accessed. On the device the generation is done using the cuRAND library.
It is advised to create as few ``randomGenerator_t`` objects as possible since they can consume a large amount of memory on the
device. If possible, create many ``randomTensorView_t`` views from a single ``randomGenerator_t``.

.. doxygenclass:: matx::randomGenerator_t
    :members:
.. doxygenclass:: matx::randomTensorView_t
    :members:

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_tensor/ViewTests.cu
   :language: cpp
   :start-after: example-begin random-test-1
   :end-before: example-end random-test-1
   :dedent:
