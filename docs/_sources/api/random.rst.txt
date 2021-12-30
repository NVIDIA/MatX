Random Number Generation
########################

MatX provides the capability to generate random numbers using the cuRAND library on tensor view objects using the APIs below.
It is advised to create as few ``randomGenerator_t`` objects as possible since they can consume a large amount of memory on the
device. If possible, create many ``randomTensorView_t`` views from a single ``randomGenerator_t``.

.. doxygenclass:: matx::randomGenerator_t
    :members:
.. doxygenclass:: matx::randomTensorView_t
    :members:
