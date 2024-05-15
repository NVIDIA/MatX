.. _corr_func:

corr
####

Cross-correlation of two inputs

.. doxygenfunction:: corr(const In1Type &i1, const In2Type &i2, matxConvCorrMode_t mode, matxConvCorrMethod_t method)
.. doxygenfunction:: corr(const In1Type &i1, const In2Type &i2, const int32_t (&axis)[1], matxConvCorrMode_t mode, matxConvCorrMethod_t method)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ConvCorr.cu
   :language: cpp
   :start-after: example-begin corr-test-1
   :end-before: example-end corr-test-1
   :dedent:
