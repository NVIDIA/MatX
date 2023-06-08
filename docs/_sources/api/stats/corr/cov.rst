.. _cov_func:

cov
###

Compute a covariance matrix

.. doxygenfunction:: cov(TensorTypeC &c, const TensorTypeA &a, cudaStream_t stream = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/Cov.cu
   :language: cpp
   :start-after: example-begin cov-test-1
   :end-before: example-end cov-test-1
   :dedent:
