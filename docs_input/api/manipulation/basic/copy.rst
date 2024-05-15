.. _copy_func:

copy
####

Copy an operator into a tensor. This is equivalent to (a = b), but may be optimized further
since it cannot be chained with other expressions.

.. doxygenfunction:: copy(OutputTensor &out, const InputTensor &in, Executor exec)
.. doxygenfunction:: copy(OutputTensor &out, const InputTensor &in, cudaStream_t stream = 0)
.. doxygenfunction:: copy(const Tensor &in, Executor exec)
.. doxygenfunction:: copy(const Tensor &in, cudaStream_t stream = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../include/matx/transforms/fft/fft_common.h
   :language: cpp
   :start-after: example-begin copy-test-1
   :end-before: example-end copy-test-1
   :dedent:
