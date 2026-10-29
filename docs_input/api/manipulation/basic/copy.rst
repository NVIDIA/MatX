.. _copy_func:

copy
####

Copy an operator into a tensor. This is equivalent to (a = b), but may be optimized further
since it cannot be chained with other expressions.

With a CUDA executor, copies between contiguous, dense tensor views of the same
element type use ``cudaMemcpyAsync`` on the executor's stream. This also applies
to the CUDA stream overloads and the overloads that return a new tensor. No
additional build options or dependencies are required. Synchronize the executor
or stream before accessing the copied data from the host.

Sparse sources, noncontiguous views, copies that convert element types, and expression inputs
use executor-dispatched assignment. Host executors continue to use assignment.
Source and destination must have matching ranks and sizes, and their memory
regions must not overlap.
With ``MATX_EN_UNSAFE_ALIAS_DETECTION`` enabled, the CUDA byte-copy path uses
the same alias check as assignment and throws on detected unsafe overlap before
issuing the copy.

.. versionadded:: 0.3.0

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
