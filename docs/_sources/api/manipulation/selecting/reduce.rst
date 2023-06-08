.. _reduce_func:

reduce
======

Reduces the input  using a generic reduction operator and optionally store the indices of the reduction

.. doxygenfunction:: reduce(OutType dest, TensorIndexType idest, const InType &in, ReduceOp op, cudaStream_t stream = 0, bool init = true)

Examples
~~~~~~~~

.. literalinclude:: ../../../../include/matx/transforms/reduce.h
   :language: cpp
   :start-after: example-begin reduce-1
   :end-before: example-end reduce-1
   :dedent:

Examples
~~~~~~~~

.. doxygenfunction:: reduce(OutType dest, const InType &in, ReduceOp op, cudaStream_t stream = 0, bool init = true)

.. literalinclude:: ../../../../include/matx/transforms/reduce.h
   :language: cpp
   :start-after: example-begin reduce-2
   :end-before: example-end reduce-2
   :dedent:
