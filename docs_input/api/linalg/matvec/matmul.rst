.. _matmul_func:

matmul
======

Matrix Multiply (GEMM)
----------------------

`matmul` performs a transformation for Generic Matrix Multiplies (GEMMs) for complex and real-valued tensors. Batching
is supported for any tensor with a rank higher than 2.

.. note::
   This function is currently is not supported with host-based executors (CPU)


.. doxygenfunction:: matmul(TensorTypeC C, const TensorTypeA A, const TensorTypeB B, cudaStream_t stream = 0, float alpha = 1.0, float beta = 0.0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matmul-test-1
   :end-before: example-end matmul-test-1
   :dedent:

Permuted A

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matmul-test-2
   :end-before: example-end matmul-test-2
   :dedent:

Permuted B

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matmul-test-3
   :end-before: example-end matmul-test-3
   :dedent:

Batched

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matmul-test-4
   :end-before: example-end matmul-test-4
   :dedent:

Strided Batched

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matmul-test-5
   :end-before: example-end matmul-test-5
   :dedent:


.. doxygenfunction:: matmul(TensorTypeC C, const TensorTypeA A, const TensorTypeB B, const int32_t (&axis)[2], cudaStream_t stream = 0, float alpha = 1.0, float beta = 0.0)

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matmul-test-6
   :end-before: example-end matmul-test-6
   :dedent:
