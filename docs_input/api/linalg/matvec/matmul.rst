.. _matmul_func:

matmul
======

Matrix Multiply (GEMM)
----------------------

`matmul` performs a transformation for Generic Matrix Multiplies (GEMMs) for complex and real-valued tensors. Batching
is supported for any tensor with a rank higher than 2.


.. doxygenfunction:: matmul(const OpA &A, const OpB &B, float alpha = 1.0, float beta = 0.0)
.. doxygenfunction:: matmul(const OpA &A, const OpB &B, const int32_t (&axis)[2], float alpha = 1.0, float beta = 0.0)

For information on experimental sparse tensor support for Sparse-Matrix x Matrix (SpMM), please see :ref:`sparse_tensor_api`.

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


.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matmul-test-6
   :end-before: example-end matmul-test-6
   :dedent:
