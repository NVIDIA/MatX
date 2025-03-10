.. _matvec_func:

matvec
######

Matrix-vector multiplication

.. doxygenfunction:: matvec

For information on experimental sparse tensor support for Sparse-Matrix x Vector (SpMV), please see :ref:`sparse_tensor_api`.

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/MatMul.cu
   :language: cpp
   :start-after: example-begin matvec-test-1
   :end-before: example-end matvec-test-1
   :dedent:
