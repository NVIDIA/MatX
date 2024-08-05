.. _matrix_norm_func:

matrix_norm
===========

Compute a norm of a matrix. Currently Frobenius or L1 norms are supported

The `order` parameter can be either `NormOrder::L1`` or `NormOrder::FROB`. `NormOrder::NONE` 
may also be used as an alias for the Frobenius norm.

.. doxygenfunction:: matrix_norm(const Op &op, NormOrder order = NormOrder::NONE)
.. doxygenfunction:: matrix_norm(const Op &op, const int (&dims)[D], NormOrder order = NormOrder::NONE)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/Norm.cu
   :language: cpp
   :start-after: example-begin matrix-norm-test-1
   :end-before: example-end matrix-norm-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_transform/Norm.cu
   :language: cpp
   :start-after: example-begin matrix-norm-test-2
   :end-before: example-end matrix-norm-test-2
   :dedent:   

