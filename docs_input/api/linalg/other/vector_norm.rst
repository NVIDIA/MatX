.. _vector_norm_func:

vector_norm
===========

Compute a norm of a vector. Currently L1 and L2 norms are supported.

The `order` parameter can be either `NormOrder::L1`` or `NormOrder::L2``

.. doxygenfunction:: vector_norm(const Op &op, NormOrder order = NormOrder::NONE)
.. doxygenfunction:: vector_norm(const Op &op, const int (&dims)[D], NormOrder order = NormOrder::NONE)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/Norm.cu
   :language: cpp
   :start-after: example-begin vector-norm-test-1
   :end-before: example-end vector-norm-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_transform/Norm.cu
   :language: cpp
   :start-after: example-begin vector-norm-test-2
   :end-before: example-end vector-norm-test-2
   :dedent:
