.. _cov_func:

cov
###

Compute a covariance matrix

.. note::
   This function is currently not supported with host-based executors (CPU)


.. doxygenfunction:: cov(const AType &a)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/Cov.cu
   :language: cpp
   :start-after: example-begin cov-test-1
   :end-before: example-end cov-test-1
   :dedent:
