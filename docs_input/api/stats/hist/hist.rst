.. _hist_func:

hist
====

Compute a histogram of input `a` with bounds specified by `upper` and `lower` and `num_levels` bins

.. note::
   This function is currently not supported with host-based executors (CPU)

.. doxygenfunction:: hist(const InputOperator &a, const typename InputOperator::value_type lower, const typename InputOperator::value_type upper, int num_levels)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_tensor/CUBTests.cu
   :language: cpp
   :start-after: example-begin hist-test-1
   :end-before: example-end hist-test-1
   :dedent:

