.. _max_func:

max
===

Reduces the input by the maximum values across the specified axes or performs
an element-wise maximum on each element in the input operators.

.. doxygenfunction:: max(const InType &in, const int (&dims)[D])
.. doxygenfunction:: max(const InType &in)
.. doxygenfunction:: max(Op t, Op t2)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin max-test-1
   :end-before: example-end max-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin max-test-2
   :end-before: example-end max-test-2
   :dedent:

.. literalinclude:: ../../../../test/00_operators/operator_func_test.cu
   :language: cpp
   :start-after: example-begin max-el-test-1
   :end-before: example-end max-el-test-1
   :dedent:   
