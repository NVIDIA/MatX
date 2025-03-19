.. _min_func:

min
===

Reduces the input by the minimum values across the specified axes or performs
an element-wise minimum on each element in the input operators.

.. doxygenfunction:: min(const InType &in, const int (&dims)[D])
.. doxygenfunction:: min(const InType &in)
.. doxygenfunction:: min(Op t, Op t2)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin min-test-1
   :end-before: example-end min-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin min-test-2
   :end-before: example-end min-test-2
   :dedent:

.. literalinclude:: ../../../../test/00_operators/operator_func_test.cu
   :language: cpp
   :start-after: example-begin min-el-test-1
   :end-before: example-end min-el-test-1
   :dedent:      
