.. _concat_func:

concat
======

Concatenate operators along a dimension. The returned operator will index into each of the 
concatenated operators, depending on where it's indexed

.. doxygenfunction:: concat

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/concat_test.cu
   :language: cpp
   :start-after: example-begin concat-test-1
   :end-before: example-end concat-test-1
   :dedent:

