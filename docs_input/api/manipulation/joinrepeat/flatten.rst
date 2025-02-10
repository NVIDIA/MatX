.. _flatten_func:

flatten
=======

Flatten an operator into a 1D operator where the total length is the product of all sizes
of the original operator

.. doxygenfunction:: flatten

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/flatten_test.cu
   :language: cpp
   :start-after: example-begin flatten-test-1
   :end-before: example-end flatten-test-1
   :dedent:

