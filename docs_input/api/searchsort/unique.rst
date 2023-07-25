.. _unique_func:

unique
======

Reduce to unique values in input. On completion `a_out` contains all unique values in `a`, and `num_found`
contains the number of unique elements.

.. doxygenfunction:: unique(const OpA &a)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin unique-test-1
   :end-before: example-end unique-test-1
   :dedent:

