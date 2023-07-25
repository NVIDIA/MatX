.. _diag_func:

diag
====

`diag` comes in two forms: a generator and an operator. The generator version is used to generate a diagonal
tensor with a given value, while the operator pulls diagonal elements from a tensor.

Operator
________
.. doxygenfunction:: matx::diag(T1 t)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-op-test-1
   :end-before: example-end diag-op-test-1
   :dedent:

Generator
_________

.. doxygenfunction:: matx::diag(const index_t (&s)[RANK], T val)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-gen-test-1
   :end-before: example-end diag-gen-test-1
   :dedent:
