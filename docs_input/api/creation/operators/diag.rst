.. _diag_func:

diag
====

`diag` comes in two forms: a generator and an operator. The generator version is used to generate a diagonal
tensor with a given value or a 1D input operator, while the operator pulls diagonal elements from a tensor.

Operator
________
.. doxygenfunction:: diag(const T1 &t, index_t k = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-op-test-1
   :end-before: example-end diag-op-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-op-test-2
   :end-before: example-end diag-op-test-2
   :dedent:

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-op-test-3
   :end-before: example-end diag-op-test-3
   :dedent:      

Generator
_________

.. doxygenfunction:: matx::diag(const index_t (&s)[RANK], T val)
.. doxygenfunction:: matx::diag(T val)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin diag-gen-test-1
   :end-before: example-end diag-gen-test-1
   :dedent:
