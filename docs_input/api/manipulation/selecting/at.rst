.. _at_func:

at
==

Selects a single value from an operator. Since `at` is a lazily-evaluated operator, it should be used
in situations where `operator()` cannot be used. For instance:

.. code-block:: cpp

    (a = b(5)).run();

The code above creates a race condition where `b(5)` is evaluated on the host before launch, but the value may
not be computed from a previous operation. Instead, the `at()` operator can be used to defer the load until 
the operation is launched:

.. code-block:: cpp

    (a = at(b, 5)).run();

.. doxygenfunction:: at(const Op &op, Is... indices)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/at_test.cu
   :language: cpp
   :start-after: example-begin at-test-1
   :end-before: example-end at-test-1
   :dedent:

