.. _fill_func:

fill
====

Generate an operator whose elements are all the same caller-supplied value.

``fill()`` has a shape-taking form (rank inferred from the shape) and a
shapeless form. The shapeless form is preferred in contexts where the shape
can be deduced by the surrounding expression, e.g. as a broadcastable scalar
inside an operator expression. ``fill<T>({}, value)`` produces a rank-0
operator whose single element is ``value``.

Unlike :ref:`ones <ones_func>` and :ref:`zeros <zeros_func>`, ``fill`` has no
default for ``T``: the value type is either deduced from ``value`` or
specified explicitly so that an int literal does not silently collapse a
floating-point fill to an integer operator.


.. doxygenfunction:: matx::fill(ShapeType &&s, T value)
.. doxygenfunction:: matx::fill(const index_t (&s)[RANK], T value)
.. doxygenfunction:: matx::fill(T value)

Examples
~~~~~~~~

Shape-taking form, fills a 1D tensor:

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin fill-gen-test-1
   :end-before: example-end fill-gen-test-1
   :dedent:

Shapeless form, broadcast as a scalar inside an operator expression:

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin fill-gen-test-2
   :end-before: example-end fill-gen-test-2
   :dedent:

Rank-0 fill via the ``{}`` shape:

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin fill-gen-test-3
   :end-before: example-end fill-gen-test-3
   :dedent:
