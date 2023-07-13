.. _allclose_func:

allclose
========

Reduce the closeness of two operators to a single scalar (0D) output. The output
from allclose is an ``int`` value since boolean reductions are not available in hardware


.. doxygenfunction:: allclose

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin allclose-test-1
   :end-before: example-end allclose-test-1
   :dedent:

