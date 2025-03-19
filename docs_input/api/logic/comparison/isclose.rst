.. _isclose_func:

isclose
=======

Determine the closeness of values across two operators using absolute and relative tolerances. The output
from isclose is an ``int`` value since it's commonly used for reductions and ``bool`` reductions using
atomics are not available in hardware.


.. doxygenfunction:: isclose

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/isclose_test.cu
   :language: cpp
   :start-after: example-begin isclose-test-1
   :end-before: example-end isclose-test-1
   :dedent:

