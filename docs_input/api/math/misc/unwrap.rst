.. _unwrap_func:

unwrap
======

Unwrap phase-like values by replacing jumps larger than a discontinuity with their period-complementary values.
This function is not optimized for parallel performance.

This matches NumPy's ``unwrap`` characteristics:

- Operates along a selected axis (default: last axis)
- Uses a configurable ``period`` (default: ``2*pi``)
- Treats ``discont < period/2`` as ``period/2``

.. doxygenfunction:: unwrap

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/unwrap_test.cu
   :language: cpp
   :start-after: example-begin unwrap-test-1
   :end-before: example-end unwrap-test-1
   :dedent:

