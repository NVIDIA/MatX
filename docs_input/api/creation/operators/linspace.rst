.. _linspace_func:

linspace
========

Return a range of linearly-spaced numbers using first and last value. The step size is
determined by the `count` parameter. `axis` (either 0 or 1) can be used to make the increasing
sequence along the specified axis.

.. doxygenfunction:: matx::linspace(T first, T last, index_t count, int axis = 0)
.. doxygenfunction:: matx::linspace(const T (&firsts)[NUM_RC], const T (&lasts)[NUM_RC], index_t count, int axis = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin linspace-gen-test-1
   :end-before: example-end linspace-gen-test-1
   :dedent:

