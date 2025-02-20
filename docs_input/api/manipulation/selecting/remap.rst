.. _remap_func:

remap
=====

Remaps an input operator by selecting items from an input index operator

.. doxygenfunction:: remap(const Op &t, Ind idx)
.. doxygenfunction:: remap(const Op &t, Ind idx, Inds... inds)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/remap_test.cu
   :language: cpp
   :start-after: example-begin remap-test-1
   :end-before: example-end remap-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/remap_test.cu
   :language: cpp
   :start-after: example-begin remap-test-2
   :end-before: example-end remap-test-2
   :dedent:

