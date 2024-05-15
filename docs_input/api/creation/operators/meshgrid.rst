.. _meshgrid_func:

meshgrid
=========

Creates a matrix with values from two vectors

.. doxygenfunction:: matx::meshgrid(Ts&&... ts)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin meshgrid-gen-test-1
   :end-before: example-end meshgrid-gen-test-1
   :dedent:

