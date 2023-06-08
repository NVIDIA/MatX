.. _fftfreq_func:

fftfreq
=========

Returns the bin centers in cycles/unit of the sampling frequency known by the user

.. doxygenfunction:: matx::fftfreq(index_t n, float d = 1.0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin fftfreq-gen-test-1
   :end-before: example-end fftfreq-gen-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/GeneratorTests.cu
   :language: cpp
   :start-after: example-begin fftfreq-gen-test-2
   :end-before: example-end fftfreq-gen-test-2
   :dedent:
