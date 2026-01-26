.. _channelize_poly_func:

channelize_poly
===============

Polyphase channelizer with a configurable number of channels

.. versionadded:: 0.6.0

.. doxygenfunction:: matx::channelize_poly(const InType &in, const FilterType &f, index_t num_channels, index_t decimation_factor)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ChannelizePoly.cu
   :language: cpp
   :start-after: example-begin channelize_poly-test-1
   :end-before: example-end channelize_poly-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_transform/ChannelizePoly.cu
   :language: cpp
   :start-after: example-begin channelize_poly-test-2
   :end-before: example-end channelize_poly-test-2
   :dedent:

