.. _channelize_poly_func:

channelize_poly
===============

Polyphase channelizer with a configurable number of channels

.. versionadded:: 0.6.0

.. doxygenfunction:: matx::channelize_poly(const InType &in, const FilterType &f, index_t num_channels, index_t decimation_factor)

CUDA performance
~~~~~~~~~~~~~~~~

For some CUDA inputs and channelizer configurations, MatX uses a fused kernel
that performs the polyphase filtering and FFT in one launch. Other
configurations use the general backend, which performs the filtering and FFT
separately. Kernel selection is automatic.

Limitations
~~~~~~~~~~~

The general backend uses cuFFT, which supports half-precision (``matxFp16`` and
``matxBf16``) transforms only for power-of-two sizes. On CUDA, half-precision
outputs with any other channel count are supported only for critically sampled
channelizers (``decimation_factor == num_channels``) with 3, 5, or 6 channels,
which always use the fused kernel. Other such configurations raise a
``matxInvalidParameter`` error.

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
