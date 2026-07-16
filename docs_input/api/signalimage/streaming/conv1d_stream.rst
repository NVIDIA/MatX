.. _make_conv1d_stream_func:

make_conv1d_stream
==================

Streaming 1D convolution: filter a signal delivered in segments. The concatenated
output equals a one-shot :ref:`conv1d <conv1d_func>` over the whole stream when the
total signal length is at least the filter length. For shorter signals one-shot
conv1d swaps the operand roles, while the streaming object keeps the signal and
filter roles fixed and returns the input-aligned result. FULL mode is
role-symmetric and matches for any length. The convolution uses the direct
(time-domain) method, which limits the filter to 1024 taps.

.. versionadded:: 1.0.0

.. doxygenstruct:: matx::Conv1DStreamParams
   :members:

.. doxygenfunction:: matx::make_conv1d_stream

.. doxygenclass:: matx::Conv1DStream
   :members:

Examples
~~~~~~~~

.. literalinclude:: ../../../../examples/streaming.cu
   :language: cpp
   :start-after: example-begin conv1d_stream-1
   :end-before: example-end conv1d_stream-1
   :dedent:

.. literalinclude:: ../../../../examples/streaming.cu
   :language: cpp
   :start-after: example-begin conv1d_stream-2
   :end-before: example-end conv1d_stream-2
   :dedent:
