.. _make_resample_poly_stream_func:

make_resample_poly_stream
=========================

Streaming polyphase resampler: resample a signal delivered in segments,
equivalent to a one-shot :ref:`resample_poly <resample_poly_func>` over the
concatenated stream

.. versionadded:: 1.1.0

.. doxygenstruct:: matx::ResamplePolyStreamParams
   :members:

.. doxygenfunction:: matx::make_resample_poly_stream

.. doxygenclass:: matx::ResamplePolyStream
   :members:

Examples
~~~~~~~~

.. literalinclude:: ../../../../examples/streaming.cu
   :language: cpp
   :start-after: example-begin resample_poly_stream-1
   :end-before: example-end resample_poly_stream-1
   :dedent:

.. literalinclude:: ../../../../examples/streaming.cu
   :language: cpp
   :start-after: example-begin resample_poly_stream-2
   :end-before: example-end resample_poly_stream-2
   :dedent:
