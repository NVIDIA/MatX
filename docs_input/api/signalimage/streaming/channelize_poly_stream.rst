.. _make_channelize_poly_stream_func:

make_channelize_poly_stream
===========================

Streaming polyphase channelizer: channelize a signal delivered in segments,
equivalent to a one-shot :ref:`channelize_poly <channelize_poly_func>` over
the concatenated stream

.. versionadded:: 1.1.0

.. doxygenstruct:: matx::ChannelizePolyStreamParams
   :members:

.. doxygenfunction:: matx::make_channelize_poly_stream

.. doxygenclass:: matx::ChannelizePolyStream
   :members:

Examples
~~~~~~~~

.. literalinclude:: ../../../../examples/streaming.cu
   :language: cpp
   :start-after: example-begin channelize_poly_stream-1
   :end-before: example-end channelize_poly_stream-1
   :dedent:

.. literalinclude:: ../../../../examples/streaming.cu
   :language: cpp
   :start-after: example-begin channelize_poly_stream-2
   :end-before: example-end channelize_poly_stream-2
   :dedent:
