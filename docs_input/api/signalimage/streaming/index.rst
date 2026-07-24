.. _streaming:

Streaming
#########

Streaming objects process arbitrarily long signals delivered in segments. An
object is constructed once via its ``make_*_stream()`` factory, then fed
segments of any (possibly varying) size with ``feed()``. A final ``flush()`` is called
at the end of the stream to emit any trailing outputs. ``flush()`` ends the stream:
further ``flush()`` calls return 0, and ``feed()`` throws until
``reset()`` starts a new stream. The concatenation of the
produced outputs equals a single one-shot call of the corresponding transform
over the whole signal, with a few caveats as documented for each streaming object.
Each object owns only a small history buffer that scales with the filter and some
object parameters (e.g., the downsampling factor for ``resample_poly``). No object-owned
allocation scales directly with the segment size.
All work runs asynchronously on the executor bound at construction.

The canonical pattern sizes one reusable output buffer with
``max_output(largest_input_segment_size)`` and passes it to every ``feed()`` and
``flush()`` call. Each call writes its outputs to the front of that buffer and
returns the number written (which may be 0). Consume the produced region
``slice(out, {0}, {count})`` before reusing the buffer. Note that zero-sized slices
are not valid, so check ``count > 0`` before creating and using the slice.
This pattern prevents any dynamic memory allocation for the outputs during the streaming operation.
See ``examples/streaming.cu`` for complete programs.

.. toctree::
   :maxdepth: 1
   :glob:

   *
