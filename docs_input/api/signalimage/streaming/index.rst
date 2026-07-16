.. _streaming:

Streaming
#########

Streaming objects filter arbitrarily long signals delivered in segments. An
object is constructed once via its ``make_*_stream()`` factory, then fed
segments of any (possibly varying) size with ``feed()``. A final ``flush()`` is called
at the end of the stream to emit any trailing outputs. ``flush()`` ends the stream:
further ``flush()`` calls return an empty slice, and ``feed()`` throws until
``reset()`` starts a new stream. The concatenation of the
produced slices equals a single one-shot call of the corresponding transform
over the whole signal, with a few caveats as documented for each streaming object.
Each object owns only a small history buffer that scales with the filter and some
object parameters (e.g., the downsampling factor for ``resample_poly``). No allocation
scales directly with the segment size.
All work runs asynchronously on the executor bound at construction.

The canonical pattern sizes one reusable output buffer with
``max_output(largest segment size)`` and passes it to every ``feed()`` and
``flush()`` call. Each call returns a slice of that buffer containing the
outputs it produced. Note that this means that the returned output slice aliases the
storage of the output buffer. This pattern prevents any dynamic memory allocation
for the outputs during the streaming operation.
See ``examples/streaming.cu`` for complete programs.

.. toctree::
   :maxdepth: 1
   :glob:

   *
