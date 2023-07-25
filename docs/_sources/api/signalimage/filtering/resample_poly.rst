.. _resample_poly_func:

resample_poly
=============

Polyphase resampler with a configurable up and downsample rate

.. doxygenfunction:: matx::resample_poly(const InType &in, const FilterType &f, index_t up, index_t down)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/ResamplePoly.cu
   :language: cpp
   :start-after: example-begin resample_poly-test-1
   :end-before: example-end resample_poly-test-1
   :dedent:

