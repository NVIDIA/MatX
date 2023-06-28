Polyphase Resampling
####################

The polyphase resampler API supports resampling an input signal by ratio ``up``/``down`` using a polyphase filter.

.. doxygenfunction:: matx::resample_poly(OutType &out, const InType &in, const FilterType &f, index_t up, index_t down, cudaStream_t stream = 0)
