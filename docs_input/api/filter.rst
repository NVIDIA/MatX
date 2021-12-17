Filtering (IIR and FIR)
#######################

The filter transformation API provides functions for executing IIR and FIR filters on tensor_t objects.

Cached API
----------
.. doxygenfunction:: matx::filter(OutType &o, const InType &i, const std::array<FilterType, NR> h_rec, const std::array<FilterType, NNR> h_nonrec, cudaStream_t stream = 0)

Non-Cached API
--------------
.. doxygenfunction:: matx::matxMakeFilter(OutType &o, const InType &i, tensor_t<FilterType, 1> &h_rec, tensor_t<FilterType, 1> &h_nonrec)
.. doxygenfunction:: matx::matxMakeFilter(OutType &o, const InType &i, const std::array<FilterType, NR> &h_rec, const std::array<FilterType, NNR> &h_nonrec)
