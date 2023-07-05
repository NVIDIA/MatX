.. _filter_func:

filter
======

`filter` provides an interface for executing IIR and FIR filters on tensors. It is primarily
used for IIR filters, but it will call the appropriate functions for FIR if the number of recursive coefficients is
0. 

.. note::
   This function is currently is not supported with host-based executors (CPU)

.. doxygenfunction:: matx::filter(OutType &o, const InType &i, const std::array<FilterType, NR> h_rec, const std::array<FilterType, NNR> h_nonrec, cudaStream_t stream = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../examples/recursive_filter.cu
   :language: cpp
   :start-after: example-begin filter-example-1
   :end-before: example-end filter-example-1
   :dedent:
