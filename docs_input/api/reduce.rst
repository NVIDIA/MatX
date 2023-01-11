Reductions
##########

The reductions API provides functions for reducing data from a higher rank to a lower rank. Statistical reductions
are listed in the :ref:`statistics` API guide section. Note that to avoid collissions with the C++ standard library, 
``min`` and ``max`` are called ``rmin`` and ``rmax`` in MatX.

.. doxygenfunction:: reduce(OutType dest, [[maybe_unused]] TensorIndexType idest, const InType &in, ReduceOp op, cudaStream_t stream = 0, bool init = true)
.. doxygenfunction:: reduce(OutType dest, const InType &in, ReduceOp op, cudaStream_t stream = 0, [[maybe_unused]] bool init = true)
.. doxygenfunction:: any(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: any(OutType dest, const InType &in, cudaStream_t stream = 0)
.. doxygenfunction:: all(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: all(OutType dest, const InType &in, cudaStream_t stream = 0)
.. doxygenfunction:: rmin(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: rmin(OutType dest, const InType &in, cudaStream_t stream = 0)
.. doxygenfunction:: rmax(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: rmax(OutType dest, const InType &in, cudaStream_t stream = 0)
.. doxygenfunction:: sum(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: sum(OutType dest, const InType &in, cudaStream_t stream = 0)
.. doxygenfunction:: argmin(OutType dest, TensorIndexType &idest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: argmin(OutType dest, TensorIndexType &idest, const InType &in, cudaStream_t stream = 0)
.. doxygenfunction:: argmax(OutType dest, const TensorIndexType &idest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: argmax(OutType dest, TensorIndexType &idest, const InType &in, cudaStream_t stream = 0)  
.. doxygenfunction:: trace
.. doxygenfunction:: find
.. doxygenfunction:: find_idx
.. doxygenfunction:: unique
