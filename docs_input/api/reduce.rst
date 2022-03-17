Reductions
##########

The reductions API provides functions for reducing data from a higher rank to a lower rank. Statistical reductions
are listed in the :ref:`statistics` API guide section. Note that to avoid collissions with the C++ standard library, 
``min`` and ``max`` are called ``rmin`` and ``rmax`` in MatX.

.. doxygenfunction:: reduce(TensorType dest, [[maybe_unused]] TensorIndexType idest, InType in, ReduceOp op, cudaStream_t stream = 0, bool init = true)
.. doxygenfunction:: reduce(TensorType &dest, const InType &in, ReduceOp op, cudaStream_t stream = 0, [[maybe_unused]] bool init = true)
.. doxygenfunction:: any
.. doxygenfunction:: all
.. doxygenfunction:: rmin
.. doxygenfunction:: rmax
.. doxygenfunction:: sum  
.. doxygenfunction:: argmin
.. doxygenfunction:: argmax
.. doxygenfunction:: trace
.. doxygenfunction:: find
.. doxygenfunction:: find_idx
.. doxygenfunction:: unique
