Reductions
##########

The reductions API provides functions for reducing data from a higher rank to a lower rank. 

.. doxygenfunction:: reduce(TensorType dest, [[maybe_unused]] TensorIndexType idest, InType in, ReduceOp op, cudaStream_t stream = 0, bool init = true)
.. doxygenfunction:: reduce(TensorType &dest, const InType &in, ReduceOp op, cudaStream_t stream = 0, bool init = true)
.. doxygenfunction:: any
.. doxygenfunction:: all
.. doxygenfunction:: rmin
.. doxygenfunction:: rmax
.. doxygenfunction:: sum  
.. doxygenfunction:: mean
.. doxygenfunction:: median
.. doxygenfunction:: var 
.. doxygenfunction:: stdd
.. doxygenfunction:: argmin
.. doxygenfunction:: argmax
