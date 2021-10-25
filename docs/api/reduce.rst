Reductions
##########

The reductions API provides functions for reducing data from a higher rank to a lower rank. 

.. doxygenfunction:: reduce
.. doxygenfunction:: any
.. doxygenfunction:: all
.. doxygenfunction:: rmin
.. doxygenfunction:: rmax
.. doxygenfunction:: sum  
.. doxygenfunction:: mean
.. doxygenfunction:: median
.. doxygenfunction:: var 
.. doxygenfunction:: stdd
.. doxygenclass:: matx::reduceOpMin
.. doxygenclass:: matx::reduceOpMax
.. doxygenclass:: matx::reduceOpSum
.. doxygenfunction:: atomicAdd(cuda::std::complex<float> *addr, cuda::std::complex<float> val)
.. doxygenfunction:: atomicAdd(cuda::std::complex<double> *addr, cuda::std::complex<double> val)
.. doxygenfunction:: atomicMin(float *addr, float val)
.. doxygenfunction:: atomicMin(double *addr, double val)
.. doxygenfunction:: atomicMax(float *addr, float val)
.. doxygenfunction:: atomicMax(double *addr, double val)
.. doxygenfunction:: __shfl_down_sync(unsigned mask, cuda::std::complex<float> var, unsigned int delta)
.. doxygenfunction:: __shfl_down_sync(unsigned mask, cuda::std::complex<double> var, unsigned int delta)
