.. _statistics:

Statistics
##########
The API below provides methods for statistics functions. 

.. doxygenfunction:: cumsum
.. doxygenfunction:: hist
.. doxygenfunction:: mean(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: mean(OutType dest, const InType &in, cudaStream_t stream = 0)  
.. doxygenfunction:: median(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: median(OutType dest, const InType &in, cudaStream_t stream = 0)  
.. doxygenfunction:: var(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: var(OutType dest, const InType &in, cudaStream_t stream = 0)  
.. doxygenfunction:: stdd(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)
.. doxygenfunction:: stdd(OutType dest, const InType &in, cudaStream_t stream = 0)  
