Sorting
#######
The API below provides methods for sorting tensor data using CUB radix sort as a backend.

.. doxygenfunction:: sort(OutputTensor &a_out, const InputOperator &a, const SortDirection_t dir, cudaExecutor exec = 0)
.. doxygenfunction:: sort(OutputTensor &a_out, const InputOperator &a, const SortDirection_t dir, [[maybe_unused]] SingleThreadHostExecutor exec)  

