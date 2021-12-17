Linear Solvers
##############

The linear solver interface provides methods for users to run a linear solver using either cuBLAS or
cuSolver as the backend.

Cached API
----------
.. doxygenfunction:: chol
.. doxygenfunction:: lu
.. doxygenfunction:: qr
.. doxygenfunction:: svd
.. doxygenfunction:: eig

Non-Cached API
--------------
.. doxygenclass:: matx::matxDnCholSolverPlan_t
    :members:
.. doxygenclass:: matx::matxDnLUSolverPlan_t
    :members:
.. doxygenclass:: matx::matxDnQRSolverPlan_t
    :members:    
.. doxygenclass:: matx::matxDnEigSolverPlan_t
    :members:    
