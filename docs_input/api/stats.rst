Statistics
##########
The API below provides methods for statistics functions. Note that some non-cached plans use a generic handle for
CUB, and just pass in the appropriate operation.

Cached API
----------
.. doxygenfunction:: cumsum
.. doxygenfunction:: hist

Non-Cached API
--------------
.. doxygenclass:: matx::matxCubPlan_t
    :members:
.. doxygenenum:: matx::CUBOperation_t    
