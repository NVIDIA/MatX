Limitations
###########

With the executor and operator APIs, MatX attempts to make a seamless compatibility layer between a simple tensor API and existing CUDA C/C++ libraries. Many of the limitations are due to the inflexibility in the CUDA libraries in allowing different data layouts, types, or other limitations.

Where possible, MatX tries to use the features of these libraries to accept advanced data layouts, but some libraries simply lack any support for this. The workaround for these issues are to copy/save the tensor in a different format before processing, but this
negatively impacts performance. MatX will eventually remove these as libraries add more compatibility. In addition to specific functions, MatX currently has limitations overall that may be relaxed in future releases. Those limitations are listed in the table below.

Note that for CPU/host execution only operator expressions are supported currently, and functions requiring a library call (FFT, GEMM, etc) are not supported at this time on the host.

.. list-table:: General MatX Limitations Matrix
  :widths: 25 75
  :header-rows: 1
  
  * - Description
    - Limitations
  * - Operators inside executors
    - Operators are currently not allowed as inputs to executors other than in algebraic statements
  * - Mixing operators and executor API
    - Adding executor API calls is not allowed in an operator statement. For example, with operator C, `exec(A, fft(B) * C)` is not allowed since fft is an executor API call and C is an operator
  * - Cache cleanup
    - Currently cache cleanup is possible by deleting all entries in a specific cache. Further improvements may include LRU or more direct control over the cache

The table below describes the current state of limitations in MatX.

.. list-table:: MatX Frontend API Limitations Matrix
  :widths: 10 10 10 10 10 10 30 10
  :header-rows: 1

  * - Function
    - Permuted Inputs Allowed
    - Strided Inputs Allowed
    - 64-bit Index Support
    - fp16 (__half) Support
    - bfloat16 (__nv_bfloat16) Support
    - Other Limitations
    - CUDA Version
  * - fft
    - Yes
    - Yes, but not with transposed views
    - Yes
    - No
    - No
    - N/A
    - 11.4
  * - gemm
    - Yes
    - No
    - CUTLASS 2.6 provider only. No with cuBLAS/cuBLASLt
    - Yes
    - Yes
    - N/A
    - 11.4
  * - Solver (svd, inv, chol, qr, lu)
    - No
    - No
    - No
    - No
    - No    
    - Requires column-major input
    - 11.4
  * - CUB (hist, sort, cumsum)
    - No
    - No
    - Yes
    - No
    - No    
    - Most functions require operations on row data in row-major order
    - 11.4
  * - Convolutions/Correlations
    - Yes
    - Yes
    - Yes
    - No
    - No    
    - N/A
    - 11.4
    
