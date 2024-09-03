.. _executor_compatibility:

Executor Compatibility
#######################

MatX's executor design allows for computations to run on different targets while leaving the code unchanged.
This document outlines the compatibility of various functions with these executors, categorized into two types:

1. **Element-wise operations**: These operations can be executed on any executor.
2. **Transforms**: These invoke library calls (e.g., CUDA libraries or CPU libraries on the host) or use custom kernels.

Note that there can be small differences in results between the Host executor and CUDA executor due to the way floating-point
arithmetic is performed. Also, on the host, most functions with the exception of reductions support multithreading.

The following table outlines the compatibility of different transforms with the different executors.

.. list-table:: Transform Executor Compatibility Matrix
  :widths: 10 10 10 10 20
  :header-rows: 1
  :class: table-alternating-row-colors
  
  * - Transform
    - Half Precision
    - Host
    - GPU
    - Notes
  * - fft
    - GPU only
    - Yes
    - Yes
    - 
  * - matmul
    - Yes
    - Yes
    - Yes
    -
  * - outer
    - Yes
    - Yes
    - Yes
    - 
  * - matvec
    - Yes
    - Yes
    - Yes
    - 
  * - chol
    - No
    - Yes
    - Yes
    - 
  * - lu
    - No
    - Yes
    - Yes
    - L & U are returned in the lower and upper half of the output respectively
  * - qr
    - No
    - Yes
    - Yes
    - Returns householder vectors and scalar factors on host
  * - eig
    - No
    - Yes
    - Yes
    - Hermitian/symmetric inputs only
  * - svd
    - No
    - Yes
    - Yes
    - Different methods on GPU for smaller matrices
  * - inv
    - No
    - No
    - Yes
    - 
  * - pinv
    - No
    - Yes
    - Yes
    - 
  * - det
    - No
    - Yes
    - Yes
    - 
  * - trace
    - No
    - Yes
    - Yes
    - 
  * - conv/corr
    - Only for `direct` method
    - Limited
    - Yes
    - Host only supprts 1D convolution using `fft` method
  * - hist
    - No
    - No
    - Yes
    - Single-threaded host
  * - sort
    - No
    - Yes
    - Yes
    - Single-threaded host
  * - cumsum
    - No
    - Limited
    - Yes
    - Host only supports 1 and 2D cumsum and is single-threaded