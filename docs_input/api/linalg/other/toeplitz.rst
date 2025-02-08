.. _toeplitz_func:

toeplitz
========

Generate a toeplitz matrix

`c` represents the first column of the matrix while `r` represents the first row. `c` and `r` must
have the same first value; if they don't match, the first value from `c` will be used.

Passing a single array/operator as input is equivalent to passing the conjugate of the same
input as the second parameter. 

.. doxygenfunction:: toeplitz(const T (&c)[D])
.. doxygenfunction:: toeplitz(const Op &c)
.. doxygenfunction:: toeplitz(const T (&c)[D1], const T (&r)[D2])
.. doxygenfunction:: toeplitz(const COp &cop, const ROp &rop)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/toeplitz_test.cu
   :language: cpp
   :start-after: example-begin toeplitz-test-1
   :end-before: example-end toeplitz-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/toeplitz_test.cu
   :language: cpp
   :start-after: example-begin toeplitz-test-2
   :end-before: example-end toeplitz-test-2
   :dedent:   

