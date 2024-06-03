.. _legendre_func:

legendre
========

Return Legendre polynomial coefficients at the input operator

.. doxygenfunction:: legendre(T1 n, T2 m, const T3 in)
.. doxygenfunction:: legendre(T1 n, T2 m, const T3 in, int (&axis)[2])
.. doxygenfunction:: legendre(T1 n, T2 m, const T3 in, cuda::std::array<int, 2> axis)  

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin legendre-test-1
   :end-before: example-end legendre-test-1
   :dedent:

