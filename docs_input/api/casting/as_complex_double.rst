.. _as_complex_double_func:

as_complex_double
=================

Cast an operator to cuda::std::complex<double>

.. doxygenfunction:: matx::as_complex_double(const T &t)
.. doxygenfunction:: matx::as_complex_double(const T1 &t1, const T2 &t2)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/cast_test.cu
   :language: cpp
   :start-after: example-begin as_complex_double-test-1
   :end-before: example-end as_complex_double-test-1
   :dedent:

.. literalinclude:: ../../../test/00_operators/complex_cast_test.cu
   :language: cpp
   :start-after: example-begin as_complex_double-test-2
   :end-before: example-end as_complex_double-test-2
   :dedent:
