.. _as_complex_float_func:

as_complex_float
=================

Cast an operator to cuda::std::complex<float>

.. doxygenfunction:: matx::as_complex_float(const T &t)
.. doxygenfunction:: matx::as_complex_float(const T1 &t1, const T2 &t2)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin as_complex_float-test-1
   :end-before: example-end as_complex_float-test-1
   :dedent:

.. literalinclude:: ../../../test/00_operators/OperatorTests.cu
   :language: cpp
   :start-after: example-begin as_complex_float-test-2
   :end-before: example-end as_complex_float-test-2
   :dedent:
