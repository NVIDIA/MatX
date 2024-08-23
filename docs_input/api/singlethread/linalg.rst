.. _st_linag:

There is a small set of single threaded basic linear algebra utilities

linalg
======

Matrix Multiply (GEMM)
----------------------

`matmul` performs a transformation for Generic Matrix Multiplies (GEMMs) for real-valued arrays

.. doxygenfunction:: matx::st::matmul_AxB(T (&C)[M][N], const T (&A)[M][K], const T (&B)[K][N])
.. doxygenfunction:: matx::st::matmul_ATxB(T (&C)[M][N], const T (&A)[K][M], const T (&B)[K][N])
.. doxygenfunction:: matx::st::matmul_AxBT(T (&C)[M][N], const T (&A)[M][K], const T (&B)[N][K])
.. doxygenfunction:: matx::st::matmul_AxB(T (&C)[M], const T (&A)[M][K], const T (&B)[K])
.. doxygenfunction:: matx::st::matmul_AxBT(T (&C)[M][N], const T (&A)[M], const T (&B)[N])
.. doxygenfunction:: matx::st::matmul_ATxB(T (&C)[N], const T (&A)[M][N], const T (&B)[M])

Dot product
-----------

`dot` performs a dot product for real-valued arrays

.. doxygenfunction:: matx::st::dot_AB(T &C, const T (&A)[M], const T (&B)[M])

Norm
----

`norm` calculates a Frobenius norm, `invnorm` calculates its inverse

.. doxygenfunction:: matx::st::norm(const T (&A)[M])
.. doxygenfunction:: matx::st::invnorm(const T (&A)[M])

Inverses
--------

.. doxygenfunction:: matx::st::invert_symmetric_3x3(T A00, T A01, T A02, T A11, T A12, T A22, T (&Ainv)[3][3])
