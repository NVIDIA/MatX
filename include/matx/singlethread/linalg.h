////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2024, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////


#pragma once

#include <stdio.h>
#include <algorithm>
#include <type_traits>
#include <math.h>
#include "matx.h"

namespace matx {

  namespace st
  {
    namespace detail
    {
      template<typename T>
      __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T helper_sqrt(const T x)
      {
        static_assert(std::is_same_v<T,float>, "helper_sqrt only implemented for float");
        if constexpr (std::is_same_v<T,float>)
        {
          return sqrtf(x);
        }
      }

      template<typename T>
      __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T helper_invsqrt(const T x)
      {
        static_assert(std::is_same_v<T,float>, "helper_invsqrt only implemented for float");
        if constexpr (std::is_same_v<T,float>)
        {
          return 1.f / sqrtf(x);
        }
      }
    };


    /**
     * Print a compile time constant sized 2D array
     *
     * This overload for matprint() prints a 2D array
     *
     * @tparam T
     *   Scalar element type contained within the 2D array
     *   Note: Presently only supports T = float due to lack
     *         of device function PrintVal support
     *
     * @tparam M
     *   Outer dimension of 2D array
     *
     * @tparam N
     *   Inner dimension of 2D array
     *
     * @param a
     *   2D array to print
     *
     */
    template<typename T, size_t M, size_t N>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matprint(const T (&a)[M][N])
    {
      static_assert(std::is_same_v<float,T>, "matprint() only implemented for types float");
      for (size_t m=0; m<M; m++)
      {
        for (size_t n=0; n<N; n++)
        {
          if constexpr (std::is_same_v<float,T>)
          {
            printf("%f, ",a[m][n]);
          }
        }
        printf("\n");
      }
      printf("\n");
    }

    /**
     * Print a compile time constant sized 1D array
     *
     * This overload for matprint() prints a 1D array.
     *
     * @tparam T
     *   Scalar element type contained within the 1D array
     *   Note: Presently only supports T = float due to lack
     *         of device function PrintVal support
     *
     * @tparam M
     *   Dimension of 1D array
     *
     * @param a
     *   1D array to print
     *
     */
    template<typename T, size_t M>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matprint(const T (&a)[M])
    {
      static_assert(std::is_same_v<float,T>, "matprint() only implemented for types float");
      for (size_t m=0; m<M; m++)
      {
        if constexpr (std::is_same_v<float,T>)
        {
          printf("%f,\n",a[m]);
        }
      }
    }

    /**
     * Matrix multiply C = A @ B
     *
     * matmul_AxB() multiplies the 2D array A (dimension MxK) by the 2D array B
     * (dimension KxN) and returns the result in 2D array C (dimension MxN)
     *
     * @tparam T
     *   Scalar element type contained within the arrays
     *
     * @tparam M
     *   Outer dimension of arrays C and A
     *
     * @tparam N
     *   Inner dimension of arrays C and B
     *
     * @tparam K
     *   Inner dimension of array A and outer dimension of array B
     *
     * @param[out] C
     *   Result output array
     *
     * @param[in] A,B
     *   Input arrays
     *
     */
    template<typename T, size_t M, size_t N, size_t K>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matmul_AxB(T (&C)[M][N], const T (&A)[M][K], const T (&B)[K][N])
    {
      #pragma unroll
      for (size_t m=0; m<M; m++)
      {
        #pragma unroll
        for (size_t n=0; n<N; n++)
        {
          T c_val = 0;
          #pragma unroll
          for (size_t k=0; k<K; k++)
          {
            c_val += A[m][k] * B[k][n];
          }
          C[m][n] = c_val;
        }
      }
    }

    /**
     * Matrix multiply C = A.T @ B
     *
     * matmul_ATxB() multiplies the transpose of the 2D array A (dimension KxM)
     * by the 2D array B (dimension KxN) and returns the result in 2D array
     * C (dimension MxN)
     *
     * @tparam T
     *   Scalar element type contained within the arrays
     *
     * @tparam M
     *   Outer dimension of array C and inner dimension of array A
     *
     * @tparam N
     *   Inner dimension of arrays C and B
     *
     * @tparam K
     *   Outer dimension of arrays A and B
     *
     * @param[out] C
     *   Result output array
     *
     * @param[in] A,B
     *   Input arrays
     *
     */
    template<typename T, size_t M, size_t N, size_t K>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matmul_ATxB(T (&C)[M][N], const T (&A)[K][M], const T (&B)[K][N])
    {
      #pragma unroll
      for (size_t m=0; m<M; m++)
      {
        #pragma unroll
        for (size_t n=0; n<N; n++)
        {
          T c_val = 0;
          #pragma unroll
          for (size_t k=0; k<K; k++)
          {
            c_val += A[k][m] * B[k][n];
          }
          C[m][n] = c_val;
        }
      }
    }

    /**
     * Matrix multiply C = A @ B.T
     *
     * matmul_AxBT() multiplies the 2D array A (dimension MxK) by the transpose
     * of the 2D array B (dimension NxK) and returns the result in 2D array
     * C (dimension MxN)
     *
     * @tparam T
     *   Scalar element type contained within the arrays
     *
     * @tparam M
     *   Outer dimension of arrays C and A
     *
     * @tparam N
     *   Inner dimension of array C and outer dimension of array B
     *
     * @tparam K
     *   Inner dimension of arrays A and B
     *
     * @param[out] C
     *   Result output array
     *
     * @param[in] A,B
     *   Input arrays
     *
     */
    template<typename T, size_t M, size_t N, size_t K>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matmul_AxBT(T (&C)[M][N], const T (&A)[M][K], const T (&B)[N][K])
    {
      #pragma unroll
      for (size_t m=0; m<M; m++)
      {
        #pragma unroll
        for (size_t n=0; n<N; n++)
        {
          T c_val = 0;
          #pragma unroll
          for (size_t k=0; k<K; k++)
          {
            c_val += A[m][k] * B[n][k];
          }
          C[m][n] = c_val;
        }
      }
    }

    /**
     * Matrix multiply C = A @ B
     *
     * matmul_AxB() multiplies the 2D array A (dimension MxK) by the 1D array B
     * (dimension Kx1) and returns the result in 1D array C (dimension Mx1)
     *
     * @tparam T
     *   Scalar element type contained within the arrays
     *
     * @tparam M
     *   Outer dimension of arrays C and A
     *
     * @tparam K
     *   Inner dimension of array A and outer dimension of array B
     *
     * @param[out] C
     *   Result output array
     *
     * @param[in] A,B
     *   Input arrays
     *
     */
    template<typename T, size_t M, size_t K>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matmul_AxB(T (&C)[M], const T (&A)[M][K], const T (&B)[K])
    {
      #pragma unroll
      for (size_t m=0; m<M; m++)
      {
        T c_val = 0;
        #pragma unroll
        for (size_t k=0; k<K; k++)
        {
          c_val += A[m][k] * B[k];
        }
        C[m] = c_val;
      }
    }

    /**
     * Matrix multiply C = A @ B.T
     *
     * matmul_AxBT() multiplies the 1D array A (dimension Mx1) by the transpose
     * of the 1D array B (dimension Nx1) and returns the result in 2D array C (dimension MxN)
     *
     * @tparam T
     *   Scalar element type contained within the arrays
     *
     * @tparam M
     *   Outer dimension of arrays C and A
     *
     * @tparam N
     *   Inner dimension of array C and outer dimension of B
     *
     * @param[out] C
     *   Result output array
     *
     * @param[in] A,B
     *   Input arrays
     *
     */
    template<typename T, size_t M, size_t N>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matmul_AxBT(T (&C)[M][N], const T (&A)[M], const T (&B)[N])
    {
      #pragma unroll
      for (size_t m=0; m<M; m++)
      {
        #pragma unroll
        for (size_t n=0; n<N; n++)
        {
          C[m][n] = A[m] * B[n];
        }
      }
    }

    /**
     * Matrix multiply C = A.T @ B
     *
     * matmul_ATxB() multiplies the transpose of the 2D array A (dimension MxN)
     * by the 1D array B (dimension Mx1) and returns the result in 2D array
     * C (dimension Nx1)
     *
     * @tparam T
     *   Scalar element type contained within the arrays
     *
     * @tparam M
     *   Outer dimension of array A
     *
     * @tparam N
     *   Outer dimension of array C and inner dimension of array A
     *
     * @param[out] C
     *   Result output array
     *
     * @param[in] A,B
     *   Input arrays
     *
     */
    template<typename T, size_t M, size_t N>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void matmul_ATxB(T (&C)[N], const T (&A)[M][N], const T (&B)[M])
    {
      #pragma unroll
      for (size_t n=0; n<N; n++)
      {
        T c_val = 0;
        #pragma unroll
        for (size_t m=0; m<M; m++)
        {
          c_val += A[m][n] * B[m];
        }
        C[n] = c_val;
      }
    }

    /**
     * Dot product C = A.dot(B)
     *
     * dot_AxB() calculates the scalar dot product for 1D vectors A and B
     *
     * @tparam T
     *   Scalar element type contained within the vector
     *
     * @tparam M
     *   Dimension of vectors A and B
     *
     * @param[out] C
     *   Result scalar
     *
     * @param[in] A,B
     *   Input vectors
     *
     */
    template<typename T, size_t M>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void dot_AB(T &C, const T (&A)[M], const T (&B)[M])
    {
      C = 0;
      #pragma unroll
      for (size_t m=0; m<M; m++)
      {
        C += A[m] * B[m];
      }
    }

    /**
     * Computes the Frobenius norm of vector A
     *
     * @tparam T
     *   Scalar element type contained within the vector
     *
     * @tparam M
     *   Dimension of vector A
     *
     * @param[in] A
     *   Input vector
     *
     * @returns the Frobenius norm of A
     *
     */
    template<typename T, size_t M>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T norm(const T (&A)[M])
    {
      T val = 0;
      for (size_t m=0; m<M; m++)
      {
        val += A[m]*A[m];
      }
      return detail::helper_sqrt(val);
    }

    /**
     * Computes the inverse of the Frobenius norm of vector A
     *
     * @tparam T
     *   Scalar element type contained within the vector
     *
     * @tparam M
     *   Dimension of vector A
     *
     * @param[in] A
     *   Input vector
     *
     * @returns the inverse of the Frobenius norm of A
     *
     */
    template<typename T, size_t M>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T invnorm(const T (&A)[M])
    {
      T val = 0;
      for (size_t m=0; m<M; m++)
      {
        val += A[m]*A[m];
      }
      return detail::helper_invsqrt(val);
    }

    /**
     * Invert a symmetric 3x3 matrix
     *
     * invert_symmetric_3x3() takes the upper triangle of elements from a symmetric
     * 3x3 matrix and computes the inverse if it exists
     *
     * @tparam T
     *   Scalar element type contained within the array
     *
     * @param[in] A00,A01,A02,A11,A12,A22
     *   elements of A.  Only the non-redundant elements are necessary.
     *
     * @param[out] Ainv
     *   Output result is the inverse of array A if return value is non-negative
     *
     * @returns 0 if successful, -1 if det(A) was 0 and array is singular (non-invertable)
     *
     */
    template<typename T>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ int invert_symmetric_3x3(T A00, T A01, T A02, T A11, T A12, T A22, T (&Ainv)[3][3])
    {
      // Calculate the determinant
      T det = A00 * (A11 * A22 - A12 * A12) -
              A01 * (A01 * A22 - A12 * A02) +
              A02 * (A01 * A12 - A11 * A02);
      if (det == 0)
      {
        return -1;
      }

      T one = 1;
      T det_inv = one / det;

      // Calculate the adjugate and divide by the determinent
      Ainv[0][0] = (A11 * A22 - A12 * A12) * det_inv;
      Ainv[0][1] = (A02 * A12 - A01 * A22) * det_inv;
      Ainv[0][2] = (A01 * A12 - A02 * A11) * det_inv;
      Ainv[1][0] = (A12 * A02 - A01 * A22) * det_inv;
      Ainv[1][1] = (A00 * A22 - A02 * A02) * det_inv;
      Ainv[1][2] = (A02 * A01 - A00 * A12) * det_inv;
      Ainv[2][0] = (A01 * A12 - A11 * A02) * det_inv;
      Ainv[2][1] = (A01 * A02 - A00 * A12) * det_inv;
      Ainv[2][2] = (A00 * A11 - A01 * A01) * det_inv;
      return 0;
    }

  }; // namespace st
}; // namespace matx
