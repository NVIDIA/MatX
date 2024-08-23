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

    template<typename T, size_t M>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T norm(const T (&A)[M])
    {
      T val = 0;
      for (size_t m=0; m<M; m++)
      {
        val += A[m]*A[m];
      }
      return helper_sqrt(val);
    }

    template<typename T, size_t M>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T invnorm(const T (&A)[M])
    {
      T val = 0;
      for (size_t m=0; m<M; m++)
      {
        val += A[m]*A[m];
      }
      return helper_invsqrt(val);
    }

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
