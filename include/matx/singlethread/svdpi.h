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
#include "linalg.h"

namespace matx {
  namespace st {
    template<typename T, size_t M, size_t N, size_t MIN_M_N, size_t MAX_M_N>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ void svdpi(const T (&A)[M][N], T (&U)[M][MIN_M_N], T(&S)[MIN_M_N], T (&VT)[MIN_M_N][N], const T (&x0)[MAX_M_N], int num_values, int max_iter)
    {
      T Ap[M][N];
      for (size_t m=0; m<M; m++)
      {
        for (size_t n=0; n<N; n++)
        {
          Ap[m][n] = A[m][n];
        }
      }

      bool ufirst = M >= N;
      for (int k=0; k<num_values; k++)
      {
        if (ufirst)
        {
          // Find column of u using power iteration method
          T u[M];
          T u_next[M];
          for (size_t m=0; m<M; m++)
          {
            u[m] = x0[m];
          }
          T AA[M][M];

          // Compute AA = Ap @ Ap.T
          matmul_AxBT(AA, Ap, Ap);

          for (int iter=0; iter<max_iter; iter++)
          {
            // Compute x = AA @ x
            matmul_AxB(u_next, AA, u);

            // Compute s = np.linalg.norm(x)
            T s = norm(u_next);

            // Compute x = x / s
            T max_err = 0;
            for (size_t m=0; m<M; m++)
            {
              T u_next_div_s = u_next[m] / s;
              T delta = fabs(u_next_div_s - u[m]);
              max_err = cuda::std::max(max_err, delta);
              u[m] = u_next_div_s;
            }
            const T EPS = 1e-8f;
            if (max_err < EPS)
            {
              break; // out of iter loop
            }
          }

          T v[N];

          // Compute v = Ap.T @ x
          matmul_ATxB(v, Ap, u);
          T s = norm(v);
          for (size_t n=0; n<N; n++)
          {
            v[n] /= s;
            VT[k][n] = v[n];
          }

          for (size_t m=0; m<M; m++)
          {
            U[m][k] = u[m];
          }
          S[k] = s;

          // Compute uv = u @ v.T
          T uv[M][N];
          matmul_AxBT(uv, u, v);

          // Compute Ap = Ap - np.ones((n,m))*s * uv
          #pragma unroll
          for (size_t m=0; m<M; m++)
          {
            for (size_t n=0; n<N; n++)
            {
              Ap[m][n] -= s*uv[m][n];
            }
          }
        }
      }
    }
  }; // namespace st
}; // namespace matx
