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

#include "matx.h"
#include "linalg.h"

namespace matx {
  namespace st {

    // Single threaded implementation of Gauss Newton algorithm to solve a nonlinear least squares problem
    // Note: This implementation is specialized for 3 parameters.
    // TODO: For a generic N parameter version, we need a single threaded NxN matrix inversion implementation
    // https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
    template<typename OPT_FUNC, int NP, int NX, int NF, int VERBOSE=0>
    class gn_base
    {
      public:
      __device__ inline void apply_bounds(float (&x)[NP])
      {
        // Do nothing in base class
      }

      __device__ void solve(float (&x)[NP], const float (&observations)[NF], const float (&n)[NF][NX])
      {
        static_assert(NP == 3, "This implementation is specialized for 3 parameters.");
        static_assert(NX == 2, "This implementation is specialized for 2 independent variables.");
        const int n_iterations = 256;

        if (VERBOSE > 0)
        {
          printf("Initial params = %e, %e, %e\n", x[0], x[1], x[2]);
        }

        for (int iter=0; iter<n_iterations; iter++)
        {
          float local_b[3] {0.f};
          float A00 {0.f};
          float A01 {0.f};
          float A02 {0.f};
          float A11 {0.f};
          float A12 {0.f};
          float A22 {0.f};

          // Calculating
          //   b = -transpose(jac) * r
          //   A = transpose(jac) * jac
          //   (A is symmetric so only calculate the upper triangle)
          float e {0.f};

          for (int k=0; k<NF; k++)
          {
            float y = observations[k];
            float y_est, dy[NP];
            OPT_FUNC::f(x, n[k], y_est, dy);
            float r = y_est - y;
            e += r*r;

            local_b[0] -= dy[0]*r;
            local_b[1] -= dy[1]*r;
            local_b[2] -= dy[2]*r;
            A00 += dy[0]*dy[0];
            A01 += dy[0]*dy[1];
            A02 += dy[0]*dy[2];
            A11 += dy[1]*dy[1];
            A12 += dy[1]*dy[2];
            A22 += dy[2]*dy[2];
          }

          // Solve for x in Ax=b, x_est = Ainv*b
          float Ainv[3][3];
          int result = invert_symmetric_3x3(A00, A01, A02, A11, A12, A22, Ainv);
          float update[3];

          if (result == 0)
          {
            update[0] = Ainv[0][0] * local_b[0] + Ainv[0][1] * local_b[1] + Ainv[0][2] * local_b[2];
            update[1] = Ainv[1][0] * local_b[0] + Ainv[1][1] * local_b[1] + Ainv[1][2] * local_b[2];
            update[2] = Ainv[2][0] * local_b[0] + Ainv[2][1] * local_b[1] + Ainv[2][2] * local_b[2];
          }
          else
          {
            // A wasn't invertable, use gradient descent
            update[0] = local_b[0];
            update[1] = local_b[1];
            update[2] = local_b[2];
          }

          x[0] += update[0];
          x[1] += update[1];
          x[2] += update[2];

          // Apply limits
          static_cast<OPT_FUNC*>(this)->apply_bounds(x);

          if (VERBOSE > 0)
          {
            printf("Iteration %d params = %e, %e, %e; cost %e\n", iter, x[0], x[1], x[2], e);
          }

          if ((fabs(update[0]) < 0.001f) &&
              (fabs(update[1]) < 0.001f) &&
              (fabs(update[2]) < 0.001f))
          {
            break; // out of iteration for loop
          }

        }
      }
    };
  } // namespace st
}; // namespace matx