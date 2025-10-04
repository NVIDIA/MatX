////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
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

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/cache.h"
#include "matx/executors/host.h"
#include "matx/executors/support.h"
#include "matx/transforms/svd/svd_cuda.h"
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/svd/svd_lapack.h"
#endif

#include <cstdio>
#include <numeric>
#include <cuda/std/__algorithm/min.h>

namespace matx {

/**
 * Returns an appropriate rcond based on the inner type. This is slightly
 * higher than the machine epsilon, as these work better to mask small/zero singular
 * values in singular or ill-conditioned matrices.
 */
template <typename T>
__MATX_INLINE__ constexpr float get_default_rcond() {
  if constexpr (is_fp32_inner_type_v<T>) {
    return 1e-6f;
  } else {
    return 1e-15f;
  }
}

/**
 * Compute the Moore-penrose pseudo-inverse of a matrix
 *
 * Perfom a generalized inverse of a matrix using its singular-value decomposition (SVD).
 * It automatically removes small singular values for stability.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor view
 * @param a
 *   Input matrix A
 * @param exec
 *   Executor
 * @param rcond
 *   Cutoff for small singular values. For stability, singular values
 *   smaller than rcond * largest_singular_value are set to 0 for each matrix
 *   in the batch. By default, rcond is the machine epsilon of the tensor dtype.
 */
template <typename OutputTensor, typename InputTensor, typename Executor>
void pinv_impl(OutputTensor &out,
              const InputTensor &a,
              const Executor &exec,
              float rcond = get_default_rcond<typename InputTensor::value_type>())
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(!(is_host_executor_v<Executor> && !MATX_EN_CPU_SOLVER), matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");
  
  using T1 = typename InputTensor::value_type;
  using inner_type = typename inner_op_type_t<T1>::type;
  constexpr int RANK = InputTensor::Rank();

  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim, "Output and input tensors must have same rank for pinv()");
  MATX_STATIC_ASSERT_STR(RANK >= 2, matxInvalidDim, "Input/Output tensor must be rank 2 or higher");
  MATX_STATIC_ASSERT_STR((std::is_same_v<T1, typename OutputTensor::value_type>), matxInavlidType, "A and Out types must match");

  const index_t m = a.Size(RANK-2); // rows
  const index_t n = a.Size(RANK-1); // cols
  const index_t k = cuda::std::min(m, n);

  // Batch size checks
  for(int i = 0 ; i < RANK-2; i++) {
    MATX_ASSERT_STR(out.Size(i) == a.Size(i), matxInvalidDim, "Out and A must have the same batch sizes");
  }

  MATX_ASSERT_STR((out.Size(RANK-1) == m) && (out.Size(RANK-2) == n), matxInvalidSize,
      "Out must be ... x n x m for A ... x m x n");
  
  /* 
    Need to perform pinv = V * S^-1 * U^H where svd(A) = U * S * V^H.
    Alternatively, can run svd(A^H) to get V, S, U^H
  */ 

  // Allocate v, s, ut
  auto aShape = a.Shape();
  auto outShape = out.Shape();

  // v is ... x n x k
  auto vShape = outShape;
  vShape[RANK-1] = k;

  // s is ... x k
  cuda::std::array<index_t, RANK-1> sShape;
  for(int i = 0; i < RANK-2; i++) {
    sShape[i] = aShape[i];
  }
  sShape[RANK-2] = k;

  // ut is ... x k x m
  auto utShape = outShape;
  utShape[RANK-2] = k;

  tensor_t<T1, RANK> v;
  tensor_t<inner_type, RANK-1> s;
  tensor_t<bool, RANK-1> s_mask;
  tensor_t<T1, RANK> ut;

  if constexpr (is_cuda_executor_v<Executor>) {
    const auto stream = exec.getStream();
    make_tensor(v, vShape, MATX_ASYNC_DEVICE_MEMORY, stream);
    make_tensor(s, sShape, MATX_ASYNC_DEVICE_MEMORY, stream);
    make_tensor(s_mask, sShape, MATX_ASYNC_DEVICE_MEMORY, stream);
    make_tensor(ut, utShape, MATX_ASYNC_DEVICE_MEMORY, stream);
  } else {
    make_tensor(v, vShape, MATX_HOST_MALLOC_MEMORY);
    make_tensor(s, sShape, MATX_HOST_MALLOC_MEMORY);
    make_tensor(s_mask, sShape, MATX_HOST_MALLOC_MEMORY);
    make_tensor(ut, utShape, MATX_HOST_MALLOC_MEMORY);
  }

  svd_impl(v, s, ut, transpose_matrix(conj(a)), exec, SVDMode::REDUCED);

  // discard small singular values
  cuda::std::array<index_t, RANK-1> cutoffShape;
  cutoffShape.fill(matxKeepDim);
  cutoffShape[RANK-2] = k; // repeat across last dim

  auto cutoff = rcond * max(s, {RANK-2});
  auto cutoff_add_axis = clone<RANK-1>(cutoff, cutoffShape);

  // Need to explicitly run before inverting s since the mask needs to be created
  // based on original singular values.
  (s_mask = s > cutoff_add_axis).run(exec);
  
  // IF required to avoid nans when singular value is 0
  (IF(s != inner_type(0), s = inner_type(1) / s)).run(exec);
  (s *= s_mask).run(exec);

  // V = V * S^-1
  auto dShape = v.Shape();
  dShape.fill(matxKeepDim);
  dShape[RANK-2] = n;
  auto d = clone<RANK>(s, dShape);
  (v = v * d).run(exec);
  
  // (V * S-1) * UT
  matmul_impl(out, v, ut, exec);
}

} // end namespace matx