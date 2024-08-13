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
#include "matx/transforms/lu/lu_cuda.h"
#ifdef MATX_EN_CPU_SOLVER
  #include "matx/transforms/lu/lu_lapack.h"
#endif

#include <cstdio>
#include <numeric>

namespace matx {

/**
 * Compute the determinant of a matrix
 *
 * Computes the terminant of a matrix by first computing the LU composition,
 * then reduces the product of the diagonal elements of U. The input and output
 * parameters may be the same tensor. In that case, the input is destroyed and
 * the output is stored in-place.
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
 */
template <typename OutputTensor, typename InputTensor, typename Executor>
void det_impl(OutputTensor &out, const InputTensor &a,
         const Executor &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(!(is_host_executor_v<Executor> && !MATX_EN_CPU_SOLVER), matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");

  static_assert(OutputTensor::Rank() == InputTensor::Rank() - 2, "Output tensor rank must be 2 less than input for det()");
  constexpr int RANK = InputTensor::Rank();
  using value_type = typename OutputTensor::value_type;
  using piv_value_type = std::conditional_t<is_cuda_executor_v<Executor>, int64_t, lapack_int_t>;
  
  auto a_new = OpToTensor(a, exec);

  if(!a_new.isSameView(a)) {
    (a_new = a).run(exec);
  }

  // Get parameters required by these tensors
  cuda::std::array<index_t, RANK - 1> s;

  // Set batching dimensions of piv
  for (int i = 0; i < RANK - 2; i++) {
    s[i] = a_new.Size(i);
  }

  index_t piv_len = cuda::std::min(a_new.Size(RANK - 1), a_new.Size(RANK - 2));
  s[RANK - 2] = piv_len;

  tensor_t<piv_value_type, RANK-1> piv;
  tensor_t<value_type, RANK> ac;

  if constexpr (is_cuda_executor_v<Executor>) {
    const auto stream = exec.getStream();
    make_tensor(piv, s, MATX_ASYNC_DEVICE_MEMORY, stream);
    make_tensor(ac, a_new.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
  } else {
    make_tensor(piv, s, MATX_HOST_MALLOC_MEMORY);
    make_tensor(ac, a_new.Shape(), MATX_HOST_MALLOC_MEMORY);
  }

  lu_impl(ac, piv, a_new, exec);

  // Determinant sign adjustment based on piv permutation
  // Create indices corresponding to no permutation to compare against
  auto pIdxShape = s;
  pIdxShape[RANK-2] = matxKeepDim;
  auto idx = range<0, 1, piv_value_type>({piv_len}, 1, 1);  // piv has 1-based indexing
  auto piv_idx = clone(idx, pIdxShape);
  
  // Calculate number of swaps for each matrix in the batch
  auto swap_count = sum(as_type<piv_value_type>(piv != piv_idx), {RANK-2});

  // Even number of swaps means positive and odd means negative
  auto signs = as_type<typename inner_op_type_t<value_type>::type>((swap_count & 1) * -2 + 1);
  (out = signs * prod(diag(ac), {RANK-2})).run(exec);
}

} // end namespace matx