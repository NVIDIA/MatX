////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/executors/host.h"
#include "matx/executors/support.h"
#include "matx/transforms/solve/solve_common.h"

#include <vector>

namespace matx {
namespace detail {

#if MATX_EN_CPU_SOLVER
template <typename T>
__MATX_INLINE__ void DenseSolveGesvDispatch(lapack_int_t *n,
                                            lapack_int_t *nrhs,
                                            T *a,
                                            lapack_int_t *lda,
                                            lapack_int_t *piv,
                                            T *b,
                                            lapack_int_t *ldb,
                                            lapack_int_t *info)
{
  if constexpr (std::is_same_v<T, float>) {
    LAPACK_CALL(sgesv)(n, nrhs, a, lda, piv, b, ldb, info);
  }
  else if constexpr (std::is_same_v<T, double>) {
    LAPACK_CALL(dgesv)(n, nrhs, a, lda, piv, b, ldb, info);
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
    LAPACK_CALL(cgesv)(n, nrhs, a, lda, piv, b, ldb, info);
  }
  else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
    LAPACK_CALL(zgesv)(n, nrhs, a, lda, piv, b, ldb, info);
  }
}

template <typename ATensor, typename BTensor, ThreadsMode MODE>
void DenseSolveLapackLoop(ATensor &a_col,
                          BTensor &b_col,
                          [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  using T = typename remove_cvref_t<ATensor>::value_type;
  constexpr int ARANK = remove_cvref_t<ATensor>::Rank();

  lapack_int_t n = static_cast<lapack_int_t>(a_col.Size(ARANK - 1));
  lapack_int_t nrhs = static_cast<lapack_int_t>(GetDenseSolveNumRhs<ATensor, BTensor>(b_col));
  lapack_int_t info = 0;
  const auto batch_size = GetNumBatches(a_col);

  std::vector<T *> a_ptrs;
  std::vector<T *> b_ptrs;
  SetBatchPointers<BatchType::MATRIX>(a_col, a_ptrs);
  if constexpr (IsDenseSolveVectorRHS<ATensor, BTensor>()) {
    SetBatchPointers<BatchType::VECTOR>(b_col, b_ptrs);
  }
  else {
    SetBatchPointers<BatchType::MATRIX>(b_col, b_ptrs);
  }

  std::vector<lapack_int_t> piv(static_cast<size_t>(n));
  for (uint32_t i = 0; i < batch_size; i++) {
    DenseSolveGesvDispatch(&n, &nrhs, a_ptrs[i], &n, piv.data(), b_ptrs[i],
                           &n, &info);
    CheckDenseSolveInfo(static_cast<int>(info), "LAPACK", "gesv");
  }
}
#endif

} // end namespace detail

template <typename OutputTensor, typename ATensor, typename BTensor,
          ThreadsMode MODE>
void dense_solve_impl([[maybe_unused]] OutputTensor &&out,
                      [[maybe_unused]] const ATensor &a,
                      [[maybe_unused]] const BTensor &b,
                      [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(MATX_EN_CPU_SOLVER, matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured");
#if MATX_EN_CPU_SOLVER
  using T = typename remove_cvref_t<OutputTensor>::value_type;

  detail::ValidateDenseSolve(out, a, b);

  auto a_new = OpToTensor(a, exec);
  auto b_new = OpToTensor(b, exec);

  if constexpr (!is_matx_transform_op<ATensor>()) {
    if (!a_new.isSameView(a)) {
      (a_new = a).run(exec);
    }
  }

  if constexpr (!is_matx_transform_op<BTensor>()) {
    if (!b_new.isSameView(b)) {
      (b_new = b).run(exec);
    }
  }

  T *a_ptr = nullptr;
  matxAlloc(reinterpret_cast<void **>(&a_ptr), a_new.Bytes(),
            MATX_HOST_MALLOC_MEMORY);
  auto a_work_t = detail::TransposeCopy(a_ptr, a_new, exec);
  auto a_col = a_work_t.PermuteMatrix();

  if constexpr (detail::IsDenseSolveVectorRHS<decltype(a_new), decltype(b_new)>()) {
    T *b_ptr = nullptr;
    matxAlloc(reinterpret_cast<void **>(&b_ptr), b_new.Bytes(),
              MATX_HOST_MALLOC_MEMORY);
    auto b_work = make_tensor<T>(b_ptr, b_new.Shape());
    (b_work = b_new).run(exec);

    detail::DenseSolveLapackLoop(a_col, b_work, exec);
    matx::copy(out, b_work, exec);
    matxFree(b_ptr);
  }
  else {
    T *b_ptr = nullptr;
    matxAlloc(reinterpret_cast<void **>(&b_ptr), b_new.Bytes(),
              MATX_HOST_MALLOC_MEMORY);
    auto b_work_t = detail::TransposeCopy(b_ptr, b_new, exec);
    auto b_col = b_work_t.PermuteMatrix();

    detail::DenseSolveLapackLoop(a_col, b_col, exec);
    matx::copy(out, b_work_t.PermuteMatrix(), exec);
    matxFree(b_ptr);
  }

  matxFree(a_ptr);
#endif
}

} // end namespace matx
