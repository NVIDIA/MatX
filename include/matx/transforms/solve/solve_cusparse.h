////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#ifndef __CUDACC_RTC__

#include <cusparse.h>

#include <numeric>

#include "matx/core/cache.h"
#include "matx/core/sparse_tensor.h"
#include "matx/core/tensor.h"
#include "matx/kernels/matvec.cuh"

namespace matx {

namespace detail {

// A tridiagonal solver that uses the cuSPARSE legacy API. The setup is
// relatively simple, which is why we forego the usual path of caching
// shared context. Rather, we just do a single-shot solve.
template <class VAL>
inline void SolveTridiagonalSystem(int m, int n, VAL *dl, VAL *dm, VAL *du,
                                   VAL *x, cudaStream_t stream) {
  cusparseHandle_t handle = nullptr; // TODO: share handle globally?
  [[maybe_unused]] cusparseStatus_t ret = cusparseCreate(&handle);
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);

  size_t workspaceSize = 0;
  void *workspace = nullptr;

  if constexpr (std::is_same_v<VAL, float>) {
    ret = cusparseSgtsv2_bufferSizeExt(handle, m, n, dl, dm, du, x, /*ldb*/ m,
                                       &workspaceSize);
  } else if constexpr (std::is_same_v<VAL, double>) {
    ret = cusparseDgtsv2_bufferSizeExt(handle, m, n, dl, dm, du, x, /*ldb*/ m,
                                       &workspaceSize);
  } else if constexpr (std::is_same_v<VAL, cuFloatComplex>) {
    ret = cusparseCgtsv2_bufferSizeExt(handle, m, n, dl, dm, du, x, /*ldb*/ m,
                                       &workspaceSize);
  } else if constexpr (std::is_same_v<VAL, cuDoubleComplex>) {
    ret = cusparseZgtsv2_bufferSizeExt(handle, m, n, dl, dm, du, x, /*ldb*/ m,
                                       &workspaceSize);
  } else {
    MATX_THROW(matxNotSupported, "Unsupported type for tri-diagonal solve");
  }
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);

  matxAlloc((void **)&workspace, workspaceSize, MATX_DEVICE_MEMORY, stream);

  if constexpr (std::is_same_v<VAL, float>) {
    ret = cusparseSgtsv2(handle, m, n, dl, dm, du, x, /*ldb*/ m, workspace);
  } else if constexpr (std::is_same_v<VAL, double>) {
    ret = cusparseDgtsv2(handle, m, n, dl, dm, du, x, /*ldb*/ m, workspace);
  } else if constexpr (std::is_same_v<VAL, cuFloatComplex>) {
    ret = cusparseCgtsv2(handle, m, n, dl, dm, du, x, /*ldb*/ m, workspace);
  } else if constexpr (std::is_same_v<VAL, cuDoubleComplex>) {
    ret = cusparseZgtsv2(handle, m, n, dl, dm, du, x, /*ldb*/ m, workspace);
  }
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);

  matxFree(workspace, stream);

  ret = cusparseDestroy(handle);
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);
}

// Batched version of tridiagonal solver.
template <class VAL>
inline void SolveBatchedTridiagonalSystem(int m, int b, VAL *dl, VAL *dm,
                                          VAL *du, VAL *x,
                                          cudaStream_t stream) {
  cusparseHandle_t handle = nullptr; // TODO: share handle globally?
  [[maybe_unused]] cusparseStatus_t ret = cusparseCreate(&handle);
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);

  size_t workspaceSize = 0;
  void *workspace = nullptr;

  if constexpr (std::is_same_v<VAL, float>) {
    ret = cusparseSgtsv2StridedBatch_bufferSizeExt(handle, m, dl, dm, du, x, b,
                                                   m, &workspaceSize);
  } else if constexpr (std::is_same_v<VAL, double>) {
    ret = cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, dl, dm, du, x, b,
                                                   m, &workspaceSize);
  } else if constexpr (std::is_same_v<VAL, cuFloatComplex>) {
    ret = cusparseCgtsv2StridedBatch_bufferSizeExt(handle, m, dl, dm, du, x, b,
                                                   m, &workspaceSize);
  } else if constexpr (std::is_same_v<VAL, cuDoubleComplex>) {
    ret = cusparseZgtsv2StridedBatch_bufferSizeExt(handle, m, dl, dm, du, x, b,
                                                   m, &workspaceSize);
  } else {
    MATX_THROW(matxNotSupported, "Unsupported type for tri-diagonal solve");
  }
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);

  matxAlloc((void **)&workspace, workspaceSize, MATX_DEVICE_MEMORY, stream);

  if constexpr (std::is_same_v<VAL, float>) {
    ret = cusparseSgtsv2StridedBatch(handle, m, dl, dm, du, x, b, m, workspace);
  } else if constexpr (std::is_same_v<VAL, double>) {
    ret = cusparseDgtsv2StridedBatch(handle, m, dl, dm, du, x, b, m, workspace);
  } else if constexpr (std::is_same_v<VAL, cuFloatComplex>) {
    ret = cusparseCgtsv2StridedBatch(handle, m, dl, dm, du, x, b, m, workspace);
  } else if constexpr (std::is_same_v<VAL, cuDoubleComplex>) {
    ret = cusparseZgtsv2StridedBatch(handle, m, dl, dm, du, x, b, m, workspace);
  }
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);

  matxFree(workspace, stream);

  ret = cusparseDestroy(handle);
  MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxSolverError);
}

template <typename Op>
__MATX_INLINE__ auto getCuSparseSolveSupportedTensor(const Op &in,
                                                     cudaStream_t stream) {
  const auto func = [&]() {
    if constexpr (is_tensor_view_v<Op>) {
      return in.Stride(Op::Rank() - 1) == 1;
    } else {
      return true;
    }
  };
  return GetSupportedTensor(in, func, MATX_ASYNC_DEVICE_MEMORY, stream);
}

} // end namespace detail

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
void sparse_dia_solve_impl(TensorTypeC &C, const TensorTypeA &a,
                           const TensorTypeB &B, const cudaExecutor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  // Transform into supported form.
  auto b = getCuSparseSolveSupportedTensor(B, stream);
  auto c = getCuSparseSolveSupportedTensor(C, stream);
  if (!is_matx_transform_op<TensorTypeB>() && !b.isSameView(B)) {
    (b = B).run(stream);
  }

  using atype = TensorTypeA;
  using btype = decltype(b);
  using ctype = decltype(c);

  using TA = typename atype::value_type;
  using TB = typename btype::value_type;
  using TC = typename ctype::value_type;

  static constexpr int RANKA = atype::Rank();
  static constexpr int RANKB = btype::Rank();
  static constexpr int RANKC = ctype::Rank();

  // Restrictions.
  static_assert(atype::Format::isDIAI(),
                "Tridiagonal solve requires I-index DIAG");
  static_assert(RANKA == 2 && RANKB == 2 && RANKC == 2,
                "tensors must have rank-2");
  static_assert(std::is_same_v<TC, TA> && std::is_same_v<TC, TB>,
                "tensors must have the same data type");
  static_assert(std::is_same_v<TC, float> || std::is_same_v<TC, double> ||
                    std::is_same_v<TC, cuda::std::complex<float>> ||
                    std::is_same_v<TC, cuda::std::complex<double>>,
                "unsupported data type");
  MATX_ASSERT(                                  // Note: B,C transposed!
      a.Size(RANKA - 1) == a.Size(RANKA - 2) && // square
          a.Size(RANKA - 1) == b.Size(RANKB - 1) &&
          a.Size(RANKA - 2) == c.Size(RANKC - 1) &&
          b.Size(RANKB - 2) == c.Size(RANKC - 2),
      matxInvalidSize);
  MATX_ASSERT(b.Stride(RANKB - 1) == 1 && c.Stride(RANKC - 1) == 1,
              matxInvalidParameter);

  // These are *run-time* checks.
  if (!c.isSameView(b)) {
    MATX_THROW(matxNotSupported, "Tridiagonal solve overwrites rhs");
  }
  using CRD = typename atype::crd_type;
  CRD *diags = a.CRDData(0);
  const index_t numD = a.crdSize(0);
  // TODO: we should also check that offsets = {-1,0,1} (host and device)?
  MATX_ASSERT(numD == 3, matxInvalidParameter);
  using T = std::conditional_t<
      std::is_same_v<TA, cuda::std::complex<double>>, cuDoubleComplex,
      std::conditional_t<std::is_same_v<TA, cuda::std::complex<float>>,
                         cuFloatComplex, TA>>;
  T *AD = reinterpret_cast<T *>(a.Data());
  T *BD = reinterpret_cast<T *>(b.Data());
  const int m = static_cast<int>(a.Size(RANKA - 2));
  const int n = static_cast<int>(b.Size(RANKB - 2));
  detail::SolveTridiagonalSystem<T>(m, n, AD, AD + m, AD + m + m, BD, stream);

  // Copy transformed output back.
  if (!c.isSameView(C)) {
    (C = c).run(stream);
  }
}

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
void sparse_batched_dia_solve_impl(TensorTypeC &C, const TensorTypeA &a,
                                   const TensorTypeB &B,
                                   const cudaExecutor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  // Transform into supported form.
  auto b = getCuSparseSolveSupportedTensor(B, stream);
  auto c = getCuSparseSolveSupportedTensor(C, stream);
  if (!is_matx_transform_op<TensorTypeB>() && !b.isSameView(B)) {
    (b = B).run(stream);
  }

  using atype = TensorTypeA;
  using btype = decltype(b);
  using ctype = decltype(c);

  using TA = typename atype::value_type;
  using TB = typename btype::value_type;
  using TC = typename ctype::value_type;

  static constexpr int RANKA = atype::Rank();
  static constexpr int RANKB = btype::Rank();
  static constexpr int RANKC = ctype::Rank();

  // Restrictions.
  static_assert(atype::Format::isBatchedDIAIUniform(),
                "Tridiagonal solve requires I-index DIAG");
  static_assert(RANKA == 3 && RANKB == 1 && RANKC == 1,
                "tensors must define batched system");
  static_assert(std::is_same_v<TC, TA> && std::is_same_v<TC, TB>,
                "tensors must have the same data type");
  static_assert(std::is_same_v<TC, float> || std::is_same_v<TC, double> ||
                    std::is_same_v<TC, cuda::std::complex<float>> ||
                    std::is_same_v<TC, cuda::std::complex<double>>,
                "unsupported data type");
  MATX_ASSERT(a.Size(RANKA - 1) == a.Size(RANKA - 2) && // square after batch
                  a.Size(RANKA - 3) * a.Size(RANKA - 2) == b.Size(RANKB - 1) &&
                  a.Size(RANKA - 3) * a.Size(RANKA - 2) == c.Size(RANKC - 1),
              matxInvalidSize);
  MATX_ASSERT(b.Stride(RANKB - 1) == 1 && c.Stride(RANKC - 1) == 1,
              matxInvalidParameter);

  // These are *run-time* checks.
  if (!c.isSameView(b)) {
    MATX_THROW(matxNotSupported, "Tridiagonal solve overwrites rhs");
  }
  using CRD = typename atype::crd_type;
  CRD *diags = a.CRDData(0);
  const index_t numD = a.crdSize(0);
  // TODO: we should also check that offsets = {-1,0,1} (host and device)?
  MATX_ASSERT(numD == 3, matxInvalidParameter);
  using T = std::conditional_t<
      std::is_same_v<TA, cuda::std::complex<double>>, cuDoubleComplex,
      std::conditional_t<std::is_same_v<TA, cuda::std::complex<float>>,
                         cuFloatComplex, TA>>;
  T *AD = reinterpret_cast<T *>(a.Data());
  T *BD = reinterpret_cast<T *>(b.Data());
  const int m = static_cast<int>(a.Size(RANKA - 2));
  const int batch = static_cast<int>(a.Size(RANKA - 3));
  const auto l = b.Size(RANKB - 1);
  MATX_ASSERT(batch * m == l, matxSolverError);
  detail::SolveBatchedTridiagonalSystem<T>(m, batch, AD, AD + l, AD + l + l, BD,
                                           stream);

  // Copy transformed output back.
  if (!c.isSameView(C)) {
    (C = c).run(stream);
  }
}

} // end namespace matx

#endif