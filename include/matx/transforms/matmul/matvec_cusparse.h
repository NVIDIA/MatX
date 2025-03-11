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

#include <cusparse.h>

#include <numeric>

#include "matx/core/cache.h"
#include "matx/core/sparse_tensor.h"
#include "matx/core/tensor.h"

namespace matx {

namespace detail {

/**
 * Parameters needed to execute a cuSPARSE SpMV.
 */
struct MatVecCUSPARSEParams_t {
  MatXDataType_t dtype;
  MatXDataType_t ptype;
  MatXDataType_t ctype;
  cudaStream_t stream;
  float alpha;
  float beta;
  index_t nse;
  index_t m;
  index_t n;
  cusparseOperation_t opA;
  // Matrix handles in cuSPARSE are data specific (unlike e.g. cuBLAS
  // where the same plan can be shared between different data buffers).
  void *ptrA0;
  void *ptrA1;
  void *ptrA2;
  void *ptrA3;
  void *ptrA4;
  void *ptrB;
  void *ptrC;
};

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
class MatVecCUSPARSEHandle_t {
public:
  using TA = typename TensorTypeA::value_type;
  using TB = typename TensorTypeB::value_type;
  using TC = typename TensorTypeC::value_type;

  // Mixed-precision compute type.
  using TCOMP = std::conditional_t<
    std::is_same_v<TC, matx::matxFp16> ||
    std::is_same_v<TC, matx::matxBf16>, float, TC>;

  /**
   * Construct a SpMV handle
   */
  MatVecCUSPARSEHandle_t(TensorTypeC &c, const TensorTypeA &a,
                         const TensorTypeB &b, cudaStream_t stream, float alpha,
                         float beta) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    params_ = GetSpMVParams(c, a, b, stream, alpha, beta);

    // Properly typed alpha, beta.
    if constexpr (std::is_same_v<TCOMP, cuda::std::complex<float>> ||
                  std::is_same_v<TCOMP, cuda::std::complex<double>>) {
      salpha_ = {alpha, 0};
      sbeta_ = {beta, 0};
    } else if constexpr (std::is_same_v<TCOMP, float> ||
                         std::is_same_v<TCOMP, double>) {
      salpha_ = alpha;
      sbeta_ = beta;
    } else {
      MATX_THROW(matxNotSupported, "SpMV currently only supports uniform FP");
    }

    [[maybe_unused]] cusparseStatus_t ret = cusparseCreate(&handle_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Create cuSPARSE handle for sparse matrix A.
    static_assert(is_sparse_tensor_v<TensorTypeA>);
    cusparseIndexType_t pt =
        MatXTypeToCuSparseIndexType<typename TensorTypeA::pos_type>();
    cusparseIndexType_t ct =
        MatXTypeToCuSparseIndexType<typename TensorTypeA::crd_type>();
    cusparseIndexBase_t zb = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType dta = MatXTypeToCudaType<TA>();
    if constexpr (TensorTypeA::Format::isCOO()) {
      ret = cusparseCreateCoo(&matA_, params_.m, params_.n, params_.nse,
                              params_.ptrA3, params_.ptrA4, params_.ptrA0, ct,
                              zb, dta);
    } else if constexpr (TensorTypeA::Format::isCSR()) {
      ret = cusparseCreateCsr(&matA_, params_.m, params_.n, params_.nse,
                              params_.ptrA2, params_.ptrA4, params_.ptrA0, pt,
                              ct, zb, dta);
    } else if constexpr (TensorTypeA::Format::isCSC()) {
      ret = cusparseCreateCsc(&matA_, params_.m, params_.n, params_.nse,
                              params_.ptrA2, params_.ptrA4, params_.ptrA0, pt,
                              ct, zb, dta);
    } else {
      MATX_THROW(matxNotSupported, "SpMV currently only supports COO/CSR/CSC");
    }
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Create cuSPARSE handle for dense vectors B and C.
    static_assert(is_tensor_view_v<TensorTypeB>);
    static_assert(is_tensor_view_v<TensorTypeC>);
    cudaDataType dtb = MatXTypeToCudaType<TB>();
    cudaDataType dtc = MatXTypeToCudaType<TC>();
    ret = cusparseCreateDnVec(&vecB_, params_.n, params_.ptrB, dtb);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
    ret = cusparseCreateDnVec(&vecC_, params_.m, params_.ptrC, dtc);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Allocate a workspace for SpMV.
    const cusparseSpMVAlg_t algo = CUSPARSE_SPMV_ALG_DEFAULT;
    const cudaDataType comptp = MatXTypeToCudaType<TCOMP>();
    ret =
        cusparseSpMV_bufferSize(handle_, params_.opA, &salpha_, matA_, vecB_,
                                &sbeta_, vecC_, comptp, algo, &workspaceSize_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
    if (workspaceSize_) {
      matxAlloc((void **)&workspace_, workspaceSize_, MATX_DEVICE_MEMORY);
    }
  }

  ~MatVecCUSPARSEHandle_t() {
    if (workspaceSize_) {
      matxFree(workspace_);
    }
    cusparseDestroy(handle_);
  }

  static detail::MatVecCUSPARSEParams_t
  GetSpMVParams(TensorTypeC &c, const TensorTypeA &a, const TensorTypeB &b,
                cudaStream_t stream, float alpha, float beta) {
    detail::MatVecCUSPARSEParams_t params;
    params.dtype = TypeToInt<typename TensorTypeA::val_type>();
    params.ptype = TypeToInt<typename TensorTypeA::pos_type>();
    params.ctype = TypeToInt<typename TensorTypeA::crd_type>();
    params.stream = stream;
    params.alpha = alpha;
    params.beta = beta;
    // TODO: simple no-batch, row-wise, no-transpose for now
    params.nse = a.Nse();
    params.m = a.Size(TensorTypeA::Rank() - 2);
    params.n = a.Size(TensorTypeA::Rank() - 1);
    params.opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // Matrix handles in cuSPARSE are data specific. Therefore, the pointers
    // to the underlying buffers are part of the SpMV parameters.
    params.ptrA0 = a.Data();
    params.ptrA1 = a.POSData(0);
    params.ptrA2 = a.POSData(1);
    params.ptrA3 = a.CRDData(0);
    params.ptrA4 = a.CRDData(1);
    params.ptrB = b.Data();
    params.ptrC = c.Data();
    return params;
  }

  __MATX_INLINE__ void Exec([[maybe_unused]] TensorTypeC &c,
                            [[maybe_unused]] const TensorTypeA &a,
                            [[maybe_unused]] const TensorTypeB &b) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL);
    const cusparseSpMVAlg_t algo = CUSPARSE_SPMV_ALG_DEFAULT;
    const cudaDataType comptp = MatXTypeToCudaType<TCOMP>();
    [[maybe_unused]] cusparseStatus_t ret =
        cusparseSpMV(handle_, params_.opA, &salpha_, matA_, vecB_, &sbeta_,
                     vecC_, comptp, algo, workspace_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
  }

private:
  cusparseHandle_t handle_ = nullptr; // TODO: share handle globally?
  cusparseSpMatDescr_t matA_ = nullptr;
  cusparseDnVecDescr_t vecB_ = nullptr;
  cusparseDnVecDescr_t vecC_ = nullptr;
  size_t workspaceSize_ = 0;
  void *workspace_ = nullptr;
  detail::MatVecCUSPARSEParams_t params_;
  TCOMP salpha_;
  TCOMP sbeta_;
};

/**
 * Crude hash on SpMV to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common SpMV parameters change.
 */
struct MatVecCUSPARSEParamsKeyHash {
  std::size_t operator()(const MatVecCUSPARSEParams_t &k) const noexcept {
    return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrA0)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrB)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrC)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.stream));
  }
};

/**
 * Test SpMV parameters for equality. Unlike the hash, all parameters must
 * match exactly to ensure the hashed kernel can be reused for the computation.
 */
struct MatVecCUSPARSEParamsKeyEq {
  bool operator()(const MatVecCUSPARSEParams_t &l,
                  const MatVecCUSPARSEParams_t &t) const noexcept {
    return l.dtype == t.dtype && l.ptype == t.ptype && l.ctype == t.ctype &&
           l.stream == t.stream && l.alpha == t.alpha && l.beta == t.beta &&
           l.nse == t.nse && l.m == t.m && l.n == t.n && l.opA == t.opA &&
           l.ptrA0 == t.ptrA0 && l.ptrA1 == t.ptrA1 && l.ptrA2 == t.ptrA2 &&
           l.ptrA3 == t.ptrA3 && l.ptrA4 == t.ptrA4 && l.ptrB == t.ptrB &&
           l.ptrC == t.ptrC;
  }
};

using spmv_cusparse_cache_t =
    std::unordered_map<MatVecCUSPARSEParams_t, std::any,
                       MatVecCUSPARSEParamsKeyHash, MatVecCUSPARSEParamsKeyEq>;

template <typename Op>
__MATX_INLINE__ auto getCuSparseSpMVSupportedTensor(const Op &in,
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
void sparse_matvec_impl(TensorTypeC &C, const TensorTypeA &a,
                        const TensorTypeB &B, const cudaExecutor &exec,
                        float alpha = 1.0, float beta = 0.0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  // Transform into supported form.
  auto b = getCuSparseSpMVSupportedTensor(B, stream);
  auto c = getCuSparseSpMVSupportedTensor(C, stream);
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
  static_assert(RANKA == 2 && RANKB == 1 && RANKC == 1,
                "tensors must have SpMV rank");
  static_assert(std::is_same_v<TC, TA> && std::is_same_v<TC, TB>,
                "tensors must have the same data type");
  static_assert(std::is_same_v<TC, matx::matxFp16> ||
                std::is_same_v<TC, matx::matxBf16> ||
                std::is_same_v<TC, float> ||
                std::is_same_v<TC, double> ||
                std::is_same_v<TC, cuda::std::complex<float>> ||
                std::is_same_v<TC, cuda::std::complex<double>>,
                "unsupported data type");
  MATX_ASSERT(a.Size(RANKA - 1) == b.Size(RANKB - 1) &&
                  a.Size(RANKA - 2) == c.Size(RANKC - 1),
              matxInvalidSize);
  MATX_ASSERT(b.Stride(RANKB - 1) == 1 && c.Stride(RANKC - 1) == 1,
              matxInvalidParameter);

  // Get parameters required by these tensors (for caching).
  auto params =
      detail::MatVecCUSPARSEHandle_t<ctype, atype, btype>::GetSpMVParams(
          c, a, b, stream, alpha, beta);

  // Lookup and cache.
  using cache_val_type = detail::MatVecCUSPARSEHandle_t<ctype, atype, btype>;
  detail::GetCache().LookupAndExec<detail::spmv_cusparse_cache_t>(
      detail::GetCacheIdFromType<detail::spmv_cusparse_cache_t>(), params,
      [&]() {
        return std::make_shared<cache_val_type>(c, a, b, stream, alpha, beta);
      },
      [&](std::shared_ptr<cache_val_type> cache_type) {
        cache_type->Exec(c, a, b);
      });

  // Copy transformed output back.
  if (!c.isSameView(C)) {
    (C = c).run(stream);
  }
}

} // end namespace matx
