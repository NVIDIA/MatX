////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#include <cusparse.h>

#include <cstdio>
#include <numeric>

#include "matx/core/cache.h"
#include "matx/core/sparse_tensor.h"
#include "matx/core/tensor.h"
#include "matx/transforms/matmul/matmul_common.h"

namespace matx {

namespace detail {

// Translate MatXType for indices to cuSPARSE index type.
template <typename T>
constexpr cusparseIndexType_t MatXTypeToCuSparseIndexType() {
  if constexpr (std::is_same_v<T, uint16_t>) {
    return CUSPARSE_INDEX_16U;
  }
  if constexpr (std::is_same_v<T, int32_t>) {
    return CUSPARSE_INDEX_32I;
  }
  if constexpr (std::is_same_v<T, int64_t>) {
    return CUSPARSE_INDEX_64I;
  }
  if constexpr (std::is_same_v<T, index_t>) {
    return CUSPARSE_INDEX_64I;
  }
}

/**
 * Parameters needed to execute a cuSPARSE GEMM.
 */
struct MatMulCUSPARSEParams_t {
  MatXDataType_t dtype;
  MatXDataType_t ptype;
  MatXDataType_t ctype;
  int rank;
  cudaStream_t stream;
  float alpha;
  float beta;
  index_t nse;
  index_t m;
  index_t n;
  index_t k;
  cusparseOperation_t opA;
  cusparseOperation_t opB;
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
class MatMulCUSPARSEHandle_t {
public:
  using TA = typename TensorTypeA::value_type;
  using TB = typename TensorTypeB::value_type;
  using TC = typename TensorTypeC::value_type;

  static constexpr int RANKA = TensorTypeC::Rank();
  static constexpr int RANKB = TensorTypeC::Rank();
  static constexpr int RANKC = TensorTypeC::Rank();

  /**
   * Construct a sparse GEMM handle
   *   SpMV
   *   SpMM        <- for now
   *   SpGEMM
   *
   */
  MatMulCUSPARSEHandle_t(TensorTypeC &c, const TensorTypeA &a,
                         const TensorTypeB &b, cudaStream_t stream, float alpha,
                         float beta) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    static_assert(RANKA == 2);
    static_assert(RANKB == 2);
    static_assert(RANKC == 2);

    MATX_ASSERT(a.Size(RANKA - 1) == b.Size(RANKB - 2), matxInvalidSize);
    MATX_ASSERT(c.Size(RANKC - 1) == b.Size(RANKB - 1), matxInvalidSize);
    MATX_ASSERT(c.Size(RANKC - 2) == a.Size(RANKA - 2), matxInvalidSize);

    params_ = GetGemmParams(c, a, b, stream, alpha, beta);

    cusparseStatus_t ret = cusparseCreate(&handle_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Create cuSPARSE handle for sparse matrix A.
    static_assert(is_sparse_tensor_v<TensorTypeA>);
    void *val_a = a.Data();
    cusparseIndexType_t pt =
        MatXTypeToCuSparseIndexType<typename TensorTypeA::pos_type>();
    cusparseIndexType_t ct =
        MatXTypeToCuSparseIndexType<typename TensorTypeA::crd_type>();
    cusparseIndexBase_t zb = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType dta = MatXTypeToCudaType<TA>();
    if constexpr (TensorTypeA::Format::isCOO()) {
      void *crd_i = a.CRDData(0);
      void *crd_j = a.CRDData(1);
      ret = cusparseCreateCoo(&matA_, params_.m, params_.k, params_.nse, crd_i,
                              crd_j, val_a, ct, zb, dta);
    } else if constexpr (TensorTypeA::Format::isCSR()) {
      void *pos = a.POSData(1);
      void *crd_j = a.CRDData(1);
      ret = cusparseCreateCsr(&matA_, params_.m, params_.k, params_.nse, pos,
                              crd_j, val_a, pt, ct, zb, dta);
    } else if constexpr (TensorTypeA::Format::isCSC()) {
      void *pos = a.POSData(1);
      void *crd_i = a.CRDData(1);
      ret = cusparseCreateCsc(&matA_, params_.m, params_.k, params_.nse, pos,
                              crd_i, val_a, pt, ct, zb, dta);
    } else {
      MATX_THROW(matxMatMulError, "SpMM currently only supports COO/CSR/CSC");
    }
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Create cuSPARSE handle for dense matrices B and C.
    static_assert(is_tensor_view_v<TensorTypeA>);
    static_assert(is_tensor_view_v<TensorTypeB>);
    void *val_b = b.Data();
    void *val_c = c.Data();
    cudaDataType dtb = MatXTypeToCudaType<TB>();
    cudaDataType dtc = MatXTypeToCudaType<TC>();
    const cusparseOrder_t order = CUSPARSE_ORDER_ROW; // TODO: support col B,C?
    ret = cusparseCreateDnMat(&matB_, params_.k, params_.n, /*ld=*/params_.n,
                              val_b, dtb, order);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
    ret = cusparseCreateDnMat(&matC_, params_.m, params_.n, /*ld=*/params_.n,
                              val_c, dtc, order);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Allocate a workspace for SpMM.
    const cusparseSpMMAlg_t algo = CUSPARSE_SPMM_ALG_DEFAULT;
    const cudaDataType comptp = dtc; // TODO: support separate comp type?!
    ret = cusparseSpMM_bufferSize(handle_, params_.opA, params_.opB,
                                  &params_.alpha, matA_, matB_, &params_.beta,
                                  matC_, comptp, algo, &workspaceSize_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
    if (workspaceSize_) {
      matxAlloc((void **)&workspace_, workspaceSize_, MATX_DEVICE_MEMORY);
    }
  }

  ~MatMulCUSPARSEHandle_t() {
    if (workspaceSize_) {
      matxFree(workspace_);
    }
    cusparseDestroy(handle_);
  }

  static detail::MatMulCUSPARSEParams_t
  GetGemmParams(TensorTypeC &c, const TensorTypeA &a, const TensorTypeB &b,
                cudaStream_t stream, float alpha, float beta) {
    detail::MatMulCUSPARSEParams_t params;
    params.dtype = TypeToInt<typename TensorTypeA::val_type>();
    params.ptype = TypeToInt<typename TensorTypeA::pos_type>();
    params.ctype = TypeToInt<typename TensorTypeA::crd_type>();
    params.rank = c.Rank();
    params.stream = stream;
    params.alpha = alpha;
    params.beta = beta;
    // TODO: simple no-batch, row-wise, no-transpose for now
    params.nse = a.Nse();
    params.m = a.Size(TensorTypeA::Rank() - 2);
    params.n = b.Size(TensorTypeB::Rank() - 1);
    params.k = a.Size(TensorTypeB::Rank() - 1);
    params.opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    params.opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    // Matrix handles in cuSPARSE are data specific. Therefore, the pointers
    // to the underlying buffers are part of the GEMM parameters.
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
    const cusparseSpMMAlg_t algo = CUSPARSE_SPMM_ALG_DEFAULT;
    const cudaDataType comptp = MatXTypeToCudaType<TC>(); // TODO: see above
    cusparseStatus_t ret =
        cusparseSpMM(handle_, params_.opA, params_.opB, &params_.alpha, matA_,
                     matB_, &params_.beta, matC_, comptp, algo, workspace_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
  }

private:
  cusparseHandle_t handle_ = nullptr; // TODO: share handle globally?
  cusparseSpMatDescr_t matA_ = nullptr;
  cusparseDnMatDescr_t matB_ = nullptr;
  cusparseDnMatDescr_t matC_ = nullptr;
  size_t workspaceSize_ = 0;
  void *workspace_ = nullptr;
  detail::MatMulCUSPARSEParams_t params_;
};

/**
 * Crude hash on GEMM to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common GEMM parameters change.
 */
struct MatMulCUSPARSEParamsKeyHash {
  std::size_t operator()(const MatMulCUSPARSEParams_t &k) const noexcept {
    return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrA0)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrB)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrC)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.stream));
  }
};

/**
 * Test GEMM parameters for equality. Unlike the hash, all parameters must
 * match exactly to ensure the hashed kernel can be reused for the computation.
 */
struct MatMulCUSPARSEParamsKeyEq {
  bool operator()(const MatMulCUSPARSEParams_t &l,
                  const MatMulCUSPARSEParams_t &t) const noexcept {
    return l.dtype == t.dtype && l.ptype == t.ptype && l.ctype == t.ctype &&
           l.rank == t.rank && l.stream == t.stream && l.alpha == t.alpha &&
           l.beta == t.beta && l.nse == t.nse && l.m == t.m && l.n == t.n &&
           l.k == t.k && l.opA == t.opA && l.opB == t.opB &&
           l.ptrA0 == t.ptrA0 && l.ptrA1 == t.ptrA1 && l.ptrA2 == t.ptrA2 &&
           l.ptrA3 == t.ptrA3 && l.ptrA4 == t.ptrA4 && l.ptrB == t.ptrB &&
           l.ptrC == t.ptrC;
  }
};

using gemm_cusparse_cache_t =
    std::unordered_map<MatMulCUSPARSEParams_t, std::any,
                       MatMulCUSPARSEParamsKeyHash, MatMulCUSPARSEParamsKeyEq>;

} // end namespace detail

template <typename Op>
__MATX_INLINE__ auto getCUSPARSESupportedTensor(const Op &in,
                                                cudaStream_t stream) {
  const auto support_func = [&in]() {
    if constexpr (is_tensor_view_v<Op>) {
      return in.Stride(Op::Rank() - 1) == 1; // TODO: more than row-wise
    } else {
      return true;
    }
  };
  return GetSupportedTensor(in, support_func, MATX_ASYNC_DEVICE_MEMORY, stream);
}

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
void sparse_matmul_impl(TensorTypeC C, const TensorTypeA A, const TensorTypeB B,
                        const cudaExecutor &exec, float alpha = 1.0,
                        float beta = 0.0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  auto a = A; // always sparse
  auto b = getCUSPARSESupportedTensor(B, stream);
  auto c = getCUSPARSESupportedTensor(C, stream);

  typedef decltype(c) ctype;
  typedef decltype(a) atype;
  typedef decltype(b) btype;

  // Get parameters required by these tensors (for caching).
  auto params =
      detail::MatMulCUSPARSEHandle_t<ctype, atype, btype>::GetGemmParams(
          c, a, b, stream, alpha, beta);

  // Lookup and cache.
  using cache_val_type = detail::MatMulCUSPARSEHandle_t<ctype, atype, btype>;
  detail::GetCache().LookupAndExec<detail::gemm_cusparse_cache_t>(
      detail::GetCacheIdFromType<detail::gemm_cusparse_cache_t>(), params,
      [&]() {
        return std::make_shared<cache_val_type>(c, a, b, stream, alpha, beta);
      },
      [&](std::shared_ptr<cache_val_type> cache_type) {
        cache_type->Exec(c, a, b);
      });
}

} // end namespace matx
