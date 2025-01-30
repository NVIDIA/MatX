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

#include <cudss.h>

#include <numeric>

#include "matx/core/cache.h"
#include "matx/core/sparse_tensor.h"
#include "matx/core/tensor.h"

namespace matx {

namespace detail {

/**
 * Parameters needed to execute a cuDSS direct SOLVE.
 */
struct SolveCUDSSParams_t {
  MatXDataType_t dtype;
  MatXDataType_t ptype;
  MatXDataType_t ctype;
  int rank;
  cudaStream_t stream;
  index_t nse;
  index_t m;
  index_t n;
  index_t k;
  // Matrix handles in cuDSS are data specific (unlike e.g. cuBLAS
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
class SolveCUDSSHandle_t {
public:
  using TA = typename TensorTypeA::value_type;
  using TB = typename TensorTypeB::value_type;
  using TC = typename TensorTypeC::value_type;

  static constexpr int RANKA = TensorTypeC::Rank();
  static constexpr int RANKB = TensorTypeC::Rank();
  static constexpr int RANKC = TensorTypeC::Rank();

  SolveCUDSSHandle_t(TensorTypeC &c, const TensorTypeA &a, const TensorTypeB &b,
                     cudaStream_t stream) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    static_assert(RANKA == 2);
    static_assert(RANKB == 2);
    static_assert(RANKC == 2);

    MATX_ASSERT(a.Size(RANKA - 1) == b.Size(RANKB - 2), matxInvalidSize);
    MATX_ASSERT(c.Size(RANKC - 1) == b.Size(RANKB - 1), matxInvalidSize);
    MATX_ASSERT(c.Size(RANKC - 2) == a.Size(RANKA - 2), matxInvalidSize);

    params_ = GetSolveParams(c, a, b, stream);

    [[maybe_unused]] cudssStatus_t ret = cudssCreate(&handle_);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);

    // Create cuDSS handle for sparse matrix A.
    static_assert(is_sparse_tensor_v<TensorTypeA>);
    MATX_ASSERT(TypeToInt<typename TensorTypeA::pos_type> ==
                    TypeToInt<typename TensorTypeA::crd_type>,
                matxNotSupported);
    cudaDataType itp = MatXTypeToCudaType<typename TensorTypeA::crd_type>();
    cudaDataType dta = MatXTypeToCudaType<TA>();
    cudssMatrixType_t mtp = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mvw = CUDSS_MVIEW_FULL;
    cudssIndexBase_t bas = CUDSS_BASE_ZERO;
    if constexpr (TensorTypeA::Format::isCSR()) {
      ret = cudssMatrixCreateCsr(&matA_, params_.m, params_.k, params_.nse,
                                 /*rowStart=*/params_.ptrA2,
                                 /*rowEnd=*/nullptr, params_.ptrA4,
                                 params_.ptrA0, itp, dta, mtp, mvw, bas);
    } else {
      MATX_THROW(matxNotSupported, "cuDSS currently only supports CSR");
    }
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);

    // Create cuDSS handle for dense matrices B and C.
    static_assert(is_tensor_view_v<TensorTypeA>);
    static_assert(is_tensor_view_v<TensorTypeB>);
    cudaDataType dtb = MatXTypeToCudaType<TB>();
    cudaDataType dtc = MatXTypeToCudaType<TC>();
    cudssLayout_t layout = CUDSS_LAYOUT_COL_MAJOR; // TODO: no ROW
    ret = cudssMatrixCreateDn(&matB_, params_.k, params_.n, /*ld=*/params_.k,
                              params_.ptrB, dtb, layout);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);
    ret = cudssMatrixCreateDn(&matC_, params_.m, params_.n, /*ld=*/params_.m,
                              params_.ptrC, dtc, layout);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);

    // Allocate configuration and data.
    ret = cudssConfigCreate(&config_);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);
    ret = cudssDataCreate(handle_, &data_);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);

    // Set configuration.
    cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
    cudssConfigParam_t par = CUDSS_CONFIG_REORDERING_ALG;
    ret = cudssConfigSet(config_, par, &reorder_alg, sizeof(cudssAlgType_t));
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);
  }

  ~SolveCUDSSHandle_t() {
    cudssConfigDestroy(config_);
    cudssDataDestroy(handle_, data_);
    cudssDestroy(handle_);
  }

  static detail::SolveCUDSSParams_t GetSolveParams(TensorTypeC &c,
                                                   const TensorTypeA &a,
                                                   const TensorTypeB &b,
                                                   cudaStream_t stream) {
    detail::SolveCUDSSParams_t params;
    params.dtype = TypeToInt<typename TensorTypeA::val_type>();
    params.ptype = TypeToInt<typename TensorTypeA::pos_type>();
    params.ctype = TypeToInt<typename TensorTypeA::crd_type>();
    params.rank = c.Rank();
    params.stream = stream;
    // TODO: simple no-batch, row-wise, no-transpose for now
    params.nse = a.Nse();
    params.m = a.Size(TensorTypeA::Rank() - 2);
    params.n = b.Size(TensorTypeB::Rank() - 1);
    params.k = a.Size(TensorTypeB::Rank() - 1);
    // Matrix handles in cuDSS are data specific. Therefore, the pointers
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
    // TODO: provide a way to expose these three different steps
    //       (analysis/factorization/solve) individually to user?
    [[maybe_unused]] cudssStatus_t ret = cudssExecute(
        handle_, CUDSS_PHASE_ANALYSIS, config_, data_, matA_, matC_, matB_);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);
    ret = cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_, data_,
                       matA_, matC_, matB_);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);
    ret = cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, matA_, matC_,
                       matB_);
    MATX_ASSERT(ret == CUDSS_STATUS_SUCCESS, matxSolverError);
  }

private:
  cudssHandle_t handle_ = nullptr; // TODO: share handle globally?
  cudssConfig_t config_ = nullptr;
  cudssData_t data_ = nullptr;
  cudssMatrix_t matA_ = nullptr;
  cudssMatrix_t matB_ = nullptr;
  cudssMatrix_t matC_ = nullptr;
  detail::SolveCUDSSParams_t params_;
};

/**
 * Crude hash on SOLVE to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common SOLVE parameters change.
 */
struct SolveCUDSSParamsKeyHash {
  std::size_t operator()(const SolveCUDSSParams_t &k) const noexcept {
    return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrA0)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrB)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrC)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.stream));
  }
};

/**
 * Test SOLVE parameters for equality. Unlike the hash, all parameters must
 * match exactly to ensure the hashed kernel can be reused for the computation.
 */
struct SolveCUDSSParamsKeyEq {
  bool operator()(const SolveCUDSSParams_t &l,
                  const SolveCUDSSParams_t &t) const noexcept {
    return l.dtype == t.dtype && l.ptype == t.ptype && l.ctype == t.ctype &&
           l.rank == t.rank && l.stream == t.stream && l.nse == t.nse &&
           l.m == t.m && l.n == t.n && l.k == t.k && l.ptrA0 == t.ptrA0 &&
           l.ptrA1 == t.ptrA1 && l.ptrA2 == t.ptrA2 && l.ptrA3 == t.ptrA3 &&
           l.ptrA4 == t.ptrA4 && l.ptrB == t.ptrB && l.ptrC == t.ptrC;
  }
};

using gemm_cudss_cache_t =
    std::unordered_map<SolveCUDSSParams_t, std::any, SolveCUDSSParamsKeyHash,
                       SolveCUDSSParamsKeyEq>;

} // end namespace detail

template <typename Op>
__MATX_INLINE__ auto getCUDSSSupportedTensor(const Op &in,
                                             cudaStream_t stream) {
  const auto support_func = [&in]() {
    if constexpr (is_tensor_view_v<Op>) {
      // TODO: we check for row-major even though we assume
      //       data is actually stored column-major; this is
      //       waiting for row-major support in the cuDSS lib
      return in.Stride(Op::Rank() - 1) == 1;
    } else {
      return true;
    }
  };
  return GetSupportedTensor(in, support_func, MATX_ASYNC_DEVICE_MEMORY, stream);
}

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
void sparse_solve_impl(TensorTypeC C, const TensorTypeA A, const TensorTypeB B,
                       const cudaExecutor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  auto a = A; // always sparse
  auto b = getCUDSSSupportedTensor(B, stream);
  auto c = getCUDSSSupportedTensor(C, stream);

  // TODO: some more checking, supported type? on device? etc.

  typedef decltype(c) ctype;
  typedef decltype(a) atype;
  typedef decltype(b) btype;

  // Get parameters required by these tensors (for caching).
  auto params = detail::SolveCUDSSHandle_t<ctype, atype, btype>::GetSolveParams(
      c, a, b, stream);

  // Lookup and cache.
  using cache_val_type = detail::SolveCUDSSHandle_t<ctype, atype, btype>;
  detail::GetCache().LookupAndExec<detail::gemm_cudss_cache_t>(
      detail::GetCacheIdFromType<detail::gemm_cudss_cache_t>(), params,
      [&]() { return std::make_shared<cache_val_type>(c, a, b, stream); },
      [&](std::shared_ptr<cache_val_type> cache_type) {
        cache_type->Exec(c, a, b);
      });
}

} // end namespace matx
