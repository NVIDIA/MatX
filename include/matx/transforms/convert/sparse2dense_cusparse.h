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

#include <numeric>

#include "matx/core/cache.h"
#include "matx/core/sparse_tensor.h"
#include "matx/core/tensor.h"

namespace matx {

namespace detail {

/**
 * Parameters needed to execute a cuSPARSE sparse2dense.
 */
struct Sparse2DenseParams_t {
  MatXDataType_t dtype;
  MatXDataType_t ptype;
  MatXDataType_t ctype;
  int rank;
  cudaStream_t stream;
  index_t nse;
  index_t m;
  index_t n;
  // Matrix handles in cuSPARSE are data specific (unlike e.g. cuBLAS
  // where the same plan can be shared between different data buffers).
  void *ptrA0;
  void *ptrA1;
  void *ptrA2;
  void *ptrA3;
  void *ptrA4;
  void *ptrO;
};

template <typename TensorTypeO, typename TensorTypeA>
class Sparse2DenseHandle_t {
public:
  using TA = typename TensorTypeA::value_type;
  using TO = typename TensorTypeO::value_type;

  static constexpr int RANKA = TensorTypeA::Rank();
  static constexpr int RANKO = TensorTypeO::Rank();

  /**
   * Construct a sparse2dense handle.
   */
  Sparse2DenseHandle_t(TensorTypeO &o, const TensorTypeA &a,
                       cudaStream_t stream) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    static_assert(RANKA == RANKO);

    params_ = GetConvParams(o, a, stream);

    [[maybe_unused]] cusparseStatus_t ret = cusparseCreate(&handle_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

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
      MATX_THROW(matxNotSupported,
                 "Sparse2Dense currently only supports COO/CSR/CSC");
    }
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    // Create cuSPARSE handle for dense matrix O.
    static_assert(is_tensor_view_v<TensorTypeO>);
    cudaDataType dto = MatXTypeToCudaType<TO>();
    const cusparseOrder_t order = CUSPARSE_ORDER_ROW;
    ret = cusparseCreateDnMat(&matO_, params_.m, params_.n, /*ld=*/params_.n,
                              params_.ptrO, dto, order);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    // Allocate a workspace for sparse2dense.
    const cusparseSparseToDenseAlg_t algo = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
    ret = cusparseSparseToDense_bufferSize(handle_, matA_, matO_, algo,
                                           &workspaceSize_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);
    if (workspaceSize_) {
      matxAlloc((void **)&workspace_, workspaceSize_, MATX_DEVICE_MEMORY);
    }
  }

  ~Sparse2DenseHandle_t() {
    if (workspaceSize_) {
      matxFree(workspace_);
    }
    cusparseDestroy(handle_);
  }

  static detail::Sparse2DenseParams_t
  GetConvParams(TensorTypeO &o, const TensorTypeA &a, cudaStream_t stream) {
    detail::Sparse2DenseParams_t params;
    params.dtype = TypeToInt<typename TensorTypeA::val_type>();
    params.ptype = TypeToInt<typename TensorTypeA::pos_type>();
    params.ctype = TypeToInt<typename TensorTypeA::crd_type>();
    params.rank = a.Rank();
    params.stream = stream;
    // TODO: simple no-batch, row-wise, no-transpose for now
    params.nse = a.Nse();
    params.m = a.Size(TensorTypeA::Rank() - 2);
    params.n = a.Size(TensorTypeA::Rank() - 1);
    // Matrix handles in cuSPARSE are data specific. Therefore, the pointers
    // to the underlying buffers are part of the conversion parameters.
    params.ptrA0 = a.Data();
    params.ptrA1 = a.POSData(0);
    params.ptrA2 = a.POSData(1);
    params.ptrA3 = a.CRDData(0);
    params.ptrA4 = a.CRDData(1);
    params.ptrO = o.Data();
    return params;
  }

  __MATX_INLINE__ void Exec([[maybe_unused]] TensorTypeO &o,
                            [[maybe_unused]] const TensorTypeA &a) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL);
    const cusparseSparseToDenseAlg_t algo = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
    [[maybe_unused]] cusparseStatus_t ret =
        cusparseSparseToDense(handle_, matA_, matO_, algo, workspace_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);
  }

private:
  cusparseHandle_t handle_ = nullptr; // TODO: share handle globally?
  cusparseSpMatDescr_t matA_ = nullptr;
  cusparseDnMatDescr_t matO_ = nullptr;
  size_t workspaceSize_ = 0;
  void *workspace_ = nullptr;
  detail::Sparse2DenseParams_t params_;
};

/**
 * Crude hash on Sparse2Dense to get a reasonably good delta for collisions.
 * This doesn't need to be perfect, but fast enough to not slow down lookups,
 * and different enough so the common conversion parameters change.
 */
struct Sparse2DenseParamsKeyHash {
  std::size_t operator()(const Sparse2DenseParams_t &k) const noexcept {
    return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrA0)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrO)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.stream));
  }
};

/**
 * Test SOLVE parameters for equality. Unlike the hash, all parameters must
 * match exactly to ensure the hashed kernel can be reused for the computation.
 */
struct Sparse2DenseParamsKeyEq {
  bool operator()(const Sparse2DenseParams_t &l,
                  const Sparse2DenseParams_t &t) const noexcept {
    return l.dtype == t.dtype && l.ptype == t.ptype && l.ctype == t.ctype &&
           l.rank == t.rank && l.stream == t.stream && l.nse == t.nse &&
           l.m == t.m && l.n == t.n && l.ptrA0 == t.ptrA0 &&
           l.ptrA1 == t.ptrA1 && l.ptrA2 == t.ptrA2 && l.ptrA3 == t.ptrA3 &&
           l.ptrA4 == t.ptrA4 && l.ptrO == t.ptrO;
  }
};

using sparse2dense_cache_t =
    std::unordered_map<Sparse2DenseParams_t, std::any,
                       Sparse2DenseParamsKeyHash, Sparse2DenseParamsKeyEq>;

} // end namespace detail

template <typename Op>
__MATX_INLINE__ auto getSparse2DenseSupportedTensor(const Op &in,
                                                    cudaStream_t stream) {
  const auto support_func = [&in]() { return true; };
  return GetSupportedTensor(in, support_func, MATX_ASYNC_DEVICE_MEMORY, stream);
}

template <typename OutputTensorType, typename InputTensorType>
void sparse2dense_impl(OutputTensorType O, const InputTensorType A,
                       const cudaExecutor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  auto a = A; // always sparse
  auto o = getSparse2DenseSupportedTensor(O, stream);

  // TODO: some more checking, supported type? on device? etc.

  typedef decltype(o) otype;
  typedef decltype(a) atype;

  // Get parameters required by these tensors (for caching).
  auto params =
      detail::Sparse2DenseHandle_t<otype, atype>::GetConvParams(o, a, stream);

  // Lookup and cache.
  using cache_val_type = detail::Sparse2DenseHandle_t<otype, atype>;
  detail::GetCache().LookupAndExec<detail::sparse2dense_cache_t>(
      detail::GetCacheIdFromType<detail::sparse2dense_cache_t>(), params,
      [&]() { return std::make_shared<cache_val_type>(o, a, stream); },
      [&](std::shared_ptr<cache_val_type> cache_type) {
        cache_type->Exec(o, a);
      });
}

} // end namespace matx
