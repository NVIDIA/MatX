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
 * Parameters needed to execute a cuSPARSE sparse2sparse.
 */
struct Sparse2SparseParams_t {
  MatXDataType_t dtype;
  MatXDataType_t ptype;
  MatXDataType_t ctype;
  cudaStream_t stream;
  index_t nse;
  index_t m;
  index_t n;
  // Matrix handles in cuSPARSE are data specific (unlike e.g. cuBLAS
  // where the same plan can be shared between different data buffers).
  void *ptrO1;
  void *ptrA0;
  void *ptrA1;
  void *ptrA2;
  void *ptrA3;
};

// Helper method to wrap pointer/size in new storage.
template <typename T>
__MATX_INLINE__ static auto wrapDefaultNonOwningStorage(T *ptr, size_t sz) {
  raw_pointer_buffer<T, matx_allocator<T>> buf{ptr, sz * sizeof(T),
                                               /*owning=*/false};
  return basic_storage<decltype(buf)>{std::move(buf)};
}

template <typename TensorTypeO, typename TensorTypeA>
class Sparse2SparseHandle_t {
public:
  using TA = typename TensorTypeA::value_type;
  using TO = typename TensorTypeO::value_type;

  using VAL = typename TensorTypeO::val_type;
  using POS = typename TensorTypeO::pos_type;
  using CRD = typename TensorTypeO::crd_type;

  /**
   * Construct a sparse2sparse handle.
   */
  Sparse2SparseHandle_t(TensorTypeO &o, const TensorTypeA &a,
                        cudaStream_t stream) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    params_ = GetConvParams(o, a, stream);

    [[maybe_unused]] cusparseStatus_t ret = cusparseCreate(&handle_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    static_assert(is_sparse_tensor_v<TensorTypeA>);
    static_assert(is_sparse_tensor_v<TensorTypeO>);

    if constexpr (TensorTypeA::Format::isCOO() &&
                  TensorTypeO::Format::isCSR()) {
      // For speed-of-operation, CSR output shamelessly
      // steals the values and j-index buffers from COO.
      VAL *val = reinterpret_cast<VAL *>(params_.ptrA0);
      CRD *crd = reinterpret_cast<CRD *>(params_.ptrA3);
      o.SetVal(wrapDefaultNonOwningStorage<VAL>(val, params_.nse));
      o.SetCrd(1, wrapDefaultNonOwningStorage<CRD>(crd, params_.nse));
      o.SetSparseDataImpl();
    } else {
      MATX_THROW(matxNotSupported,
                 "Sparse2Sparse currently only supports COO2CSR");
    }
  }

  ~Sparse2SparseHandle_t() { cusparseDestroy(handle_); }

  static detail::Sparse2SparseParams_t
  GetConvParams(TensorTypeO &o, const TensorTypeA &a, cudaStream_t stream) {
    detail::Sparse2SparseParams_t params;
    params.dtype = TypeToInt<VAL>();
    params.ptype = TypeToInt<POS>();
    params.ctype = TypeToInt<CRD>();
    params.stream = stream;
    // TODO: simple no-batch, row-wise, no-transpose for now
    params.nse = a.Nse();
    params.m = a.Size(TensorTypeA::Rank() - 2);
    params.n = a.Size(TensorTypeA::Rank() - 1);
    // Matrix handles in cuSPARSE are data specific. Therefore, the pointers to
    // the underlying buffers are part of the conversion parameters. In this
    // case, only the position pointers uniquely determine the sparse output,
    // since the value and coordinate data will be re-allocated on execution.
    params.ptrO1 = o.POSData(1);
    params.ptrA0 = a.Data();
    params.ptrA1 = a.POSData(0);
    params.ptrA2 = a.CRDData(0);
    params.ptrA3 = a.CRDData(1);
    return params;
  }

  __MATX_INLINE__ void Exec([[maybe_unused]] TensorTypeO &o,
                            [[maybe_unused]] const TensorTypeA &a) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL);
    const cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
    // Legacy API takes specific types only.
    CRD *crd = reinterpret_cast<CRD *>(params_.ptrA2);
    POS *pos = reinterpret_cast<POS *>(params_.ptrO1);
    const int nse = static_cast<int>(params_.nse);
    const int m = static_cast<int>(params_.m);
    [[maybe_unused]] cusparseStatus_t ret =
        cusparseXcoo2csr(handle_, crd, nse, m, pos, base);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);
  }

private:
  cusparseHandle_t handle_ = nullptr; // TODO: share handle globally?
  detail::Sparse2SparseParams_t params_;
};

/**
 * Crude hash on Sparse2Sparse to get a reasonably good delta for collisions.
 * This doesn't need to be perfect, but fast enough to not slow down lookups,
 * and different enough so the common conversion parameters change.
 */
struct Sparse2SparseParamsKeyHash {
  std::size_t operator()(const Sparse2SparseParams_t &k) const noexcept {
    return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrO1)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrA0)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.stream));
  }
};

/**
 * Test Sparse2Sparse parameters for equality. Unlike the hash, all parameters
 * must match exactly to ensure the hashed kernel can be reused for the
 * computation.
 */
struct Sparse2SparseParamsKeyEq {
  bool operator()(const Sparse2SparseParams_t &l,
                  const Sparse2SparseParams_t &t) const noexcept {
    return l.dtype == t.dtype && l.ptype == t.ptype && l.ctype == t.ctype &&
           l.stream == t.stream && l.nse == t.nse && l.m == t.m && l.n == t.n &&
           l.ptrO1 == t.ptrO1 && l.ptrA0 == t.ptrA0 && l.ptrA1 == t.ptrA1 &&
           l.ptrA2 == t.ptrA2 && l.ptrA3 == t.ptrA3;
  }
};

using sparse2sparse_cache_t =
    std::unordered_map<Sparse2SparseParams_t, std::any,
                       Sparse2SparseParamsKeyHash, Sparse2SparseParamsKeyEq>;

} // end namespace detail

template <typename OutputTensorType, typename InputTensorType>
void sparse2sparse_impl(OutputTensorType &o, const InputTensorType &a,
                        const cudaExecutor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  using atype = InputTensorType;
  using otype = OutputTensorType;

  using TA = typename atype::value_type;
  using TO = typename otype::value_type;

  static constexpr int RANKA = atype::Rank();
  static constexpr int RANKO = otype::Rank();

  // Restrictions.
  static_assert(RANKA == 2 && RANKO == 2, "tensors must have rank-2");
  static_assert(std::is_same_v<TA, TO>, "tensors must have the same data type");
  static_assert(std::is_same_v<typename atype::crd_type, int32_t> &&
                    std::is_same_v<typename otype::pos_type, int32_t> &&
                    std::is_same_v<typename otype::crd_type, int32_t>,
                "unsupported index type");

  // Get parameters required by these tensors (for caching).
  auto params =
      detail::Sparse2SparseHandle_t<otype, atype>::GetConvParams(o, a, stream);

  // Lookup and cache.
  using cache_val_type = detail::Sparse2SparseHandle_t<otype, atype>;
  detail::GetCache().LookupAndExec<detail::sparse2sparse_cache_t>(
      detail::GetCacheIdFromType<detail::sparse2sparse_cache_t>(), params,
      [&]() { return std::make_shared<cache_val_type>(o, a, stream); },
      [&](std::shared_ptr<cache_val_type> cache_type) {
        cache_type->Exec(o, a);
      });
}

} // end namespace matx
