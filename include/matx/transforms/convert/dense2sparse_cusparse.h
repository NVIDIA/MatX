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
#include <typeinfo>

#include "matx/core/cache.h"
#include "matx/core/sparse_tensor.h"
#include "matx/core/tensor.h"

namespace matx {

namespace detail {

/**
 * Parameters needed to execute a cuSPARSE dense2sparse.
 */
struct Dense2SparseParams_t {
  MatXDataType_t dtype;
  MatXDataType_t ptype;
  MatXDataType_t ctype;
  cudaStream_t stream;
  index_t m;
  index_t n;
  // Matrix handles in cuSPARSE are data specific (unlike e.g. cuBLAS
  // where the same plan can be shared between different data buffers).
  void *ptrO1;
  void *ptrO2;
  void *ptrA;
  size_t format_hash;  // Hash of the sparse tensor format type
};

// Helper method to construct storage.
template <typename T>
__MATX_INLINE__ static Storage<T> makeDefaultNonOwningStorage(size_t sz,
                                                        matxMemorySpace_t space,
                                                        cudaStream_t stream) {
  T *ptr = nullptr;
  if (sz != 0) {
    matxAlloc(reinterpret_cast<void **>(&ptr), sz * sizeof(T), space, stream);
  }
  return Storage<T>(ptr, sz);
}

template <typename TensorTypeO, typename TensorTypeA>
class Dense2SparseHandle_t {
public:
  using TA = typename TensorTypeA::value_type;
  using TO = typename TensorTypeO::value_type;

  using VAL = typename TensorTypeO::val_type;
  using POS = typename TensorTypeO::pos_type;
  using CRD = typename TensorTypeO::crd_type;

  /**
   * Construct a dense2sparse handle.
   */
  Dense2SparseHandle_t(TensorTypeO &o, const TensorTypeA &a,
                       cudaStream_t stream) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    params_ = GetConvParams(o, a, stream);

    [[maybe_unused]] cusparseStatus_t ret = cusparseCreate(&handle_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    // Create cuSPARSE handle for dense matrix A.
    static_assert(is_tensor_view_v<TensorTypeA>);
    cudaDataType dta = MatXTypeToCudaType<TA>();
    const cusparseOrder_t order = CUSPARSE_ORDER_ROW;
    ret = cusparseCreateDnMat(&matA_, params_.m, params_.n, /*ld=*/params_.n,
                              params_.ptrA, dta, order);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    // Create cuSPARSE handle for sparse matrix O.
    static_assert(is_sparse_tensor_v<TensorTypeO>);
    cusparseIndexType_t pt = MatXTypeToCuSparseIndexType<POS>();
    cusparseIndexType_t ct = MatXTypeToCuSparseIndexType<CRD>();
    cusparseIndexBase_t zb = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType dto = MatXTypeToCudaType<TO>();
    if constexpr (TensorTypeO::Format::isCOO()) {
      ret = cusparseCreateCoo(&matO_, params_.m, params_.n, 0, nullptr, nullptr,
                              nullptr, ct, zb, dto);
    } else if constexpr (TensorTypeO::Format::isCSR()) {
      ret = cusparseCreateCsr(&matO_, params_.m, params_.n, 0, params_.ptrO2,
                              nullptr, nullptr, pt, ct, zb, dto);
    } else if constexpr (TensorTypeO::Format::isCSC()) {
      ret = cusparseCreateCsc(&matO_, params_.m, params_.n, 0, params_.ptrO2,
                              nullptr, nullptr, pt, ct, zb, dto);
    } else {
      MATX_THROW(matxNotSupported,
                 "Dense2Sparse currently only supports COO/CSR/CSC");
    }
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    // Allocate a workspace for dense2sparse.
    const cusparseDenseToSparseAlg_t algo = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    ret = cusparseDenseToSparse_bufferSize(handle_, matA_, matO_, algo,
                                           &workspaceSize_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);
    if (workspaceSize_) {
      matxAlloc((void **)&workspace_, workspaceSize_, MATX_DEVICE_MEMORY,
                stream);
    }
    ret =
        cusparseDenseToSparse_analysis(handle_, matA_, matO_, algo, workspace_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    // Get the number of nonzero elements.
    [[maybe_unused]] int64_t num_rows_tmp, num_cols_tmp, nnz;
    ret = cusparseSpMatGetSize(matO_, &num_rows_tmp, &num_cols_tmp, &nnz);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);

    // Pre-allocate sparse tensor output.
    if constexpr (TensorTypeO::Format::isCOO()) {
      // Since top-level positions is not part of cuSPARSE COO,
      // the nnz is updated explicitly here before allocating
      // the new components of COO.
      POS *pos = reinterpret_cast<POS *>(params_.ptrO1);
      matxMemorySpace_t space = GetPointerKind(pos);
      if (space == MATX_DEVICE_MEMORY || space == MATX_ASYNC_DEVICE_MEMORY) {
        cudaMemcpy(pos + 1, &nnz, sizeof(POS), cudaMemcpyHostToDevice);
      } else {
        pos[1] = static_cast<POS>(nnz);
      }
      o.SetVal(makeDefaultNonOwningStorage<VAL>(nnz, space, stream));
      o.SetCrd(0, makeDefaultNonOwningStorage<CRD>(nnz, space, stream));
      o.SetCrd(1, makeDefaultNonOwningStorage<CRD>(nnz, space, stream));
      o.SetSparseDataImpl();
      ret = cusparseCooSetPointers(matO_, o.CRDData(0), o.CRDData(1), o.Data());
    } else if constexpr (TensorTypeO::Format::isCSR()) {
      matxMemorySpace_t space = GetPointerKind(params_.ptrO2);
      o.SetVal(makeDefaultNonOwningStorage<VAL>(nnz, space, stream));
      o.SetCrd(1, makeDefaultNonOwningStorage<CRD>(nnz, space, stream));
      o.SetSparseDataImpl();
      ret = cusparseCsrSetPointers(matO_, o.POSData(1), o.CRDData(1), o.Data());
    } else if constexpr (TensorTypeO::Format::isCSC()) {
      matxMemorySpace_t space = GetPointerKind(params_.ptrO2);
      o.SetVal(makeDefaultNonOwningStorage<VAL>(nnz, space, stream));
      o.SetCrd(1, makeDefaultNonOwningStorage<CRD>(nnz, space, stream));
      o.SetSparseDataImpl();
      ret = cusparseCscSetPointers(matO_, o.POSData(1), o.CRDData(1), o.Data());
    }
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);
  }

  ~Dense2SparseHandle_t() {
    if (workspaceSize_) {
      matxFree(workspace_, params_.stream);
    }
    cusparseDestroy(handle_);
  }

  static detail::Dense2SparseParams_t
  GetConvParams(TensorTypeO &o, const TensorTypeA &a, cudaStream_t stream) {
    detail::Dense2SparseParams_t params;
    params.dtype = TypeToInt<VAL>();
    params.ptype = TypeToInt<POS>();
    params.ctype = TypeToInt<CRD>();
    params.stream = stream;
    // TODO: simple no-batch, row-wise, no-transpose for now
    params.m = a.Size(TensorTypeA::Rank() - 2);
    params.n = a.Size(TensorTypeA::Rank() - 1);
    // Matrix handles in cuSPARSE are data specific. Therefore, the pointers to
    // the underlying buffers are part of the conversion parameters. In this
    // case, only the position pointers uniquely determine the sparse output,
    // since the value and coordinate data will be re-allocated on execution.
    params.ptrO1 = o.POSData(0);
    params.ptrO2 = o.POSData(1);
    params.ptrA = a.Data();
    // Add format type hash to distinguish between COO/CSR/CSC etc
    params.format_hash = typeid(typename TensorTypeO::Format).hash_code();
    return params;
  }

  __MATX_INLINE__ void Exec([[maybe_unused]] TensorTypeO &o,
                            [[maybe_unused]] const TensorTypeA &a) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL);
    const cusparseDenseToSparseAlg_t algo = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    [[maybe_unused]] cusparseStatus_t ret =
        cusparseDenseToSparse_convert(handle_, matA_, matO_, algo, workspace_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxCudaError);
  }

private:
  cusparseHandle_t handle_ = nullptr; // TODO: share handle globally?
  cusparseDnMatDescr_t matA_ = nullptr;
  cusparseSpMatDescr_t matO_ = nullptr;
  size_t workspaceSize_ = 0;
  void *workspace_ = nullptr;
  detail::Dense2SparseParams_t params_;
};

/**
 * Crude hash on Dense2Sparse to get a reasonably good delta for collisions.
 * This doesn't need to be perfect, but fast enough to not slow down lookups,
 * and different enough so the common conversion parameters change.
 */
struct Dense2SparseParamsKeyHash {
  std::size_t operator()(const Dense2SparseParams_t &k) const noexcept {
    return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrO1)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.ptrA)) +
           std::hash<uint64_t>()(reinterpret_cast<uint64_t>(k.stream)) +
           std::hash<size_t>()(k.format_hash);
  }
};

/**
 * Test Dense2Sparse parameters for equality. Unlike the hash, all parameters
 * must match exactly to ensure the hashed kernel can be reused for the
 * computation.
 */
struct Dense2SparseParamsKeyEq {
  bool operator()(const Dense2SparseParams_t &l,
                  const Dense2SparseParams_t &t) const noexcept {
    return l.dtype == t.dtype && l.ptype == t.ptype && l.ctype == t.ctype &&
           l.stream == t.stream && l.m == t.m && l.n == t.n &&
           l.ptrO1 == t.ptrO1 && l.ptrO2 == t.ptrO2 && l.ptrA == t.ptrA &&
           l.format_hash == t.format_hash;
  }
};

using dense2sparse_cache_t =
    std::unordered_map<Dense2SparseParams_t, std::any,
                       Dense2SparseParamsKeyHash, Dense2SparseParamsKeyEq>;

template <typename Op>
__MATX_INLINE__ auto getD2SSupportedTensor(const Op &in, cudaStream_t stream) {
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

template <typename OutputTensorType, typename InputTensorType>
void dense2sparse_impl(OutputTensorType &o, const InputTensorType &A,
                       const cudaExecutor &exec) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  // Transform into supported form.
  auto a = getD2SSupportedTensor(A, stream);
  if (!is_matx_transform_op<InputTensorType>() && !a.isSameView(A)) {
    (a = A).run(stream);
  }

  using atype = decltype(a);
  using otype = OutputTensorType;

  using TA = typename atype::value_type;
  using TO = typename otype::value_type;

  static constexpr int RANKA = atype::Rank();
  static constexpr int RANKO = otype::Rank();

  // Restrictions.
  static_assert(RANKA == RANKO, "tensors must have same rank");
  static_assert(std::is_same_v<TA, TO>, "tensors must have the same data type");
  static_assert(std::is_same_v<TO, int8_t> ||
                    std::is_same_v<TO, matx::matxFp16> ||
                    std::is_same_v<TO, matx::matxBf16> ||
                    std::is_same_v<TO, float> || std::is_same_v<TO, double> ||
                    std::is_same_v<TO, cuda::std::complex<float>> ||
                    std::is_same_v<TO, cuda::std::complex<double>>,
                "unsupported data type");
  MATX_ASSERT(a.Stride(RANKA - 1) == 1, matxInvalidParameter);

  // Get parameters required by these tensors (for caching).
  auto params =
      detail::Dense2SparseHandle_t<otype, atype>::GetConvParams(o, a, stream);

  // Lookup and cache.
  using cache_val_type = detail::Dense2SparseHandle_t<otype, atype>;
  detail::GetCache().LookupAndExec<detail::dense2sparse_cache_t>(
      detail::GetCacheIdFromType<detail::dense2sparse_cache_t>(), params,
      [&]() { return std::make_shared<cache_val_type>(o, a, stream); },
      [&](std::shared_ptr<cache_val_type> cache_type) {
        cache_type->Exec(o, a);
      },
      exec);
}

} // end namespace matx
