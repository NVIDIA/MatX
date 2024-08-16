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

namespace matx {

#ifdef MATX_EN_NVPL
  #ifndef nvpl_scomplex_t
    #define nvpl_scomplex_t cuda::std::complex<float>
    #define nvpl_dcomplex_t cuda::std::complex<double>
  #endif
  #include <nvpl_lapack.h>
  using lapack_int_t = nvpl_int_t;
  #define LAPACK_CALL(fn) NVPL_LAPACK_##fn
#elif defined(MATX_EN_OPENBLAS_LAPACK)
  #ifdef MATX_OPENBLAS_64BITINT
    #define lapack_int int64_t
  #else
    #define lapack_int int32_t
  #endif
  #define lapack_complex_float cuda::std::complex<float>
  #define lapack_complex_double cuda::std::complex<double>
  #include <lapack.h>
  using lapack_int_t = lapack_int;
  #define LAPACK_CALL(fn) LAPACK_##fn
#else
  using lapack_int_t = index_t;
#endif  

/* Parameter enums */

// Which part (lower or upper) of the dense matrix was filled
// and should be used by the function
enum class SolverFillMode {
  UPPER,
  LOWER
};

enum class EigenMode {
  NO_VECTOR, // Only eigenvalues are computed
  VECTOR     // Both eigenvalues and eigenvectors are computed
};

// SVD modes for computing columns of U and rows of VT, which are
// termed jobu and jobvt in LAPACK/cuSolver. The same option is used for
// both jobu and jobvt in MatX.
enum class SVDMode {
  ALL,     // Compute all columns of U and all rows of V^T
           // Equivalent to jobu = jobvt = 'A'
  REDUCED, // Compute only the first min(m,n) columns of U and rows of V^T
           // Equivalent to jobu = jobvt = 'S'
  NONE     // Compute no columns of U or rows of V^T
           // Equivalent to jobu = jobvt = 'N'
};

// Controls the LAPACK driver used for SVD on host.
enum class SVDHostAlgo {
  QR,  // QR based (corresponds to GESVD)
  DC   // Divide and Conquer based (corresponds to GESDD)
};

namespace detail {

__MATX_INLINE__ char SVDModeToChar(SVDMode jobz) {
  switch (jobz) {
    case SVDMode::ALL:
      return 'A';
    case SVDMode::REDUCED:
      return 'S';
    case SVDMode::NONE:
      return 'N';
    default:
      MATX_ASSERT_STR(false, matxInvalidParameter, "Mode for SVD not supported");
      return '\0';
  }
}


template <typename Op, typename Executor>
__MATX_INLINE__ auto getSolverSupportedTensor(const Op &in, const Executor &exec) {
  constexpr int RANK = Op::Rank();

  bool supported = true;
  if constexpr (!(is_tensor_view_v<Op>)) {
    supported = false;
  } else {

    // Need inner dimension(s) to be contiguous
    if (in.Stride(RANK - 1) != (index_t)1) {
      supported = false;
    }

    if constexpr (RANK >= 2) {
      if (in.Stride(RANK - 2) != in.Size(RANK - 1)) {
        supported = false;
      }
    }
  }

  if (supported) {
    return in;
  } else {
    if constexpr (is_cuda_executor_v<Executor>) {
      return make_tensor<typename Op::value_type>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
    } else {
      return make_tensor<typename Op::value_type>(in.Shape(), MATX_HOST_MALLOC_MEMORY);
    }
  }
}


/* Solver utility functions */

enum class BatchType {
  VECTOR = 1,
  MATRIX = 2
};

/**
 * @brief Sets batch pointers for a batched tensor of arbitrary rank.
 * 
 * Clears the given batch pointers vector and then populates it
 * with pointers to the data of the tensor for batched operations.
 * Handles both batched matrices and vectors.
 * 
 * @tparam BTYPE
 *   Whether the input is a batch of matrices or vectors
 * @tparam TensorType
 *   Type of input tensor a
 * @tparam PointerType
 *   Tensor value type
 * 
 * @param a
 *   The tensor for which batch pointers are to be set.
 * @param batch_ptrs
 *   The vector to be filled with pointers
 */
template <BatchType BTYPE, typename TensorType, typename PointerType = typename TensorType::value_type>
__MATX_INLINE__ void SetBatchPointers(const TensorType &a, std::vector<PointerType *> &batch_ptrs)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  batch_ptrs.clear();
  
  if constexpr (BTYPE == BatchType::VECTOR && TensorType::Rank() == 1) {
    // single vector case
    batch_ptrs.push_back(&a(0));
  }
  else if constexpr (BTYPE == BatchType::MATRIX && TensorType::Rank() == 2) {
    // single matrix case
    batch_ptrs.push_back(&a(0, 0));
  }
  else {
    // batched vectors or matrices
    using shape_type = typename TensorType::desc_type::shape_type;
    int batch_offset = static_cast<int>(BTYPE);
    cuda::std::array<shape_type, TensorType::Rank()> idx{0};
    auto a_shape = a.Shape();
    size_t total_iter = std::accumulate(a_shape.begin(), a_shape.begin() + TensorType::Rank() - batch_offset, 1, std::multiplies<shape_type>());
    for (size_t iter = 0; iter < total_iter; iter++) {
      auto ap = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
      batch_ptrs.push_back(ap);
      UpdateIndices<TensorType, shape_type, TensorType::Rank()>(a, idx, batch_offset);
    }
  }
}

template <typename TensorType>
__MATX_INLINE__ uint32_t GetNumBatches(const TensorType &a)
{
  uint32_t cnt = 1;
  for (int i = 3; i <= TensorType::Rank(); i++) {
    cnt *= static_cast<uint32_t>(a.Size(TensorType::Rank() - i));
  }

  return cnt;
}


/**
 * Dense cuSolver base class that all dense cuda solver types inherit common methods
 * and structures from. The dense solvers used in the 64-bit cuSolver API all
 * use host and device workspace, as well as an "info" allocation to point to
 * issues during solving.
 */
class matxDnCUDASolver_t {
public:
  matxDnCUDASolver_t()
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    [[maybe_unused]] cusolverStatus_t  ret;
    ret = cusolverDnCreate(&handle);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

    ret = cusolverDnCreateParams(&dn_params);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  matxError_t SetAdvancedOptions(cusolverDnFunction_t function,
                                 cusolverAlgMode_t algo)
  {
    [[maybe_unused]] cusolverStatus_t ret = cusolverDnSetAdvOptions(dn_params, function, algo);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

    return matxSuccess;
  }

  virtual ~matxDnCUDASolver_t()
  {
    matxFree(d_workspace, cudaStreamDefault);
    matxFree(h_workspace, cudaStreamDefault);
    matxFree(d_info, cudaStreamDefault);
    cusolverDnDestroy(handle);
  }

  void AllocateWorkspace(size_t batches)
  {
    if (dspace > 0) {
      matxAlloc(&d_workspace, batches * dspace, MATX_DEVICE_MEMORY);
    }

    matxAlloc((void **)&d_info, batches * sizeof(*d_info), MATX_DEVICE_MEMORY);

    if (hspace > 0) {
      matxAlloc(&h_workspace, batches * hspace, MATX_HOST_MEMORY);
    }
  }

  virtual void GetWorkspaceSize() = 0;

protected:
  cusolverDnHandle_t handle;
  cusolverDnParams_t dn_params;
  std::vector<void *> batch_a_ptrs;
  int *d_info;
  void *d_workspace = nullptr;
  void *h_workspace = nullptr;
  size_t hspace;
  size_t dspace;
};

#if MATX_EN_CPU_SOLVER
/**
 * Dense LAPACK base class that all dense host solver types inherit common methods
 * and structures from. Depending on the decomposition, it may require different
 * types of workspace arrays.  
 *
 * @tparam ValueType
 *   Input tensor type
 * 
 */
template <typename ValueType>
class matxDnHostSolver_t {
  
public:
  matxDnHostSolver_t()
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  }

  virtual ~matxDnHostSolver_t()
  {
    matxFree(work);
    matxFree(rwork);
    matxFree(iwork);
  }

  void AllocateWorkspace([[maybe_unused]] size_t batches)
  {
    if (lwork > 0) {
      matxAlloc(&work, lwork * sizeof(ValueType), MATX_HOST_MALLOC_MEMORY);
    }

    // used for eig and svd complex types
    if (lrwork > 0) {
      matxAlloc(&rwork, lrwork * sizeof(typename inner_op_type_t<ValueType>::type), MATX_HOST_MALLOC_MEMORY);
    }

    // used for all eig types
    if (liwork > 0) {
      matxAlloc(&iwork, liwork * sizeof(lapack_int_t), MATX_HOST_MALLOC_MEMORY);
    }
  }

  virtual void GetWorkspaceSize() {};

protected:
  std::vector<void *> batch_a_ptrs;
  void *work = nullptr;  // work array of input type
  void *rwork = nullptr; // real valued work array
  void *iwork = nullptr; // integer valued work array
  lapack_int_t lwork = -1;
  lapack_int_t lrwork = -1;
  lapack_int_t liwork = -1;
};
#endif

} // end namespace detail

} // end namespace matx