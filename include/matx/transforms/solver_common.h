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

#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "matx/transforms/solver_status.h"

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

#include "matx/core/operator_options.h"

namespace detail {

/**
 * Check a cuSolver call's return status: a failure throws matxSolverError in
 * every build type.
 */
inline void CheckSolverLibraryStatus(cusolverStatus_t status, const char *routine)
{
  if (status != CUSOLVER_STATUS_SUCCESS) {
    MATX_THROW(matxSolverError,
               std::string(routine) + " failed with cusolverStatus_t " + std::to_string(static_cast<int>(status)));
  }
}

/**
 * Check a cuBLAS call's return status in a solver: a failure throws
 * matxSolverError in every build type.
 */
inline void CheckSolverCublasStatus(cublasStatus_t status, const char *routine)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    MATX_THROW(matxSolverError,
               std::string(routine) + " failed with cublasStatus_t " + std::to_string(static_cast<int>(status)));
  }
}

/**
 * Throw if MatX was built without a CPU solver backend (NVPL, or OpenBLAS with
 * LAPACK), in every build type. Without one a host solver call cannot run, and
 * returning would leave its outputs, and any status, looking valid.
 */
inline void RequireCpuSolver()
{
#if !MATX_EN_CPU_SOLVER
  MATX_THROW(matxInvalidExecutor,
    "Trying to run a host Solver executor but host Solver support is not configured; "
    "build with -DMATX_EN_NVPL=ON, or -DMATX_EN_OPENBLAS=ON with an OpenBLAS that provides LAPACK");
#endif
}

/**
 * Check `info` values for a CUDA solver call without a status: copy
 * `h_info.size()` values from `d_info` to `h_info`, synchronize the stream, and
 * in debug builds assert that each is zero. The copy and synchronization run in
 * every build type. If given, `describe_positive` describes a positive value in
 * the routine's own terms; other values use DescribeSolverInfo().
 */
inline void CheckSolverInfosOnHost(const int *d_info, std::vector<int> &h_info, cudaStream_t stream,
                                   [[maybe_unused]] const char *routine,
                                   [[maybe_unused]] std::string (*describe_positive)(int) = nullptr)
{
  MATX_CUDA_CHECK(cudaMemcpyAsync(h_info.data(), d_info, sizeof(int) * h_info.size(), cudaMemcpyDeviceToHost, stream));
  MATX_CUDA_CHECK(cudaStreamSynchronize(stream));
  for ([[maybe_unused]] const auto &info : h_info) {
    MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
      (std::string(routine) + ": " +
       (info > 0 && describe_positive != nullptr ? describe_positive(info) : DescribeSolverInfo(info))).c_str());
  }
}

/** Describe a positive getrf `info` value: U(info,info) is exactly zero */
inline std::string DescribeSingularU(int info)
{
  return "U is singular: U(" + std::to_string(info) + "," + std::to_string(info) + ") = 0";
}

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
__MATX_INLINE__ auto getSolverSupportedTensor(const Op &in, const Executor &exec, bool force = false) {
  constexpr int RANK = Op::Rank();

  bool supported = !force; // If we're forcing a new tensor just make it unsupported
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
    return make_tensor<typename Op::value_type>(in.Data(), in.Descriptor());
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
    size_t total_iter = std::accumulate(a_shape.begin(), a_shape.begin() + TensorType::Rank() - batch_offset, static_cast<shape_type>(1), std::multiplies<shape_type>());
    for (size_t iter = 0; iter < total_iter; iter++) {
      auto ap = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
      batch_ptrs.push_back(ap);
      UpdateIndices<TensorType, shape_type, TensorType::Rank()>(a, idx, batch_offset);
    }
  }
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

    CheckSolverLibraryStatus(cusolverDnCreate(&handle), "cusolverDnCreate");
    const cusolverStatus_t ret = cusolverDnCreateParams(&dn_params);
    if (ret != CUSOLVER_STATUS_SUCCESS) {
      // The destructor does not run for a constructor that throws
      cusolverDnDestroy(handle);
      CheckSolverLibraryStatus(ret, "cusolverDnCreateParams");
    }
  }

  matxError_t SetAdvancedOptions(cusolverDnFunction_t function,
                                 cusolverAlgMode_t algo)
  {
    CheckSolverLibraryStatus(cusolverDnSetAdvOptions(dn_params, function, algo), "cusolverDnSetAdvOptions");
    return matxSuccess;
  }

  // Also runs when a derived plan's constructor throws, such as from a failed
  // workspace query, before any workspace is allocated
  virtual ~matxDnCUDASolver_t()
  {
    if (d_workspace != nullptr) {
      matxFree(d_workspace, cudaStreamDefault);
    }
    if (h_workspace != nullptr) {
      matxFree(h_workspace, cudaStreamDefault);
    }
    if (d_info != nullptr) {
      matxFree(d_info, cudaStreamDefault);
    }
    cusolverDnDestroyParams(dn_params);
    cusolverDnDestroy(handle);
  }

  void AllocateWorkspace([[maybe_unused]] size_t batches, [[maybe_unused]] bool batched_api, const cudaExecutor &exec, size_t info_per_batch = 1)
  {
    const auto stream = exec.getStream();
    if (batched_api) {
      // Newer cuSolver
      if (dspace > 0) {
        matxAlloc(&d_workspace, dspace, MATX_ASYNC_DEVICE_MEMORY, stream);
      }

      // cuSolver has a bug where the workspace needs to be zeroed before using it when the type is complex.
      // Zero it out for all types for now.
      cudaMemsetAsync(d_workspace, 0, dspace, stream);
      matxAlloc((void **)&d_info, sizeof(*d_info) * batches * info_per_batch, MATX_ASYNC_DEVICE_MEMORY, stream);

      if (hspace > 0) {
        matxAlloc(&h_workspace, hspace, MATX_HOST_MEMORY);
      }
    }
    else {
      if (dspace > 0) {
        matxAlloc(&d_workspace, batches * dspace, MATX_ASYNC_DEVICE_MEMORY, stream);
      }

      matxAlloc((void **)&d_info, batches * info_per_batch * sizeof(*d_info), MATX_ASYNC_DEVICE_MEMORY, stream);

      if (hspace > 0) {
        matxAlloc(&h_workspace, batches * hspace, MATX_HOST_MEMORY);
      }
    }
  }

  virtual void GetWorkspaceSize() = 0;

protected:
  cusolverDnHandle_t handle;
  cusolverDnParams_t dn_params;
  std::vector<void *> batch_a_ptrs;
  int *d_info = nullptr;
  void *d_workspace = nullptr;
  void *h_workspace = nullptr;
  size_t hspace = 0;
  size_t dspace = 0;
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
