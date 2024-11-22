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

#include "cublas_v2.h"
#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include <cstdio>
#include <numeric>

namespace matx {

/**
 * @brief Algorithm to use for matrix inverse
 *
 */
typedef enum {
  MAT_INVERSE_ALGO_LU,
} MatInverseAlgo_t;



namespace detail {
/**
 * Parameters needed to execute a matrix inverse. Since the matrix inverse
 * reuses internally-allocated memory and overwrites the A array with the
 * factored rows, we distinguish unique inverses mostly by the data pointer in A
 * and A inverse. This ensures that two inverses of the same size, but two
 * different input tensors don't reuse the same plan.
 */
struct InverseParams_t {
  MatInverseAlgo_t algo;
  index_t n;
  void *A;
  void *A_inv;
  MatXDataType_t dtype;
  size_t batch_size;
  cudaStream_t stream;
};

enum class MatInverseLUBackend {
  cuBLASGetRf,
  cuBLASMatInv,
  cuSolverGetRfRs
};

template <typename TensorTypeAInv, typename TensorTypeA, MatInverseAlgo_t ALGO = MAT_INVERSE_ALGO_LU>
class matxInversePlan_t {
  constexpr static int RANK = TensorTypeA::Rank();
  static_assert(RANK == TensorTypeAInv::Rank(), "Input and output tensor ranks must match");
  using T1 = typename TensorTypeAInv::value_type;
  // Linear systems less than or equal to this threshold in size use the cublas*matinvBatched
  // functions. This is one fused kernel rather than two separate kernels and it does not
  // overwrite the input, so in some cases we do not require a temporary work buffer for the input.
  static constexpr int BATCHED_SINGLE_CALL_INV_THRESHOLD = 32;

public:
  /**
   * Construct a matrix inverse handle
   *
   * Creates a handle for executing a matrix inverse. There are several methods
   * of performing a matrix inverse with various tradeoffs, so an algorithm type
   * is supplied to give flexibility. To perform a matrix inversion the input
   * matrix must be square, and non-singular.
   *
   * @tparam T1
   *    Data type of A matrix
   * @tparam RANK
   *    Rank of A matrix
   * @tparam ALGO
   *    Inverse algorithm to use
   *
   * @param a
   *   Input tensor view
   * @param a_inv
   *   Inverse of A (if it exists)
   * @param stream
   *   CUDA stream on which the operation runs
   *
   */
  matxInversePlan_t(TensorTypeAInv &a_inv, const TensorTypeA &a, cudaStream_t stream)
  {
    static_assert(RANK >= 2);

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Ok to remove since we're just passing a list of RO pointers
    //using a_nc = typename std::remove_const<decltype(a)>(a);

    // Ensure matrix is square
    MATX_ASSERT(a.Size(RANK - 1) == a.Size(RANK - 2), matxInvalidSize);

    for (int i = 0; i < a.Rank(); i++) {
      MATX_ASSERT(a.Size(i) == a_inv.Size(i), matxInvalidSize);
    }

    params = GetInverseParams(a_inv, a, stream);

    if constexpr (ALGO == MAT_INVERSE_ALGO_LU) {
      // If we're doing a single batch, use cuSolver since it's faster than cuBLAS. Otherwise if we're operating on a small
      // matrix, use the cuBLAS Inv API. If neither of those works, fall back to the regular cuBlasGetRf path
      if (params.batch_size == 1) {
        backend = MatInverseLUBackend::cuSolverGetRfRs;
      }
      else {
        backend = (a.Size(TensorTypeA::Rank()-1) <= BATCHED_SINGLE_CALL_INV_THRESHOLD) ? 
                          MatInverseLUBackend::cuBLASMatInv : 
                          MatInverseLUBackend::cuBLASGetRf;
      }
            
      const bool use_input_workbuf = UseInputWorkBuffer(a);
      // The cuBLAS getr*Batched LU decomposition functions overwrite the input, so
      // we use a temporary buffer to store the inputs.
      if (use_input_workbuf) {
        make_tensor(a_workbuf, a.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
      }

      if (backend == MatInverseLUBackend::cuSolverGetRfRs) {
        [[maybe_unused]] cusolverStatus_t solver_ret;
        solver_ret = cusolverDnCreate(&cusolver_handle);
        MATX_ASSERT(solver_ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

        solver_ret = cusolverDnCreateParams(&dn_params);
        MATX_ASSERT(solver_ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

        solver_ret = cusolverDnXgetrf_bufferSize(
            cusolver_handle,
            dn_params,
            static_cast<int64_t>(params.n),
            static_cast<int64_t>(params.n),
            MatXTypeToCudaType<typename TensorTypeA::value_type>(),
            a_workbuf.Data(),
            static_cast<int64_t>(params.n),
            MatXTypeToCudaType<typename TensorTypeA::value_type>(),
            &dspace,
            &hspace);

        MATX_ASSERT_STR_EXP(solver_ret, CUSOLVER_STATUS_SUCCESS, matxSolverError, "Error in cusolverDnXgetrf_bufferSize");            

        if (hspace > 0) {
          matxAlloc(&h_workspace, hspace, MATX_HOST_MEMORY);
        }  

        if (dspace > 0) {
          matxAlloc(&d_workspace, dspace, MATX_DEVICE_MEMORY);
        }

        // cuSolver uses a 64-bit pivot, so allocate a large size vs 
        matxAlloc((void **)&d_pivot,
                  a.Size(RANK - 1) * sizeof(int64_t),
                  MATX_ASYNC_DEVICE_MEMORY, stream);
      }  
      else if ( backend == MatInverseLUBackend::cuBLASGetRf || 
                backend == MatInverseLUBackend::cuBLASMatInv) {
        // cuBLAS requires a list of pointers to each matrix. Construct that list
        // here as our batch dims
        std::vector<const T1 *> in_pointers;
        std::vector<T1 *> out_pointers;

        ret = cublasCreate(&cublas_handle);
        MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxInverseError);      

        if constexpr (RANK == 2) {
          if (use_input_workbuf) {
            in_pointers.push_back(&a_workbuf(0, 0));
          } else {
            if constexpr (is_tensor_view_v<TensorTypeA>) {
              // We know this is a tensor view because we are not using a_workbuf
              in_pointers.push_back(&a(0, 0));
            }
          }
          out_pointers.push_back(&a_inv(0, 0));
        }
        else {
          cuda::std::array<index_t, TensorTypeA::Rank()> a_idx{0};
          cuda::std::array<index_t, TensorTypeAInv::Rank()> a_inv_idx{0};
          constexpr int batch_offset = 2;
          auto a_shape = a.Shape();
          // Get total number of batches
          for (size_t iter = 0; iter < params.batch_size; iter++) {
            if (use_input_workbuf) {
              auto ip = cuda::std::apply([&workbuf = a_workbuf](auto... param) { return workbuf.GetPointer(param...); }, a_idx);
              in_pointers.push_back(ip);
              UpdateIndices<decltype(a_workbuf), index_t, TensorTypeA::Rank()>(a_workbuf, a_idx, batch_offset);
            } else {
              if constexpr (is_tensor_view_v<TensorTypeA>) {
                // We know this is a tensor view because we are not using a_workbuf
                auto ip = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, a_idx);
                in_pointers.push_back(ip);
                UpdateIndices<TensorTypeA, index_t, TensorTypeA::Rank()>(a, a_idx, batch_offset);
              }
            }

            auto op = cuda::std::apply([&a_inv](auto... param) { return a_inv.GetPointer(param...); }, a_inv_idx);
            out_pointers.push_back(op);
            UpdateIndices<TensorTypeAInv, index_t, TensorTypeAInv::Rank()>(a_inv, a_inv_idx, batch_offset);
          }
        }

        // Allocate any workspace needed by inverse
        matxAlloc((void **)&d_A_array, in_pointers.size() * sizeof(T1 *),
                  MATX_ASYNC_DEVICE_MEMORY, stream);
        matxAlloc((void **)&d_A_inv_array, out_pointers.size() * sizeof(T1 *),
                  MATX_ASYNC_DEVICE_MEMORY, stream);

        if (backend == MatInverseLUBackend::cuBLASGetRf) {
          // The single function inverse calls do not save the pivots
          matxAlloc((void **)&d_pivot,
                    a.Size(RANK - 1) * in_pointers.size() * sizeof(*d_info),
                    MATX_ASYNC_DEVICE_MEMORY, stream);                 
        }

        cudaMemcpyAsync(d_A_array, in_pointers.data(),
                        in_pointers.size() * sizeof(T1 *), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_A_inv_array, out_pointers.data(),
                        out_pointers.size() * sizeof(T1 *), cudaMemcpyHostToDevice, stream);   
      }   

      matxAlloc((void **)&d_info, params.batch_size * sizeof(*d_info),
                MATX_ASYNC_DEVICE_MEMORY, stream);
      matxAlloc((void **)&h_info, params.batch_size * sizeof(*h_info),
                MATX_HOST_MEMORY, stream);

    }
    else {
      MATX_THROW(matxInvalidType, "Invalid inverse algorithm");
    }
  }

  __MATX_INLINE__ bool UseInputWorkBuffer(const TensorTypeA &a)
  {
    if constexpr (!is_tensor_view_v<TensorTypeA>) {
      return true;
    } else {
      return  backend == MatInverseLUBackend::cuBLASGetRf || 
              backend == MatInverseLUBackend::cuSolverGetRfRs || 
              !a.IsContiguous();
    }
  }

  static InverseParams_t GetInverseParams(TensorTypeAInv &a_inv,
                                          const TensorTypeA &a,
                                          cudaStream_t stream)
  {
    InverseParams_t params;
    if constexpr (is_tensor_view_v<TensorTypeA>) {
      params.A = a.Data();
    } else {
      params.A = nullptr;
    }
    params.A_inv = a_inv.Data();
    params.algo = ALGO;
    params.n = a.Size(RANK - 1);
    params.dtype = TypeToInt<T1>();
    params.stream = stream;

    if constexpr (ALGO == MAT_INVERSE_ALGO_LU) {
      if constexpr (RANK == 2) {
        params.batch_size = 1;
      }
      else {
        params.batch_size = a.TotalSize() / (a.Size(RANK - 1) * a.Size(RANK - 2));
      }
    }

    return params;
  }


  /**
   * Inverse handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxInversePlan_t()
  {
    if (backend == MatInverseLUBackend::cuBLASGetRf || backend == MatInverseLUBackend::cuBLASMatInv) {
      if (d_A_array) { matxFree(d_A_array, cudaStreamDefault); d_A_array = nullptr; }
      if (d_A_inv_array) { matxFree(d_A_inv_array, cudaStreamDefault); d_A_inv_array = nullptr; }
      if (d_pivot) { matxFree(d_pivot, cudaStreamDefault); d_pivot = nullptr; }
      cublasDestroy(cublas_handle);
    }
    else if (backend == MatInverseLUBackend::cuSolverGetRfRs) {
      cusolverDnDestroy(cusolver_handle);
      if (d_workspace) matxFree(d_workspace, cudaStreamDefault);
      if (h_workspace) matxFree(h_workspace, cudaStreamDefault); 
    }

    if (d_info) { matxFree(d_info, cudaStreamDefault); d_info = nullptr; }
    if (h_info) { matxFree(h_info); h_info = nullptr; }    
  }

  /**
   * Execute a matrix inverse
   *
   * Execute a matrix inverse operation on matrix A with the chosen algorithm.
   *
   * @note Views being passed to matxInverse_t must be column-major order for
   * now
   *
   * @tparam T1
   *   Type of matrix A
   * @param a_inv
   *   Output tensor or operator into which the inverse of A is written, if it exists
   * @param a
   *   Input tensor or operator for which the inverse will be computed, if it exists
   * @param stream
   *   CUDA stream
   *
   */
  inline void Exec([[maybe_unused]] TensorTypeAInv &a_inv, const TensorTypeA &a, cudaStream_t stream)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    if (backend == MatInverseLUBackend::cuBLASGetRf || backend == MatInverseLUBackend::cuBLASMatInv) {
      cublasSetStream(cublas_handle, stream);
    }

    if (UseInputWorkBuffer(a)) {
      (a_workbuf = a).run(stream);
    }

    if constexpr (ALGO == MAT_INVERSE_ALGO_LU) {
      if (backend == MatInverseLUBackend::cuBLASGetRf) {
        if constexpr (std::is_same_v<T1, float>) {
          ret =
              cublasSgetrfBatched(cublas_handle, static_cast<int>(params.n), d_A_array, static_cast<int>(params.n), d_pivot,
                                  d_info, static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, double>) {
          ret =
              cublasDgetrfBatched(cublas_handle, static_cast<int>(params.n), d_A_array, static_cast<int>(params.n), d_pivot,
                                  d_info, static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
          ret =
              cublasCgetrfBatched(cublas_handle, static_cast<int>(params.n),
                                  reinterpret_cast<cuComplex *const *>(d_A_array),
                                  static_cast<int>(params.n), d_pivot, d_info,
                                  static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
          ret = cublasZgetrfBatched(
              cublas_handle, static_cast<int>(params.n),
              reinterpret_cast<cuDoubleComplex *const *>(d_A_array),
              static_cast<int>(params.n), d_pivot, d_info,
              static_cast<int>(params.batch_size));
        }

        MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxLUError);

        // Perform LU factorization
        if constexpr (std::is_same_v<T1, float>) {
          ret = cublasSgetriBatched(cublas_handle, static_cast<int>(params.n), d_A_array,
                                    static_cast<int>(params.n), d_pivot,
                                    d_A_inv_array, static_cast<int>(params.n),
                                    d_info, static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, double>) {
          ret = cublasDgetriBatched(cublas_handle, static_cast<int>(params.n), d_A_array,
                                    static_cast<int>(params.n), d_pivot,
                                    d_A_inv_array, static_cast<int>(params.n),
                                    d_info, static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
          ret = cublasCgetriBatched(
              cublas_handle, static_cast<int>(params.n),
              reinterpret_cast<cuComplex *const *>(d_A_array),
              static_cast<int>(params.n), d_pivot,
              reinterpret_cast<cuComplex *const *>(d_A_inv_array),
              static_cast<int>(params.n), d_info,
              static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
          ret = cublasZgetriBatched(
              cublas_handle, static_cast<int>(params.n),
              reinterpret_cast<cuDoubleComplex *const *>(d_A_array),
              static_cast<int>(params.n), d_pivot,
              reinterpret_cast<cuDoubleComplex *const *>(d_A_inv_array),
              static_cast<int>(params.n), d_info,
              static_cast<int>(params.batch_size));
        }

        MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxInverseError);

        cudaMemcpyAsync(h_info, d_info, sizeof(int) * params.batch_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (size_t i = 0; i < params.batch_size; i++) {
          if (h_info[i] != 0) {
            MATX_THROW(matxLUError, "inverse failed");
          }
        }         
      } 
      else if (backend == MatInverseLUBackend::cuBLASMatInv) {
        // For linear systems of size <= BATCHED_SINGLE_CALL_INV_THRESHOLD, we can use the more
        // efficient single call inverse functions.
        if constexpr (std::is_same_v<T1, float>) {
          ret = cublasSmatinvBatched(cublas_handle, static_cast<int>(params.n), d_A_array,
                                    static_cast<int>(params.n),
                                    d_A_inv_array, static_cast<int>(params.n),
                                    d_info, static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, double>) {
          ret = cublasDmatinvBatched(cublas_handle, static_cast<int>(params.n), d_A_array,
                                    static_cast<int>(params.n),
                                    d_A_inv_array, static_cast<int>(params.n),
                                    d_info, static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
          ret = cublasCmatinvBatched(cublas_handle, static_cast<int>(params.n),
                                    reinterpret_cast<cuComplex *const *>(d_A_array),
                                    static_cast<int>(params.n),
                                    reinterpret_cast<cuComplex *const *>(d_A_inv_array),
                                    static_cast<int>(params.n),
                                    d_info, static_cast<int>(params.batch_size));
        }
        else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
          ret = cublasZmatinvBatched(cublas_handle, static_cast<int>(params.n),
                                    reinterpret_cast<cuDoubleComplex *const *>(d_A_array),
                                    static_cast<int>(params.n),
                                    reinterpret_cast<cuDoubleComplex *const *>(d_A_inv_array),
                                    static_cast<int>(params.n),
                                    d_info, static_cast<int>(params.batch_size));
        }
        MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxInverseError);

        cudaMemcpyAsync(h_info, d_info, sizeof(int) * params.batch_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (size_t i = 0; i < params.batch_size; i++) {
          if (h_info[i] != 0) {
            MATX_THROW(matxLUError, "inverse failed");
          }
        }         
      }
      else if (backend == MatInverseLUBackend::cuSolverGetRfRs) {
        MATX_ASSERT_STR(params.batch_size == 1, matxInvalidParameter, "cuSolverGetRfRs backend only used for single batches");

        // cuSolver has a bug that requires this workspace to be zeroed each time
        cudaMemsetAsync(d_workspace, 0, this->dspace, stream);
        
        [[maybe_unused]] cusolverStatus_t solver_ret;
        solver_ret = cusolverDnXgetrf(
            cusolver_handle,
            dn_params,
            static_cast<int64_t>(params.n),
            static_cast<int64_t>(params.n),
            MatXTypeToCudaType<typename TensorTypeA::value_type>(),
            a_workbuf.Data(),
            static_cast<int64_t>(params.n),
            reinterpret_cast<int64_t*>(d_pivot),
            MatXTypeToCudaType<typename TensorTypeA::value_type>(),
            d_workspace,
            dspace,
            h_workspace,
            hspace, 
            d_info);            
        MATX_ASSERT_STR_EXP(solver_ret, CUSOLVER_STATUS_SUCCESS, matxSolverError, "Error in cusolverDnXgetrf");

        cudaMemcpyAsync(h_info, d_info, sizeof(int) * params.batch_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (size_t i = 0; i < params.batch_size; i++) {
          if (h_info[i] != 0) {
            MATX_THROW(matxLUError, "inverse failed");
          }
        }

        // We're Solving Ax = b, so setting "b" to the identity matrix will give us A^-1
        (a_inv = eye<typename TensorTypeA::value_type, 2>({params.n, params.n})).run(stream);

        solver_ret = cusolverDnXgetrs(
            cusolver_handle,
            dn_params,
            CUBLAS_OP_N,
            static_cast<int64_t>(params.n),
            static_cast<int64_t>(params.n),
            MatXTypeToCudaType<typename TensorTypeA::value_type>(),
            a_workbuf.Data(),
            static_cast<int64_t>(params.n),
            reinterpret_cast<int64_t*>(d_pivot),
            MatXTypeToCudaType<typename TensorTypeA::value_type>(),
            a_inv.Data(),
            static_cast<int64_t>(params.n),
            d_info);          
            
          MATX_ASSERT_STR_EXP(solver_ret, CUSOLVER_STATUS_SUCCESS, matxSolverError, "Error in cusolverDnXgetrs");          
      }
      else {
        MATX_THROW(matxInvalidParameter, "Invalid LU backend for inv()");
      }
    }
  }

private:
  // Member variables
  cublasStatus_t ret = CUBLAS_STATUS_SUCCESS;
  MatInverseLUBackend backend;

  // cuSolver's getrf is faster than cuBLAS's
  cusolverDnHandle_t cusolver_handle;
  cusolverDnParams_t dn_params;
  size_t hspace = 0;
  size_t dspace = 0;
  void *d_workspace = nullptr;
  void *h_workspace = nullptr;  
  matx::tensor_t<typename TensorTypeA::value_type, TensorTypeA::Rank()> d_B;

  InverseParams_t params;
  cublasHandle_t cublas_handle;
  matx::tensor_t<typename TensorTypeA::value_type, TensorTypeA::Rank()> a_workbuf;
  int *d_pivot { nullptr };
  int *d_info { nullptr };
  int *h_info { nullptr };
  T1 **d_A_array { nullptr };
  T1 **d_A_inv_array { nullptr };
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common inverse parameters change
 */
struct InverseParamsKeyHash {
  std::size_t operator()(const InverseParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n)) + (std::hash<uint64_t>()(k.batch_size)) +
           (std::hash<uint64_t>()((uint64_t)k.A)) +
           (std::hash<uint64_t>()((uint64_t)k.A_inv)) +
           (std::hash<uint64_t>()((uint64_t)k.stream));
  }
};

/**
 * Test inverse parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct InverseParamsKeyEq {
  bool operator()(const InverseParams_t &l, const InverseParams_t &t) const
      noexcept
  {
    return l.n == t.n && l.A == t.A && l.A_inv == t.A_inv &&
           l.stream == t.stream && l.algo == t.algo &&
           l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

using inv_cache_t = std::unordered_map<InverseParams_t, std::any, InverseParamsKeyHash, InverseParamsKeyEq>;


}

/**
 * @brief Perform a matrix inverse
 *
 * @tparam TensorTypeAInv Inverse type
 * @tparam TensorTypeA Input type
 * @tparam ALGO Algorithm to use
 * @param a_inv Inverse tensor
 * @param a Input tensor
 * @param stream CUDA stream
 */
template <typename TensorTypeAInv, typename TensorTypeA, MatInverseAlgo_t ALGO = MAT_INVERSE_ALGO_LU>
void inv_impl(TensorTypeAInv &a_inv, const TensorTypeA &a,
         cudaStream_t stream = 0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  static_assert(TensorTypeAInv::Rank() == TensorTypeA::Rank(), "Input and output ranks must match");
  // Get parameters required by these tensors
  auto params = detail::matxInversePlan_t<TensorTypeAInv, TensorTypeA, ALGO>::GetInverseParams(a_inv, a, stream);

  using cache_val_type = detail::matxInversePlan_t<TensorTypeAInv, TensorTypeA, ALGO>;
  detail::GetCache().LookupAndExec<detail::inv_cache_t>(
    detail::GetCacheIdFromType<detail::inv_cache_t>(),
    params,
    [&]() {
      return std::make_shared<cache_val_type>(a_inv, a, stream);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Exec(a_inv, a, stream);
    }
  );
}


} // end namespace matx
