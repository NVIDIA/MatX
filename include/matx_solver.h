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
#include "cusolverDn.h"
#include "matx_error.h"
#include "matx_tensor.h"
#include <cstdio>
#include <numeric>

namespace matx {
namespace detail {
/**
 * Dense solver base class that all dense solver types inherit common methods
 * and structures from. The dense solvers used in the 64-bit cuSolver API all
 * use host and device workspace, as well as an "info" allocation to point to
 * issues during solving.
 */
class matxDnSolver_t {
public:
  matxDnSolver_t()
  {
    [[maybe_unused]] cusolverStatus_t  ret;
    ret = cusolverDnCreate(&handle);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

    ret = cusolverDnCreateParams(&dn_params);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  matxError_t SetAdvancedOptions(cusolverDnFunction_t function,
                                 cusolverAlgMode_t algo)
  {
    cusolverStatus_t ret = cusolverDnSetAdvOptions(dn_params, function, algo); 
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

    return matxSuccess;
  }

  virtual ~matxDnSolver_t()
  {
    matxFree(d_workspace);
    matxFree(h_workspace);
    matxFree(d_info);
    cusolverDnDestroy(handle);
  }

  template <typename TensorType>
  void SetBatchPointers(TensorType &a)
  {
    if constexpr (TensorType::Rank() == 2) {
      batch_a_ptrs.push_back(&a(0, 0));
    }
    else {
      using shape_type = typename TensorType::desc_type::shape_type;
      int batch_offset = 3;
      std::array<shape_type, TensorType::Rank()> idx{0};
      auto a_shape = a.Shape();
      // Get total number of batches
      size_t total_iter = std::accumulate(a_shape.begin(), a_shape.begin() + TensorType::Rank() - batch_offset, 1, std::multiplies<shape_type>());
      for (size_t iter = 0; iter < total_iter; iter++) {
        auto ap = std::apply([&a](auto... param) { return a.GetPointer(param...); }, idx);
        batch_a_ptrs.push_back(ap);

        // Update all but the last 2 indices
        UpdateIndices<TensorType, shape_type, TensorType::Rank()>(a, idx, batch_offset);
      }
    }

  }

  /**
   * Get a transposed view of a tensor into a user-supplied buffer
   *
   * @param tp
   *   Pointer to pre-allocated memory
   * @param a
   *   Tensor to transpose
   * @param stream
   *   CUDA stream
   */
  template <typename TensorType>
  static inline auto
  TransposeCopy(typename TensorType::scalar_type *tp, const TensorType &a, cudaStream_t stream = 0)
  {
    auto pa = a.PermuteMatrix();
    auto tv = make_tensor(tp, pa.Shape());
    matx::copy(tv, pa, stream);
    return tv;
  }

  template <typename TensorType>
  static inline uint32_t GetNumBatches(const TensorType &a)
  {
    uint32_t cnt = 1;
    for (int i = 3; i <= TensorType::Rank(); i++) {
      cnt *= static_cast<uint32_t>(a.Size(TensorType::Rank() - i));
    }

    return cnt;
  }

  void AllocateWorkspace(size_t batches)
  {
    matxAlloc(&d_workspace, batches * dspace, MATX_DEVICE_MEMORY);
    matxAlloc((void **)&d_info, batches * sizeof(*d_info), MATX_DEVICE_MEMORY);
    matxAlloc(&h_workspace, batches * hspace, MATX_HOST_MEMORY);
  }

  virtual void GetWorkspaceSize(size_t *host, size_t *device) = 0;

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

/**
 * Parameters needed to execute a cholesky factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnCholParams_t {
  int64_t n;
  void *A;
  size_t batch_size;
  cublasFillMode_t uplo;
  MatXDataType_t dtype;
};

template <typename OutputTensor, typename ATensor>
class matxDnCholSolverPlan_t : public matxDnSolver_t {
  static_assert(OutputTensor::Rank() == ATensor::Rank(), "Cholesky input/output tensor ranks must match");
  using T1 = typename OutputTensor::scalar_type;
  static constexpr int RANK = OutputTensor::Rank();

public:
  /**
   * Plan for solving
   * \f$\textbf{A} = \textbf{L} * \textbf{L^{H}}\f$ or \f$\textbf{A} =
   * \textbf{U} * \textbf{U^{H}}\f$ using the Cholesky method
   *
   * Creates a handle for solving the factorization of A = M * M^H of a dense
   * matrix using the Cholesky method, where M is either the upper or lower
   * triangular portion of A. Input matrix A must be a square Hermitian matrix
   * positive-definite where only the upper or lower triangle is used.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param a
   *   Input tensor view
   * @param uplo
   *   Use upper or lower triangle for computation
   *
   */
  matxDnCholSolverPlan_t(const ATensor &a,
                         cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
  {
    static_assert(RANK >= 2);

    params = GetCholParams(a, uplo);
    GetWorkspaceSize(&hspace, &dspace);
    AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize(size_t *host, size_t *device) override
  {
    cusolverStatus_t ret = cusolverDnXpotrf_bufferSize(handle, dn_params, params.uplo,
                                            params.n, MatXTypeToCudaType<T1>(),
                                            params.A, params.n,
                                            MatXTypeToCudaType<T1>(), device,
                                            host);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnCholParams_t GetCholParams(const ATensor &a,
                                      cublasFillMode_t uplo)
  {
    DnCholParams_t params;
    params.batch_size = matxDnSolver_t::GetNumBatches(a);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.uplo = uplo;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutputTensor &out, const ATensor &a,
            cudaStream_t stream, cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
  {
    // Ensure matrix is square
    MATX_ASSERT(a.Size(RANK - 1) == a.Size(RANK - 2), matxInvalidSize);

    // Ensure output size matches input
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT(out.Size(i) == a.Size(i), matxInvalidSize);
    }

    cusolverDnSetStream(handle, stream);

    SetBatchPointers(out);
    if (out.Data() != a.Data()) {
      matx::copy(out, a, stream);
    }

    // At this time cuSolver does not have a batched 64-bit cholesky interface.
    // Change this to use the batched version once available.
    for (size_t i = 0; i < batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXpotrf(
          handle, dn_params, uplo, params.n, MatXTypeToCudaType<T1>(),
          batch_a_ptrs[i], params.n, MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(d_workspace) + i * dspace, dspace,
          reinterpret_cast<uint8_t *>(h_workspace) + i * hspace, hspace,
          d_info + i);

      MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
    }
  }

  /**
   * Cholesky solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnCholSolverPlan_t() {}

private:
  DnCholParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnCholParamsKeyHash {
  std::size_t operator()(const DnCholParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n)) + (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test cholesky parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct DnCholParamsKeyEq {
  bool operator()(const DnCholParams_t &l, const DnCholParams_t &t) const
      noexcept
  {
    return l.n == t.n && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

// Static caches of inverse handles
static matxCache_t<DnCholParams_t, DnCholParamsKeyHash, DnCholParamsKeyEq>
    dnchol_cache;


/***************************************** LU FACTORIZATION
 * *********************************************/

/**
 * Parameters needed to execute an LU factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnLUParams_t {
  int64_t m;
  int64_t n;
  void *A;
  void *piv;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename OutputTensor, typename PivotTensor, typename ATensor>
class matxDnLUSolverPlan_t : public matxDnSolver_t {
  static constexpr int RANK = OutputTensor::Rank();
  using T1 = typename OutputTensor::scalar_type;
  static_assert(RANK-1 == PivotTensor::Rank(), "Pivot tensor rank must be one less than output");
  static_assert(std::is_same_v<typename PivotTensor::scalar_type, int64_t>, "Pivot tensor type must be int64_t");  

public:
  /**
   * Plan for factoring A such that \f$\textbf{P} * \textbf{A} = \textbf{L} *
   * \textbf{U}\f$
   *
   * Creates a handle for factoring matrix A into the format above. Matrix must
   * not be singular.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param piv
   *   Pivot indices
   * @param a
   *   Input tensor view
   *
   */
  matxDnLUSolverPlan_t(PivotTensor &piv,
                       const ATensor &a)
  {
    static_assert(RANK >= 2);

    params = GetLUParams(piv, a);
    GetWorkspaceSize(&hspace, &dspace);
    AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize(size_t *host, size_t *device) override
  {
    cusolverStatus_t ret = cusolverDnXgetrf_bufferSize(handle, dn_params, params.m,
                                            params.n, MatXTypeToCudaType<T1>(),
                                            params.A, params.m,
                                            MatXTypeToCudaType<T1>(), device,
                                            host);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnLUParams_t GetLUParams(PivotTensor &piv,
                                  const ATensor &a) noexcept
  {
    DnLUParams_t params;
    params.batch_size = matxDnSolver_t::GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.piv = piv.Data();
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutputTensor &out, PivotTensor &piv,
            const ATensor &a, const cudaStream_t stream = 0)
  {
    cusolverDnSetStream(handle, stream);
    int info;

    if constexpr (RANK == 2) {
      batch_piv_ptrs.push_back(&piv(0));
    }
    else if constexpr (RANK == 3) {
      for (int i = 0; i < piv.Size(0); i++) {
        batch_piv_ptrs.push_back(&piv(i, 0));
      }
    }
    else {
      for (int i = 0; i < piv.Size(0); i++) {
        for (int j = 0; j < piv.Size(1); j++) {
          batch_piv_ptrs.push_back(&piv(i, j, 0));
        }
      }
    }

    SetBatchPointers(out);

    if (out.Data() != a.Data()) {
      matx::copy(out, a, stream);
    }

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXgetrf(
          handle, dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
          batch_a_ptrs[i], params.m, batch_piv_ptrs[i],
          MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(d_workspace) + i * dspace, dspace,
          reinterpret_cast<uint8_t *>(h_workspace) + i * hspace, hspace,
          d_info + i);

      MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

      // This will block. Figure this out later
      cudaMemcpy(&info, d_info + i, sizeof(info), cudaMemcpyDeviceToHost);
      MATX_ASSERT(info == 0, matxSolverError);
    }
  }

  /**
   * LU solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnLUSolverPlan_t() {}

private:
  std::vector<int64_t *> batch_piv_ptrs;
  DnLUParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnLUParamsKeyHash {
  std::size_t operator()(const DnLUParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test LU parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnLUParamsKeyEq {
  bool operator()(const DnLUParams_t &l, const DnLUParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size &&
           l.dtype == t.dtype;
  }
};

// Static caches of LU handles
static matxCache_t<DnLUParams_t, DnLUParamsKeyHash, DnLUParamsKeyEq> dnlu_cache;


/***************************************** QR FACTORIZATION
 * *********************************************/

/**
 * Parameters needed to execute a QR factorization. We distinguish unique
 * factorizations mostly by the data pointer in A
 */
struct DnQRParams_t {
  int64_t m;
  int64_t n;
  void *A;
  void *tau;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename OutTensor, typename TauTensor, typename ATensor>
class matxDnQRSolverPlan_t : public matxDnSolver_t {
  using T1 = typename OutTensor::scalar_type;
  static constexpr int RANK = OutTensor::Rank();
  static_assert(OutTensor::Rank()-1 == TauTensor::Rank(), "Tau tensor must be one rank less than output tensor");
  static_assert(OutTensor::Rank() == ATensor::Rank(), "Output tensor must match A tensor rank in SVD");

public:
  /**
   * Plan for factoring A such that \f$\textbf{A} = \textbf{Q} * \textbf{R}\f$
   *
   * Creates a handle for factoring matrix A into the format above. QR
   * decomposition in cuBLAS/cuSolver does not return the Q matrix directly, and
   * it must be computed separately used the Householder reflections in the tau
   * output, along with the overwritten A matrix input. The input and output
   * parameters may be the same tensor. In that case, the input is destroyed and
   * the output is stored in-place.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param tau
   *   Scaling factors for reflections
   * @param a
   *   Input tensor view
   *
   */
  matxDnQRSolverPlan_t(TauTensor &tau,
                       const ATensor &a)
  {
    static_assert(RANK >= 2);

    params = GetQRParams(tau, a);
    GetWorkspaceSize(&hspace, &dspace);
    AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize(size_t *host, size_t *device) override
  {
    cusolverStatus_t ret = cusolverDnXgeqrf_bufferSize(
            handle, dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
            params.A, params.m, MatXTypeToCudaType<T1>(), params.tau,
            MatXTypeToCudaType<T1>(), device, host);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnQRParams_t GetQRParams(TauTensor &tau,
                                  const ATensor &a)
  {
    DnQRParams_t params;

    params.batch_size = matxDnSolver_t::GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.tau = tau.Data();
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutTensor &out, TauTensor &tau,
            const ATensor &a, cudaStream_t stream = 0)
  {
    // Ensure output size matches input
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT(out.Size(i) == a.Size(i), matxInvalidSize);
    }

    SetBatchPointers(out);

    if constexpr (RANK == 2) {
      batch_tau_ptrs.push_back(&tau(0));
    }
    else if constexpr (RANK == 3) {
      for (int i = 0; i < tau.Size(0); i++) {
        batch_tau_ptrs.push_back(&tau(i, 0));
      }
    }
    else {
      for (int i = 0; i < tau.Size(0); i++) {
        for (int j = 0; j < tau.Size(1); j++) {
          batch_tau_ptrs.push_back(&tau(i, j, 0));
        }
      }
    }

    if (out.Data() != a.Data()) {
      matx::copy(out, a, stream);
    }

    cusolverDnSetStream(handle, stream);
    int info;

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXgeqrf(
          handle, dn_params, params.m, params.n, MatXTypeToCudaType<T1>(),
          batch_a_ptrs[i], params.m, MatXTypeToCudaType<T1>(),
          batch_tau_ptrs[i], MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(d_workspace) + i * dspace, dspace,
          reinterpret_cast<uint8_t *>(h_workspace) + i * hspace, hspace,
          d_info + i);

      MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

      // This will block. Figure this out later
      cudaMemcpy(&info, d_info + i, sizeof(info), cudaMemcpyDeviceToHost);
      MATX_ASSERT(info == 0, matxSolverError);
    }
  }

  /**
   * QR solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnQRSolverPlan_t() {}

private:
  std::vector<T1 *> batch_tau_ptrs;
  DnQRParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnQRParamsKeyHash {
  std::size_t operator()(const DnQRParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test QR parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnQRParamsKeyEq {
  bool operator()(const DnQRParams_t &l, const DnQRParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.batch_size == t.batch_size &&
           l.dtype == t.dtype;
  }
};

// Static caches of QR handles
static matxCache_t<DnQRParams_t, DnQRParamsKeyHash, DnQRParamsKeyEq> dnqr_cache;


/********************************************** SVD
 * *********************************************/

/**
 * Parameters needed to execute singular value decomposition. We distinguish
 * unique factorizations mostly by the data pointer in A.
 */
struct DnSVDParams_t {
  int64_t m;
  int64_t n;
  char jobu;
  char jobvt;
  void *A;
  void *U;
  void *V;
  void *S;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename UTensor, typename STensor, typename VTensor, typename ATensor>
class matxDnSVDSolverPlan_t : public matxDnSolver_t {
  using T1 = typename ATensor::scalar_type;
  using T2 = typename UTensor::scalar_type;
  using T3 = typename STensor::scalar_type;
  using T4 = typename VTensor::scalar_type;
  static constexpr int RANK = UTensor::Rank();
  static_assert(UTensor::Rank()-1 == STensor::Rank(), "S tensor must be 1 rank lower than U tensor in SVD");
  static_assert(UTensor::Rank() == ATensor::Rank(), "U tensor must match A tensor rank in SVD");
  static_assert(UTensor::Rank() == VTensor::Rank(), "U tensor must match V tensor rank in SVD");

public:
  /**
   * Plan for factoring A such that \f$\textbf{A} = \textbf{U} * \textbf{\Sigma}
   * * \textbf{V^{H}}\f$
   *
   * Creates a handle for decomposing matrix A into the format above.
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam T2
   *  Data type of U matrix
   * @tparam T3
   *  Data type of S vector
   * @tparam T4
   *  Data type of V matrix
   * @tparam RANK
   *  Rank of A, U, and V matrices, and RANK-1 of S
   *
   * @param u
   *   Output tensor view for U matrix
   * @param s
   *   Output tensor view for S matrix
   * @param v
   *   Output tensor view for V matrix
   * @param a
   *   Input tensor view for A matrix
   * @param jobu
   *   Specifies options for computing all or part of the matrix U: = 'A'. See
   * cuSolver documentation for more info
   * @param jobvt
   *   specifies options for computing all or part of the matrix V**T. See
   * cuSolver documentation for more info
   *
   */
  matxDnSVDSolverPlan_t(UTensor &u,
                        STensor &s,
                        VTensor &v,
                        const ATensor &a, const char jobu = 'A',
                        const char jobvt = 'A')
  {
    static_assert(RANK >= 2);

    T1 *tmp;
    matxAlloc(reinterpret_cast<void **>(&tmp), a.Bytes(), MATX_DEVICE_MEMORY);
    MATX_ASSERT(tmp != nullptr, matxOutOfMemory);

    scratch = make_tensor_p<T1>(tmp, a.Shape());
    params = GetSVDParams(u, s, v, *scratch, jobu, jobvt);

    GetWorkspaceSize(&hspace, &dspace);

    SetBatchPointers(*scratch);
    AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize(size_t *host, size_t *device) override
  {
    cusolverStatus_t ret =
        cusolverDnXgesvd_bufferSize(
            handle, dn_params, params.jobu, params.jobvt, params.m, params.n,
            MatXTypeToCudaType<T1>(), params.A, params.m,
            MatXTypeToCudaType<T3>(), params.S, MatXTypeToCudaType<T2>(),
            params.U, params.m, MatXTypeToCudaType<T4>(), params.V, params.n,
            MatXTypeToCudaType<T1>(), device, host);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnSVDParams_t
  GetSVDParams(UTensor &u, STensor &s,
               VTensor &v, const ATensor &a,
               const char jobu = 'A', const char jobvt = 'A')
  {
    DnSVDParams_t params;
    params.batch_size = matxDnSolver_t::GetNumBatches(a);
    params.m = a.Size(RANK - 2);
    params.n = a.Size(RANK - 1);
    params.A = a.Data();
    params.U = u.Data();
    params.V = v.Data();
    params.S = s.Data();
    params.jobu = jobu;
    params.jobvt = jobvt;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(UTensor &u, STensor &s,
            VTensor &v, const ATensor &a,
            const char jobu = 'A', const char jobvt = 'A',
            cudaStream_t stream = 0)
  {
    if constexpr (RANK == 2) {
      batch_s_ptrs.push_back(&s(0));
      batch_u_ptrs.push_back(&u(0, 0));
      batch_v_ptrs.push_back(&v(0, 0));
    }
    else if constexpr (RANK == 3) {
      for (int i = 0; i < a.Size(0); i++) {
        batch_s_ptrs.push_back(&s(i, 0));
        batch_u_ptrs.push_back(&u(i, 0, 0));
        batch_v_ptrs.push_back(&v(i, 0, 0));
      }
    }
    else {
      for (int i = 0; i < a.Size(0); i++) {
        for (int j = 0; j < a.Size(1); j++) {
          batch_s_ptrs.push_back(&s(i, j, 0));
          batch_u_ptrs.push_back(&u(i, j, 0, 0));
          batch_v_ptrs.push_back(&v(i, j, 0, 0));
        }
      }
    }

    cusolverDnSetStream(handle, stream);
    matx::copy(*scratch, a, stream);
    int info;

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXgesvd(
          handle, dn_params, jobu, jobvt, params.m, params.n,
          MatXTypeToCudaType<T1>(), batch_a_ptrs[i], params.m,
          MatXTypeToCudaType<T3>(), batch_s_ptrs[i], MatXTypeToCudaType<T2>(),
          batch_u_ptrs[i], params.m, MatXTypeToCudaType<T4>(), batch_v_ptrs[i],
          params.n, MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(d_workspace) + i * dspace, dspace,
          reinterpret_cast<uint8_t *>(h_workspace) + i * hspace, hspace,
          d_info + i);

      MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

      // This will block. Figure this out later
      cudaMemcpy(&info, d_info + i, sizeof(info), cudaMemcpyDeviceToHost);
      MATX_ASSERT(info == 0, matxSolverError);
    }
  }

  /**
   * SVD solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnSVDSolverPlan_t() {}

private:
  std::vector<T1 *> batch_s_ptrs;
  std::vector<T1 *> batch_v_ptrs;
  std::vector<T1 *> batch_u_ptrs;
  matx::tensor_t<T1, RANK> *scratch = nullptr;
  DnSVDParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnSVDParamsKeyHash {
  std::size_t operator()(const DnSVDParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.n)) +
           (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test SVD parameters for equality. Unlike the hash, all parameters must match.
 */
struct DnSVDParamsKeyEq {
  bool operator()(const DnSVDParams_t &l, const DnSVDParams_t &t) const noexcept
  {
    return l.n == t.n && l.m == t.m && l.jobu == t.jobu && l.jobvt == t.jobvt &&
           l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

// Static caches of SVD handles
static matxCache_t<DnSVDParams_t, DnSVDParamsKeyHash, DnSVDParamsKeyEq>
    dnsvd_cache;


/*************************************** Eigenvalues and eigenvectors
 * *************************************/

/**
 * Parameters needed to execute singular value decomposition. We distinguish
 * unique factorizations mostly by the data pointer in A.
 */
struct DnEigParams_t {
  int64_t m;
  cusolverEigMode_t jobz;
  cublasFillMode_t uplo;
  void *A;
  void *out;
  void *W;
  size_t batch_size;
  MatXDataType_t dtype;
};

template <typename OutputTensor, typename WTensor, typename ATensor>
class matxDnEigSolverPlan_t : public matxDnSolver_t {
public:
  using T2 = typename WTensor::scalar_type;
  using T1 = typename ATensor::scalar_type;
  static constexpr int RANK = OutputTensor::Rank();
  static_assert(RANK == ATensor::Rank(), "Output and A tensor ranks must match for eigen solver");
  static_assert(RANK-1 == WTensor::Rank(), "W tensor must be one rank lower than output for eigen solver");

  /**
   * Plan computing eigenvalues/vectors on A such that:
   *
   * \f$\textbf{A} * textbf{V} = \textbf{V} * \textbf{\Lambda}\f$
   *
   *
   * @tparam T1
   *  Data type of A matrix
   * @tparam T2
   *  Data type of W matrix
   * @tparam RANK
   *  Rank of A matrix
   *
   * @param w
   *   Eigenvalues of A
   * @param a
   *   Input tensor view
   * @param jobz
   *   CUSOLVER_EIG_MODE_VECTOR to compute eigenvectors or
   * CUSOLVER_EIG_MODE_NOVECTOR to not compute
   * @param uplo
   *   Where to store data in A
   *
   */
  matxDnEigSolverPlan_t(WTensor &w,
                        const ATensor &a,
                        cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR,
                        cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
  {
    static_assert(RANK >= 2);

    params = GetEigParams(w, a, jobz, uplo);
    GetWorkspaceSize(&hspace, &dspace);
    AllocateWorkspace(params.batch_size);
  }

  void GetWorkspaceSize(size_t *host, size_t *device) override
  {
    cusolverStatus_t ret = cusolverDnXsyevd_bufferSize(
                    handle, dn_params, params.jobz, params.uplo, params.m,
                    MatXTypeToCudaType<T1>(), params.A, params.m,
                    MatXTypeToCudaType<T2>(), params.W,
                    MatXTypeToCudaType<T1>(), device,
                    host);
    MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);
  }

  static DnEigParams_t GetEigParams(WTensor &w,
                                    const ATensor &a,
                                    cusolverEigMode_t jobz,
                                    cublasFillMode_t uplo)
  {
    DnEigParams_t params;
    params.batch_size = matxDnSolver_t::GetNumBatches(a);
    params.m = a.Size(RANK - 1);
    params.A = a.Data();
    params.W = w.Data();
    params.jobz = jobz;
    params.uplo = uplo;
    params.dtype = TypeToInt<T1>();

    return params;
  }

  void Exec(OutputTensor &out, WTensor &w,
            const ATensor &a,
            cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR,
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER,
            cudaStream_t stream = 0)
  {
    // Ensure matrix is square
    MATX_ASSERT(a.Size(RANK - 1) == a.Size(RANK - 2), matxInvalidSize);

    // Ensure output size matches input
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT(out.Size(i) == a.Size(i), matxInvalidSize);
    }

    if constexpr (RANK == 2) {
      batch_w_ptrs.push_back(&w(0));
    }
    else if constexpr (RANK == 3) {
      for (int i = 0; i < a.Size(0); i++) {
        batch_w_ptrs.push_back(&w(i, 0));
      }
    }
    else {
      for (int i = 0; i < a.Size(0); i++) {
        for (int j = 0; j < a.Size(1); j++) {
          batch_w_ptrs.push_back(&w(i, j, 0));
        }
      }
    }

    SetBatchPointers(out);

    if (out.Data() != a.Data()) {
      matx::copy(out, a, stream);
    }

    cusolverDnSetStream(handle, stream);
    int info;

    // At this time cuSolver does not have a batched 64-bit LU interface. Change
    // this to use the batched version once available.
    for (size_t i = 0; i < batch_a_ptrs.size(); i++) {
      auto ret = cusolverDnXsyevd(
          handle, dn_params, jobz, uplo, params.m, MatXTypeToCudaType<T1>(),
          batch_a_ptrs[i], params.m, MatXTypeToCudaType<T2>(), batch_w_ptrs[i],
          MatXTypeToCudaType<T1>(),
          reinterpret_cast<uint8_t *>(d_workspace) + i * dspace, dspace,
          reinterpret_cast<uint8_t *>(h_workspace) + i * hspace, hspace,
          d_info + i);

      MATX_ASSERT(ret == CUSOLVER_STATUS_SUCCESS, matxSolverError);

      // This will block. Figure this out later
      cudaMemcpy(&info, d_info + i, sizeof(info), cudaMemcpyDeviceToHost);
      MATX_ASSERT(info == 0, matxSolverError);
    }
  }

  /**
   * Eigen solver handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxDnEigSolverPlan_t() {}

private:
  std::vector<T1 *> batch_w_ptrs;
  DnEigParams_t params;
};

/**
 * Crude hash to get a reasonably good delta for collisions. This doesn't need
 * to be perfect, but fast enough to not slow down lookups, and different enough
 * so the common solver parameters change
 */
struct DnEigParamsKeyHash {
  std::size_t operator()(const DnEigParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.m)) + (std::hash<uint64_t>()(k.batch_size));
  }
};

/**
 * Test Eigen parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct DnEigParamsKeyEq {
  bool operator()(const DnEigParams_t &l, const DnEigParams_t &t) const noexcept
  {
    return l.m == t.m && l.batch_size == t.batch_size && l.dtype == t.dtype;
  }
};

// Static caches of Eig handles
static matxCache_t<DnEigParams_t, DnEigParamsKeyHash, DnEigParamsKeyEq>
    dneig_cache;
}

/**
 * Perform a Cholesky decomposition using a cached plan
 *
 * See documentation of matxDnCholSolverPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
 * library by deducing all parameters needed to perform a Cholesky decomposition
 * from only the matrix A. The input and output parameters may be the same
 * tensor. In that case, the input is destroyed and the output is stored
 * in-place.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor
 * @param a
 *   Input tensor
 * @param stream
 *   CUDA stream
 * @param uplo
 *   Part of matrix to fill
 */
template <typename OutputTensor, typename ATensor>
void chol(OutputTensor &out, const ATensor &a,
          cudaStream_t stream = 0,
          cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
{
  using T1 = typename OutputTensor::scalar_type;

  /* Temporary WAR
     cuSolver doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. Eventually this may be fixed in
     cuSolver.
  */
  T1 *tp;
  matxAlloc(reinterpret_cast<void **>(&tp), a.Bytes(), MATX_ASYNC_DEVICE_MEMORY,
            stream);
  auto tv = detail::matxDnSolver_t::TransposeCopy(tp, a, stream);

  // Get parameters required by these tensors
  auto params = detail::matxDnCholSolverPlan_t<OutputTensor, ATensor>::GetCholParams(tv, uplo);
  params.uplo = uplo;

  // Get cache or new inverse plan if it doesn't exist
  auto ret = detail::dnchol_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxDnCholSolverPlan_t<OutputTensor, ATensor>{tv, uplo};
    detail::dnchol_cache.Insert(params, static_cast<void *>(tmp));
    tmp->Exec(tv, tv, stream, uplo);
  }
  else {
    auto chol_type =
        static_cast<detail::matxDnCholSolverPlan_t<OutputTensor, ATensor> *>(ret.value());
    chol_type->Exec(tv, tv, stream, uplo);
  }

  /* Temporary WAR
   * Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), stream);
}


/**
 * Perform a LU decomposition using a cached plan
 *
 * See documentation of matxDnLUSolverPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
 * library by deducing all parameters needed to perform an LU decomposition from
 * only the matrix A. The input and output parameters may be the same tensor. In
 * that case, the input is destroyed and the output is stored in-place.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor view
 * @param piv
 *   Output of pivot indices
 * @param a
 *   Input matrix A
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename PivotTensor, typename ATensor>
void lu(OutputTensor &out, PivotTensor &piv,
        const ATensor &a, const cudaStream_t stream = 0)
{
  using T1 = typename OutputTensor::scalar_type;

  /* Temporary WAR
     cuSolver doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. Eventually this may be fixed in
     cuSolver.
  */
  T1 *tp;
  matxAlloc(reinterpret_cast<void **>(&tp), a.Bytes(), MATX_ASYNC_DEVICE_MEMORY,
            stream);
  auto tv = detail::matxDnSolver_t::TransposeCopy(tp, a, stream);
  auto tvt = tv.PermuteMatrix();

  // Get parameters required by these tensors
  auto params = detail::matxDnLUSolverPlan_t<OutputTensor, PivotTensor, ATensor>::GetLUParams(piv, tvt);

  // Get cache or new LU plan if it doesn't exist
  auto ret = detail::dnlu_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxDnLUSolverPlan_t<OutputTensor, PivotTensor, ATensor>{piv, tvt};

    detail::dnlu_cache.Insert(params, static_cast<void *>(tmp));
    tmp->Exec(tvt, piv, tvt, stream);
  }
  else {
    auto lu_type = static_cast<detail::matxDnLUSolverPlan_t<OutputTensor, PivotTensor, ATensor> *>(ret.value());
    lu_type->Exec(tvt, piv, tvt, stream);
  }

  /* Temporary WAR
   * Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), stream);
}


/**
 * Compute the determinant of a matrix
 *
 * Computes the terminant of a matrix by first computing the LU composition,
 * then reduces the product of the diagonal elements of U. The input and output
 * parameters may be the same tensor. In that case, the input is destroyed and
 * the output is stored in-place.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor view
 * @param a
 *   Input matrix A
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputTensor>
void det(OutputTensor &out, const InputTensor &a,
         const cudaStream_t stream = 0)
{
  static_assert(OutputTensor::Rank() == InputTensor::Rank() - 2, "Output tensor rank must be 2 less than input for det()");
  constexpr int RANK = InputTensor::Rank();

  // Get parameters required by these tensors
  tensorShape_t<RANK - 1> s;

  // Set batching dimensions of piv
  for (int i = 0; i < RANK - 2; i++) {
    s.SetSize(i, a.Size(i));
  }

  s.SetSize(RANK - 2, std::min(a.Size(RANK - 1), a.Size(RANK - 2)));

  auto piv = make_tensor<int64_t>(s);
  auto ac = make_tensor<typename OutputTensor::scalar_type>(a.Shape());

  lu(ac, piv, a, stream);
  prod(out, diag(ac), stream);
}


/**
 * Perform a QR decomposition using a cached plan
 *
 * See documentation of matxDnQRSolverPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
 * library by deducing all parameters needed to perform a QR decomposition from
 * only the matrix A. The input and output parameters may be the same tensor. In
 * that case, the input is destroyed and the output is stored in-place.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor view
 * @param tau
 *   Output of reflection scalar values
 * @param a
 *   Input tensor A
 * @param stream
 *   CUDA stream
 */
template <typename OutTensor, typename TauTensor, typename ATensor>
void qr(OutTensor &out, TauTensor &tau,
        const ATensor &a, cudaStream_t stream = 0)
{
  using T1 = typename OutTensor::scalar_type;

  /* Temporary WAR
     cuSolver doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. Eventually this may be fixed in
     cuSolver.
  */
  T1 *tp;
  matxAlloc(reinterpret_cast<void **>(&tp), a.Bytes(), MATX_ASYNC_DEVICE_MEMORY,
            stream);
  auto tv = detail::matxDnSolver_t::TransposeCopy(tp, a, stream);
  auto tvt = tv.PermuteMatrix();

  // Get parameters required by these tensors
  auto params = detail::matxDnQRSolverPlan_t<OutTensor, TauTensor, ATensor>::GetQRParams(tau, tvt);

  // Get cache or new QR plan if it doesn't exist
  auto ret = detail::dnqr_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxDnQRSolverPlan_t<OutTensor, TauTensor, ATensor>{tau, tvt};

    detail::dnqr_cache.Insert(params, static_cast<void *>(tmp));
    tmp->Exec(tvt, tau, tvt, stream);
  }
  else {
    auto qr_type = static_cast<detail::matxDnQRSolverPlan_t<OutTensor, TauTensor, ATensor> *>(ret.value());
    qr_type->Exec(tvt, tau, tvt, stream);
  }

  /* Temporary WAR
   * Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), stream);
}


/**
 * Perform a SVD decomposition using a cached plan
 *
 * See documentation of matxDnSVDSolverPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
 * library by deducing all parameters needed to perform a SVD decomposition from
 * only the matrix A.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param u
 *   U matrix output
 * @param s
 *   Sigma matrix output
 * @param v
 *   V matrix output
 * @param a
 *   Input matrix A
 * @param stream
 *   CUDA stream
 * @param jobu
 *   Specifies options for computing all or part of the matrix U: = 'A'. See
 * cuSolver documentation for more info
 * @param jobvt
 *   specifies options for computing all or part of the matrix V**T. See
 * cuSolver documentation for more info
 *
 */
template <typename UTensor, typename STensor, typename VTensor, typename ATensor>
void svd(UTensor &u, STensor &s,
         VTensor &v, const ATensor &a,
         cudaStream_t stream = 0, const char jobu = 'A', const char jobvt = 'A')
{
  using T1 = typename ATensor::scalar_type;
  using T2 = typename UTensor::scalar_type;
  using T3 = typename STensor::scalar_type;
  using T4 = typename VTensor::scalar_type;

  /* Temporary WAR
     cuSolver doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. Eventually this may be fixed in
     cuSolver.
  */
  T1 *tp;
  matxAlloc(reinterpret_cast<void **>(&tp), a.Bytes(), MATX_ASYNC_DEVICE_MEMORY,
            stream);
  auto tv = detail::matxDnSolver_t::TransposeCopy(tp, a, stream);
  auto tvt = tv.PermuteMatrix();

  // Get parameters required by these tensors
  auto params = detail::matxDnSVDSolverPlan_t<UTensor, STensor, VTensor, decltype(tvt)>::GetSVDParams(
      u, s, v, tvt, jobu, jobvt);

  // Get cache or new QR plan if it doesn't exist
  auto ret = detail::dnsvd_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxDnSVDSolverPlan_t{u, s, v, tvt, jobu, jobvt};

    detail::dnsvd_cache.Insert(params, static_cast<void *>(tmp));
    tmp->Exec(u, s, v, tvt, jobu, jobvt, stream);
  }
  else {
    auto svd_type =
        static_cast<detail::matxDnSVDSolverPlan_t<UTensor, STensor, VTensor, decltype(tvt)> *>(ret.value());
    svd_type->Exec(u, s, v, tvt, jobu, jobvt, stream);
  }
}

/**
 * Perform a Eig decomposition using a cached plan
 *
 * See documentation of matxDnEigSolverPlan_t for a description of how the
 * algorithm works. This function provides a simple interface to the cuSolver
 * library by deducing all parameters needed to perform a eigen decomposition
 * from only the matrix A. The input and output parameters may be the same
 * tensor. In that case, the input is destroyed and the output is stored
 * in-place.
 *
 * @tparam T1
 *   Data type of matrix A
 * @tparam RANK
 *   Rank of matrix A
 *
 * @param out
 *   Output tensor view
 * @param w
 *   Eigenvalues output
 * @param a
 *   Input matrix A
 * @param stream
 *   CUDA stream
 * @param jobz
 *   CUSOLVER_EIG_MODE_VECTOR to compute eigenvectors or
 * CUSOLVER_EIG_MODE_NOVECTOR to not compute
 * @param uplo
 *   Where to store data in A
 */
template <typename OutputTensor, typename WTensor, typename ATensor>
void eig([[maybe_unused]] OutputTensor &out, WTensor &w,
         const ATensor &a, cudaStream_t stream = 0,
         cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR,
         cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER)
{
  /* Temporary WAR
     cuSolver doesn't support row-major layouts. Since we want to make the
     library appear as though everything is row-major, we take a performance hit
     to transpose in and out of the function. Eventually this may be fixed in
     cuSolver.
  */
  using T1 = typename OutputTensor::scalar_type;

  T1 *tp;
  matxAlloc(reinterpret_cast<void **>(&tp), a.Bytes(), MATX_ASYNC_DEVICE_MEMORY,
            stream);
  auto tv = detail::matxDnSolver_t::TransposeCopy(tp, a, stream);

  // Get parameters required by these tensors
  auto params =
      detail::matxDnEigSolverPlan_t<OutputTensor, WTensor, ATensor>::GetEigParams(w, tv, jobz, uplo);

  // Get cache or new eigen plan if it doesn't exist
  auto ret = detail::dneig_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxDnEigSolverPlan_t<OutputTensor, WTensor, ATensor>{w, tv, jobz, uplo};

    detail::dneig_cache.Insert(params, static_cast<void *>(tmp));
    tmp->Exec(tv, w, tv, jobz, uplo, stream);
  }
  else {
    auto eig_type =
        static_cast<detail::matxDnEigSolverPlan_t<OutputTensor, WTensor, ATensor> *>(ret.value());
    eig_type->Exec(tv, w, tv, jobz, uplo, stream);
  }

  /* Copy and free async buffer for transpose */
  matx::copy(out, tv.PermuteMatrix(), stream);  
}

} // end namespace matx
