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

#include <cublasLt.h>

#if MATX_ENABLE_CUTLASS == 1
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#endif

#include <cstdio>
#include <numeric>

#include "cublas_v2.h"
#include "matx/core/cache.h"
#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"

namespace matx {

/**
 * Defines a provider type for a GEMM. The provider is directly tied to the
 * underlying library used for the gemm, and certain providers provide
 * capabilities that others may not have.
 */
typedef enum {
  PROVIDER_TYPE_CUTLASS = 0,  ///< CUTLASS library
  PROVIDER_TYPE_CUBLASLT = 1, ///< cuBLASLt library
  PROVIDER_TYPE_AUTO,         ///< Automatically select

  PROVIDER_TYPE_SENTINEL ///< Sentinel value. Do not use
} MatXMatMulProvider_t;


namespace detail {
typedef enum {
  MEM_ORDER_ROW_MAJOR = 0,
  MEM_ORDER_COL_MAJOR = 1,
} MemOrder_t;

union MatMulScaleType_t {
  float f32;
  double f64;
  float cf32[2];
  double cf64[2];
};

/**
 * Parameters needed to execute a GEMM. For the most part, these are very
 * similar to that of a standard GEMM call
 */
struct MatMulParams_t {
  index_t a_rows = 0;
  index_t a_cols = 0;
  index_t b_rows = 0;
  index_t b_cols = 0;
  index_t c_rows = 0;
  index_t c_cols = 0;
  index_t m = 0;
  index_t n = 0;
  index_t k = 0;
  index_t lda;
  index_t ldb;
  index_t ldc;
  int32_t batch; // Must be int32_t for cuBLASLt
  index_t astride; // batch stride
  index_t bstride; // batch stride
  index_t cstride; // batch stride
  MatXMatMulProvider_t prov;
  cudaStream_t stream;
  MatXDataType_t dtype;
  cublasOperation_t opA;
  cublasOperation_t opB;
};

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, 
          MatXMatMulProvider_t PROV = PROVIDER_TYPE_CUBLASLT>
class matxMatMulHandle_t {
public:
  using T1 = typename TensorTypeC::scalar_type;
  using T2 = typename TensorTypeA::scalar_type;
  using T3 = typename TensorTypeB::scalar_type;
  static constexpr int RANK = TensorTypeC::Rank();
  static_assert(TensorTypeC::Rank() == TensorTypeB::Rank());
  static_assert(TensorTypeC::Rank() == TensorTypeA::Rank());

  /**
   * Construct a GEMM handle
   *
   * Creates a GEMM handle for the view shapes and provider type given. The view
   * shapres are used to create the underlying metadata used for the GEMM, so a
   * handle should only be used for views of identical sizes. The provider
   * chooses the underlying library used to perform the GEMM. Certain providers
   * have more features than others and may perform differently than others. At
   * the moment, it is recommended to try different providers for a given matrix
   * size until the optimal provider is found. Different providers may also be
   * used by creating multiple handles.
   *
   * @tparam T1
   *    Type of C matrix
   * @tparam T2
   *    Type of A matrix
   * @tparam T3
   *    Type of B matrix
   * @tparam PROV
   *    Provider type chosen from MatXMatMulProvider_t type
   *
   * @param c
   *   C matrix view
   * @param a
   *   A matrix view
   * @param b
   *   B matrix view
   *
   */
  matxMatMulHandle_t(TensorTypeC &c, const TensorTypeA &a,
                     const TensorTypeB &b)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    MATX_STATIC_ASSERT_STR((PROV != PROVIDER_TYPE_CUTLASS) || MATX_ENABLE_CUTLASS, matxMatMulError,
                  "Must use -DCUTLASS_DIR in CMake to enable CUTLASS support");
    static_assert(TensorTypeA::Rank() == TensorTypeB::Rank());
    static_assert(TensorTypeA::Rank() == TensorTypeC::Rank());
    static_assert(RANK >= 2);
    MATX_ASSERT(a.Size(RANK - 1) == b.Size(RANK - 2), matxInvalidSize);
    MATX_ASSERT(c.Size(RANK - 1) == b.Size(RANK - 1), matxInvalidSize);
    MATX_ASSERT(c.Size(RANK - 2) == a.Size(RANK - 2), matxInvalidSize);

    // Ensure batch dimensions are equal
    for (int i = 0; i < RANK - 2; i++) {
      MATX_ASSERT(a.Size(i) == b.Size(i), matxInvalidSize);
      MATX_ASSERT(a.Size(i) == c.Size(i), matxInvalidSize);
    }

    // This must come before the things below to properly set class parameters
    params_ = GetGemmParams(c, a, b);

    // // Workspace buffer
    matxAlloc((void **)&workspace, workspaceSize, MATX_DEVICE_MEMORY);

    if constexpr (PROV == PROVIDER_TYPE_CUBLASLT) {
      ConfigureCublasLt();
    }
  }

  template <typename InputType>
  static void SetAlphaBeta([[maybe_unused]] char *const palpha,
                           [[maybe_unused]] char *const pbeta,
                           [[maybe_unused]] float const alpha,
                           [[maybe_unused]] float const beta)
  {
    // For now we don't give much flexibility on compute type/alpha
    if constexpr (std::is_same_v<InputType, cuda::std::complex<float>> ||
                  is_complex_half_v<InputType>) {
      cuComplex *calpha = reinterpret_cast<cuComplex *>(palpha);
      cuComplex *cbeta = reinterpret_cast<cuComplex *>(pbeta);
      *calpha = {alpha, 0};
      *cbeta = {beta, 0};
    }
    else if constexpr (std::is_same_v<InputType, cuda::std::complex<double>>) {
      cuDoubleComplex *dalpha = reinterpret_cast<cuDoubleComplex *>(palpha);
      cuDoubleComplex *dbeta = reinterpret_cast<cuDoubleComplex *>(pbeta);
      *dalpha = {alpha, 0};
      *dbeta = {beta, 0};
    }
    else if constexpr (std::is_same_v<InputType, double>) {
      double *dalpha = reinterpret_cast<double *>(palpha);
      double *dbeta = reinterpret_cast<double *>(pbeta);
      *dalpha = alpha;
      *dbeta = beta;
    }
    else if constexpr (is_matx_half_v<InputType> ||
                       std::is_same_v<InputType, float>) {
      float *talpha = reinterpret_cast<float *>(palpha);
      float *tbeta = reinterpret_cast<float *>(pbeta);
      *talpha = alpha;
      *tbeta = beta;
    }
    else {
      MATX_THROW(matxInvalidType, "Invalid type when deducing alpha/beta");
    }
  }

  static detail::MatMulParams_t GetGemmParams(TensorTypeC &c, const TensorTypeA &a,
                     const TensorTypeB &b)
  {
    /* If a user passes in a tensor where the last two dimensions are transposed we retain
       the original size parameters, but tell the underlying libraries that the tensors are
       in column-major ordering. The exception to this is when a transposed half-precision
       complex type is used. In that case we have to make a temporary copy of the tensor to
       put the data in planar format for the libraries. Since we now use the temporary tensor
       as input to the GEMM, the data is no longer transposed in memory and we simply use
       the same memory layout as a non-transposed real matrix would use.
    */    
    detail::MatMulParams_t params;
    params.dtype = TypeToInt<T1>();
    params.prov = PROV;

    // Batches
    params.batch = 1;
    params.astride = 0;
    params.bstride = 0;
    params.cstride = 0;

    // If we have a 3D or above tensor, the upper dims are batch dimensions. We
    // only batch on the third dimension and loop anything else above;
    if constexpr (RANK >= 3) {
      params.batch = static_cast<int32_t>(a.Size(RANK - 3));
      params.astride = a.Stride(RANK-3);
      params.bstride = b.Stride(RANK-3);
      params.cstride = c.Stride(RANK-3);
    } 

    // If the user wants C transposed (as a permuted view), we need the output
    // matrix to still be MxN in memory. The reason is the permuted view will
    // handle viewing it as an NxM. To accomplish this we use the identity C' =
    // B'A', so we swap A and B and permute them.
    if (c.Stride(RANK - 2) == 1 && c.Size(RANK - 1) != 1) {
      // TODO this looks like repeat logic from what I put in elsewhere...
      // track this down later.   For now adding an assert to see if it ever pops up.
      // If it does not we should delete this code.
      MATX_ASSERT_STR(false, matxInvalidDim, "Internal Matmul error.  This should not be hit\n");
      if constexpr (PROV == PROVIDER_TYPE_CUBLASLT) {
        if (b.Stride(RANK - 2) == 1) {
          params.opA = CUBLAS_OP_N;
          params.a_rows = b.Size(RANK - 1);
          params.a_cols = b.Size(RANK - 2);
          params.lda = b.Stride(RANK - 1);
        }
        else if (b.Stride(RANK - 1) == 1) {
          params.opA = CUBLAS_OP_T;
          params.a_rows = b.Size(RANK - 2);
          params.a_cols = b.Size(RANK - 1);
          params.lda = b.Stride(RANK - 2);
        }

        if (a.Stride(RANK - 2) == 1) {
          params.opB = CUBLAS_OP_N;
          params.b_rows = a.Size(RANK - 1);
          params.b_cols = a.Size(RANK - 2);
          params.ldb = a.Stride(RANK - 1);
        }
        else if (a.Stride(RANK - 1) == 1) {
          params.opB = CUBLAS_OP_T;
          params.b_rows = a.Size(RANK - 2);
          params.b_cols = a.Size(RANK - 1);
          params.ldb = a.Stride(RANK - 2);
        }
          
        params.c_rows = params.a_rows;
        params.c_cols = params.b_cols;
        params.ldc = c.Stride(RANK - 1);
      }
      else if constexpr (PROV == PROVIDER_TYPE_CUTLASS) {
        params.opA = CUBLAS_OP_N;
        params.opB = CUBLAS_OP_N;
        params.m = static_cast<int>(b.Size(RANK - 1));
        params.n = static_cast<int>(a.Size(RANK - 2));
        params.k =
            static_cast<int>(a.Size(RANK - 2)); // Gemm Problem dimensions
        params.lda = static_cast<int>(b.Stride(RANK - 1));
        params.ldb = static_cast<int>(a.Stride(RANK - 1));
        params.ldc = static_cast<int>(c.Stride(RANK - 1));
      }      
    }
    else {
      if constexpr (PROV == PROVIDER_TYPE_CUBLASLT) {
        if constexpr (is_complex_half_v<typename TensorTypeA::scalar_type>) {
          // For half complex we always copy to a new tensor so it is always cublas op N
          params.opA = CUBLAS_OP_N;
        } else if ( a.Stride(RANK-1) > 1 // last stride > 1
                  || (a.Stride(RANK-1) == 1 && a.Stride(RANK-2) == 1 && a.Size(RANK-1) != 1)) { // last strides both equal 1 and size > 1 
          params.opA = CUBLAS_OP_T;
        } else { // otherwise row major
          params.opA = CUBLAS_OP_N;
        }

        if constexpr (is_complex_half_v<typename TensorTypeB::scalar_type>) {
          // For half complex we always copy to a new tensor so it is always cublas op N
          params.opB = CUBLAS_OP_N;
        } else if ( b.Stride(RANK-1) > 1 // last stride > 1
                  || (b.Stride(RANK-1) == 1 && b.Stride(RANK-2) == 1 && b.Size(RANK-1) != 1)) { // last strides both equal 1 and size > 1 
          params.opB = CUBLAS_OP_T;
        } else { // otherwise row major
          params.opB = CUBLAS_OP_N;
        }
        
        params.a_rows = a.Size(RANK - 2);
        params.a_cols = a.Size(RANK - 1);
        params.b_rows = b.Size(RANK - 2);
        params.b_cols = b.Size(RANK - 1);
       
        // set lda/ldb according to transpose modes
        params.ldb = (params.opB == CUBLAS_OP_T) ? b.Stride(RANK - 1) : b.Stride(RANK - 2); 
        params.lda = (params.opA == CUBLAS_OP_T) ? a.Stride(RANK - 1) : a.Stride(RANK - 2);

        // for complex half we have copied to planar row-major
        if (is_complex_half_v<typename TensorTypeB::scalar_type>) {
          params.ldb = b.Size(RANK-1);
        }

        // for complex half we have copied to planar row-major
        if constexpr (is_complex_half_v<typename TensorTypeB::scalar_type>) {
          params.lda = a.Size(RANK-1);
        }

        params.c_rows = params.a_rows;
        params.c_cols = params.b_cols;
        params.ldc = c.Stride(RANK - 2);
       
      }
      else if constexpr (PROV == PROVIDER_TYPE_CUTLASS) {
        params.opA = CUBLAS_OP_N;
        params.opB = CUBLAS_OP_N;
        params.m = static_cast<int>(a.Size(RANK - 2));
        params.n = static_cast<int>(b.Size(RANK - 1));
        params.k =
            static_cast<int>(a.Size(RANK - 1)); // Gemm Problem dimensions
        params.lda = static_cast<int>(a.Stride(RANK - 2));
        params.ldb = static_cast<int>(b.Stride(RANK - 2));
        params.ldc = static_cast<int>(c.Stride(RANK - 2));
      }
    }

    return params;
  }

  /**
   * GEMM handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxMatMulHandle_t()
  {
    matxFree(workspace);

    if constexpr (PROV == PROVIDER_TYPE_CUBLASLT) {
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(Cdesc);
      cublasLtMatrixLayoutDestroy(Bdesc);
      cublasLtMatrixLayoutDestroy(Adesc);
      cublasLtMatmulDescDestroy(operationDesc);
    }

    matxFree(a_hp);
    matxFree(b_hp);
    matxFree(c_hp);        
  }

/**
 * Execute a Matrix multiply (GEMM)
 *
 * Execute a matrix multiply operation on two rank=2 input tensors into an
 * output tensor. Using BLAS notation, tensor A has dimensions MxK, B is KxN,
 * and C is MxN. Concretely:
 *
 * \f$\textbf{C} = \alpha\textbf{A}\textbf{B} + \beta\textbf{C}\f$
 *
 * MatX will perform runtime checks ensuring that the dimension constraints are
 * met on all views. Unlike BLAS GEMMS, most parameters of the GEMM call are
 * deduced from the view itself; there is no need to specify dimensions or
 * transpose operations. MatX will attempt to perform the GEMM in the most
 * efficient way possible given the knowledge of the view.
 *
 * While GEMMs are strictly rank=2 functions, rank 3 and higher tensors may be
 * passed to this function, which has the effect of batching across the higher
 * dimensions.
 *
 * @note views being passed to matxGemm must not be permuted and must have a
 * contigous stride currently.
 *
 * @tparam T1
 *   Type of beta
 * @tparam T2
 *   Type of alpha
 * @param c
 *   Output tensor C
 * @param a
 *   Input tensor A
 * @param b
 *   Input tensor B
 * @param stream
 *   CUDA stream
 * @param alpha
 *   Alpha value
 * @param beta
 *   Beta value
 *
 */
  __MATX_INLINE__ void Exec(TensorTypeC &c, const TensorTypeA &a,
                   const TensorTypeB &b, cudaStream_t stream,
                   float alpha = 1.0f, float beta = 0.0f)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    // Reorder C/A to match cutlass API
    MatMulDispatchA(a, b, c, stream, alpha, beta);
  }

private:
  // Member variables
  cublasLtHandle_t ltHandle;
  cublasStatus_t ret = CUBLAS_STATUS_SUCCESS;

  // cuBLASLt variables;
  cublasHandle_t handle;
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr;
  cublasLtMatrixLayout_t Bdesc = nullptr;
  cublasLtMatrixLayout_t Cdesc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatrixTransformDesc_t transformDescI = nullptr;
  cublasLtMatrixTransformDesc_t transformDescO = nullptr;
  cublasLtMatrixLayout_t AtransformDesc = nullptr;
  cublasLtMatrixLayout_t BtransformDesc = nullptr;
  cublasLtMatrixLayout_t CtransformDesc = nullptr;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  void *c_hp = nullptr; // Make these void since they only work on complex types
  void *a_hp = nullptr;
  void *b_hp = nullptr;
  size_t workspaceSize = 1 << 22UL; // 4MB buffer suggested by cuBLAS team
  void *workspace = nullptr;
  detail::MatMulParams_t params_;

  void ConfigureCublasLt()
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    ret = cublasLtCreate(&ltHandle);
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatmulPreferenceCreate(&preference);
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatmulDescCreate(
                    &operationDesc, MatXTypeToCudaComputeType<T1>(),
                    MatXTypeToCudaType<T1>());
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatmulPreferenceSetAttribute(
                    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspaceSize,
                    sizeof(workspaceSize));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t colOrder = CUBLASLT_ORDER_COL;

    auto op = CUBLAS_OP_N;
    // A operation
    ret = cublasLtMatmulDescSetAttribute(
                    operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op,
                    sizeof(op));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    // B operation
    ret = cublasLtMatmulDescSetAttribute(
                    operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op,
                    sizeof(op));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    // Update this later when we're more flexible on compute type
    int32_t scaleType;
    if constexpr (std::is_same_v<T1, float> || is_matx_half_v<T1>) {
      scaleType = CUDA_R_32F;
    }
    else if constexpr (is_complex_half_v<T1> ||
                       std::is_same_v<T1, cuda::std::complex<float>>) {
      scaleType = CUDA_C_32F;
    }
    else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
      scaleType = CUDA_C_64F;
    }
    else {
      scaleType = CUDA_R_64F;
    }

    ret = cublasLtMatmulDescSetAttribute(
                    operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scaleType,
                    sizeof(scaleType));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    // Matrix layouts
    ret = cublasLtMatrixLayoutCreate(
                    &Adesc, MatXTypeToCudaType<T2>(), params_.a_rows,
                    params_.a_cols, params_.lda);
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret =cublasLtMatrixLayoutCreate(
                    &Bdesc, MatXTypeToCudaType<T3>(), params_.b_rows,
                    params_.b_cols, params_.ldb);
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatrixLayoutCreate(
                    &Cdesc, MatXTypeToCudaType<T1>(), params_.c_rows,
                    params_.c_cols, params_.ldc);
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    // Matrix data order
    if (params_.opA == CUBLAS_OP_T) {
      ret = cublasLtMatrixLayoutSetAttribute(
                      Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder,
                      sizeof(colOrder));
    }
    else {
      ret = cublasLtMatrixLayoutSetAttribute(
                      Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder,
                      sizeof(rowOrder));      
    }
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    if (params_.opB == CUBLAS_OP_T) {
      ret = cublasLtMatrixLayoutSetAttribute(
                      Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &colOrder,
                      sizeof(colOrder));
    }
    else {
      ret = cublasLtMatrixLayoutSetAttribute(
                      Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder,
                      sizeof(rowOrder));      
    }
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatrixLayoutSetAttribute(
                    Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder,
                    sizeof(rowOrder));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatrixLayoutSetAttribute(
                    Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &params_.batch,
                    sizeof(params_.batch));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatrixLayoutSetAttribute(
                    Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &params_.batch,
                    sizeof(params_.batch));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    ret = cublasLtMatrixLayoutSetAttribute(
                    Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &params_.batch,
                    sizeof(params_.batch));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    int64_t stride;

    if constexpr (is_complex_half_v<T2>) {
      // for complex half we have copied to planar row major
      // we know the layout of this matrix is compact
      stride = params_.a_rows * params_.a_cols * 2;
    }
    else {
      stride = params_.astride;
    }

    ret = cublasLtMatrixLayoutSetAttribute(
                    Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                    sizeof(stride));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    if constexpr (is_complex_half_v<T3>) {
      // for complex half we have copied to planar row major
      // we know the layout of this matrix is compact
      stride = params_.b_rows * params_.b_cols * 2;
    }
    else {
      stride = params_.bstride;
    }

    ret = cublasLtMatrixLayoutSetAttribute(
                    Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                    sizeof(stride));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

    if constexpr (is_complex_half_v<T1>) {
      // for complex half we have copied to planar row major
      // we know the layout of this matrix is compact
      stride = params_.c_rows * params_.c_cols * 2;
    }
    else {
      stride = params_.cstride;
    }

    ret = cublasLtMatrixLayoutSetAttribute(
                    Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                    sizeof(stride));
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);    

    if constexpr (is_complex_half_v<T1> && is_complex_half_v<T2>) {
      // for complex half we have copied to planar row major
      size_t planarA = (params_.a_rows * params_.a_cols * sizeof(T1)) / 2;
      size_t planarB = (params_.b_rows * params_.b_cols * sizeof(T1)) / 2;
      size_t planarC = (params_.c_rows * params_.c_cols * sizeof(T1)) / 2;

      ret = cublasLtMatrixLayoutSetAttribute(
                      Adesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &planarA,
                      sizeof(planarA));
      MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

      ret = cublasLtMatrixLayoutSetAttribute(
                      Bdesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &planarB,
                      sizeof(planarB));
      MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);

      ret = cublasLtMatrixLayoutSetAttribute(
                      Cdesc, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET, &planarC,
                      sizeof(planarC));
      MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);
    }

    int res;
    ret = cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc,
                                               Bdesc, Cdesc, Cdesc, preference,
                                               1, &heuristicResult,
                                               &res);
    MATX_ASSERT(ret == CUBLAS_STATUS_SUCCESS, matxMatMulError);
    MATX_ASSERT(res > 0, matxMatMulError);
  }

  // TODO: Fix the unused parameters once we support mixes of col/row on cublas
  template <MemOrder_t OrderA, MemOrder_t OrderB, MemOrder_t OrderC>
  __MATX_INLINE__ void
  MatMulLaunch(const TensorTypeA &a, const TensorTypeB &b,
               TensorTypeC &c, cudaStream_t stream,
               [[maybe_unused]] float alpha, [[maybe_unused]] float beta)
  {

    MATX_ASSERT_STR(PROV < PROVIDER_TYPE_SENTINEL, matxInvalidParameter, "Provider type out of range");
    if constexpr ((PROV == PROVIDER_TYPE_CUTLASS) &&
                  (is_complex_half_v<T1> || is_complex_half_v<T2>)) {
      MATX_THROW(matxInvalidType,
                 "CUTLASS does not support complex fp16/bf16 yet");
    }

    if constexpr ((is_complex_half_v<T1> && !is_complex_half_v<T2>) ||
                  (is_complex_half_v<T2> && !is_complex_half_v<T3>) ||
                  (is_complex_half_v<T1> && !is_complex_half_v<T3>)) {
      MATX_THROW(matxInvalidType,
                 "A/B/C types must all be half complex if any of them are");
    }
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    // Make copies of each tensor in case we have to do a transformation before
    // the GEMM
    [[maybe_unused]] TensorTypeA a_adj { a };
    [[maybe_unused]] TensorTypeB b_adj { b };
    [[maybe_unused]] TensorTypeC c_adj { c };

    // If the tensors are complex half precision, we need to do a planar
    // transform since all libraries expect this format at the moment.
    if constexpr (is_complex_half_v<T1>) {
      
      auto a_shape = a.Shape();
      *(a_shape.begin() + a.Rank() - 2) = a.Size(a.Rank() - 2) * 2;
      matxAlloc(&a_hp, a.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
      auto a_planar = make_tensor<typename T2::value_type>(reinterpret_cast<typename T2::value_type*>(a_hp), a_shape, false);

      auto b_shape = b.Shape();
      *(b_shape.begin() + b.Rank() - 2) = b.Size(b.Rank() - 2) * 2;
      matxAlloc(&b_hp, b.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
      auto b_planar = make_tensor<typename T3::value_type>(reinterpret_cast<typename T3::value_type*>(b_hp), b_shape, false);
      
      auto c_shape = c.Shape();
      *(c_shape.begin() + c.Rank() - 2) = c.Size(c.Rank() - 2) * 2;
      matxAlloc(&c_hp, c.Bytes(), MATX_ASYNC_DEVICE_MEMORY, stream);
      auto c_planar = make_tensor<typename T1::value_type>(reinterpret_cast<typename T1::value_type*>(c_hp), c_shape, false);

      // Convert A/B to planar layout
      (a_planar = planar(a)).run(stream);
      (b_planar = planar(b)).run(stream);

      // update pointers to planar data. 
      // must use Reset because types for planar are different
      a_adj.Reset(reinterpret_cast<T1 *>(a_planar.Data()));
      b_adj.Reset(reinterpret_cast<T2 *>(b_planar.Data()));
      c_adj.Reset(reinterpret_cast<T3 *>(c_planar.Data()));
    }

    // Prep for batch looping
    using shape_type = typename TensorTypeA::desc_type::shape_type;
    [[maybe_unused]] std::array<shape_type, TensorTypeA::Rank()> idx{0};
    [[maybe_unused]] auto a_shape = a.Shape();    
    [[maybe_unused]] size_t total_iter = 1;
    
    if constexpr (RANK > 3) {
      // Get total number of batches
      total_iter = std::accumulate(a_shape.begin(), a_shape.begin() + TensorTypeA::Rank() - 3, 1, std::multiplies<shape_type>());
    }

    // For cuBLASLt most of the parameters have already been set in the
    // configure stage
    if constexpr (PROV == PROVIDER_TYPE_CUBLASLT) {
      MatMulScaleType_t salpha, sbeta;
      memset(&salpha, 0, sizeof(salpha));
      memset(&sbeta, 0, sizeof(sbeta));

      if constexpr (std::is_same_v<T1, cuda::std::complex<float>> ||
                    is_complex_half_v<T1>) {
        salpha.cf32[0] = alpha;
        sbeta.cf32[0] = beta;
      }
      else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
        salpha.cf64[0] = alpha;
        sbeta.cf64[0] = beta;
      }
      else if constexpr (std::is_same_v<T1, float> || is_matx_half_v<T1>) {
        salpha.f32 = alpha;
        sbeta.f32 = beta;
      }
      else if constexpr (std::is_same_v<T1, double>) {
        salpha.f64 = alpha;
        sbeta.f64 = beta;
      }

      if constexpr (RANK <= 3) {
        auto res = cublasLtMatmul(
            ltHandle, operationDesc, &salpha, (void *)a_adj.Data(), Adesc,
            (void *)b_adj.Data(), Bdesc, &sbeta, (void *)c_adj.Data(), Cdesc,
            (void *)c_adj.Data(), Cdesc, &heuristicResult.algo, workspace,
            workspaceSize, stream);
        MATX_ASSERT(res == CUBLAS_STATUS_SUCCESS, matxMatMulError);
      }
      else {
        for (size_t iter = 0; iter < total_iter; iter++) {

          // Get pointers into A/B/C for this round
          auto ap = std::apply([&a_adj](auto... param) { return a_adj.GetPointer(param...); }, idx);
          auto bp = std::apply([&b_adj](auto... param) { return b_adj.GetPointer(param...); }, idx);
          auto cp = std::apply([&c_adj](auto... param) { return c_adj.GetPointer(param...); }, idx);
          auto res = cublasLtMatmul(
                  ltHandle, operationDesc, &salpha, (void *)ap,
                  Adesc, (void *)bp, Bdesc, &sbeta,
                  (void *)cp, Cdesc, (void *)cp,
                  Cdesc, &heuristicResult.algo, workspace, workspaceSize,
                  stream);

          MATX_ASSERT(res == CUBLAS_STATUS_SUCCESS, matxMatMulError);

          // Update all but the last 3 indices
          UpdateIndices<TensorTypeA, shape_type, TensorTypeA::Rank()>(a_adj, idx, 3);
        }
      }
    }

    if constexpr (RANK == 2) {
      if constexpr (PROV == PROVIDER_TYPE_CUTLASS) {
#if MATX_ENABLE_CUTLASS
        using CutlassAOrder = std::conditional_t<OrderA == MEM_ORDER_ROW_MAJOR,
                                                 cutlass::layout::RowMajor,
                                                 cutlass::layout::ColumnMajor>;
        using CutlassBOrder = std::conditional_t<OrderB == MEM_ORDER_ROW_MAJOR,
                                                 cutlass::layout::RowMajor,
                                                 cutlass::layout::ColumnMajor>;
        using CutlassCOrder = std::conditional_t<OrderC == MEM_ORDER_ROW_MAJOR,
                                                 cutlass::layout::RowMajor,
                                                 cutlass::layout::ColumnMajor>;
        using CutlassGemm =
            cutlass::gemm::device::Gemm<T1,             // Data-type of A matrix
                                        CutlassAOrder,  // Layout of A matrix
                                        T2,             // Data-type of B matrix
                                        CutlassBOrder,  // Layout of B matrix
                                        T3,             // Data-type of C matrix
                                        CutlassCOrder>; // Layout of C matrix

        typename CutlassGemm::Arguments args(
            {static_cast<int>(params_.m), static_cast<int>(params_.n),
             static_cast<int>(params_.k)}, // Gemm Problem dimensions
            {a.Data(),
             static_cast<int>(params_.lda)}, // Tensor-ref for source matrix A
            {b.Data(),
             static_cast<int>(params_.ldb)}, // Tensor-ref for source matrix B
            {c.Data(),
             static_cast<int>(params_.ldc)}, // Tensor-ref for source matrix C
            {c.Data(),
             static_cast<int>(
                 params_.ldc)}, // Tensor-ref for destination matrix D (may be
                                // different memory than source C matrix)
            {alpha, beta});     // Scalars used in the Epilogue

        CutlassGemm gemm_operator;
        cutlass::Status status = gemm_operator(args, nullptr, stream);

        MATX_ASSERT(status == cutlass::Status::kSuccess, matxMatMulError);
#else
        MATX_THROW(matxNotSupported, "CUTLASS not enabled!");
#endif
      }
    }
    else {
      static_assert(RANK > 2);
#if MATX_ENABLE_CUTLASS
      using CutlassAOrder = std::conditional_t<OrderA == MEM_ORDER_ROW_MAJOR,
                                               cutlass::layout::RowMajor,
                                               cutlass::layout::ColumnMajor>;
      using CutlassBOrder = std::conditional_t<OrderB == MEM_ORDER_ROW_MAJOR,
                                               cutlass::layout::RowMajor,
                                               cutlass::layout::ColumnMajor>;
      using CutlassCOrder = std::conditional_t<OrderC == MEM_ORDER_ROW_MAJOR,
                                               cutlass::layout::RowMajor,
                                               cutlass::layout::ColumnMajor>;
      using CutlassGemm = cutlass::gemm::device::GemmBatched<
          T1,             // Data-type of A matrix
          CutlassAOrder,  // Layout of A matrix
          T2,             // Data-type of B matrix
          CutlassBOrder,  // Layout of B matrix
          T3,             // Data-type of C matrix
          CutlassCOrder>; // Layout of C matrix
#endif

      if constexpr (RANK > 3) {
        if constexpr (PROV == PROVIDER_TYPE_CUTLASS) {
#if MATX_ENABLE_CUTLASS
        for (size_t iter = 0; iter < total_iter; iter++) {
          // Get pointers into A/B/C for this round
          auto ap = std::apply([&a_adj](auto... param) { return a_adj.GetPointer(param...); }, idx);
          auto bp = std::apply([&b_adj](auto... param) { return b_adj.GetPointer(param...); }, idx);
          auto cp = std::apply([&c_adj](auto... param) { return c_adj.GetPointer(param...); }, idx);

          typename CutlassGemm::Arguments args(
              {static_cast<int>(params_.m), static_cast<int>(params_.n),
                static_cast<int>(params_.k)}, // Gemm Problem dimensions
              {ap,
                static_cast<int>(
                    params_.lda)},     // Tensor-ref for source matrix A
              a_adj.Stride(RANK - 3), // Batch Stride A
              {bp,
                static_cast<int>(
                    params_.ldb)},     // Tensor-ref for source matrix B
              b_adj.Stride(RANK - 3), // Batch Stride B
              {cp,
                static_cast<int>(
                    params_.ldc)},     // Tensor-ref for source matrix C
              c_adj.Stride(RANK - 3), // Batch Stride C
              {cp,
                static_cast<int>(
                    params_.ldc)}, // Tensor-ref for destination matrix D (may
                                  // be different memory than source C matrix)
              c_adj.Stride(RANK - 3), // Batch Stride C
              {alpha, beta},
              params_.batch // Batch Dimension
          );                // Scalars used in the Epilogue

          CutlassGemm gemm_operator;
          cutlass::Status status = gemm_operator(args, nullptr, stream);
          MATX_ASSERT(status == cutlass::Status::kSuccess, matxMatMulError);

          // Update all but the last 2 indices
          UpdateIndices<TensorTypeA, shape_type, TensorTypeA::Rank()>(a_adj, idx, 3);
        }
#else
          MATX_THROW(matxNotSupported, "CUTLASS not enabled!");
#endif
        }
        else {
          MATX_STATIC_ASSERT_STR(PROV < PROVIDER_TYPE_SENTINEL, matxInvalidParameter, "Invalid MatMul provider");
        }
      }
    }

    // If the tensors are complex half precisions, we need to convert C back to
    // interleaved format and free all temporary buffers
    if constexpr (is_complex_half_v<T1>) {
      auto c_shape = c.Shape();
      *(c_shape.begin() + c.Rank() - 2) = c.Size(c.Rank() - 2) * 2;
      auto c_planar = make_tensor<typename T3::value_type>(
          reinterpret_cast<typename T3::value_type *>(c_adj.Data()), c_shape);

      // Convert A/B to planar layout
      (c = interleaved(c_planar)).run(stream);
    }
  }
  
  template <MemOrder_t OrderA, MemOrder_t OrderB>
  inline void MatMulDispatchC(const TensorTypeA &a,
                              const TensorTypeB &b,
                              TensorTypeC &c, cudaStream_t stream,
                              float alpha, float beta)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    if (c.Stride(RANK - 1) <= 1) {
      MatMulLaunch<OrderA, OrderB, MEM_ORDER_ROW_MAJOR>(a, b, c, stream, alpha,
                                                        beta);
    }
    else if (c.Stride(RANK - 2) <= 1) {
#if MATX_ENABLE_CUTLASS
      MatMulLaunch<OrderA, OrderB, MEM_ORDER_COL_MAJOR>(a, b, c, stream, alpha,
                                                   beta);
#else
      // This shouldn't ever kick up as transpose should happen earlier
      MATX_ASSERT_STR(false, matxInvalidDim, "matmul:  column major order on C not supported with cublas.  Try cutlass.");
#endif
    }
    else {
      MATX_THROW(matxNotSupported,
                 "Matrix multiply on Affine Matrix Not supported");
    }
  };

  template <MemOrder_t OrderA>
  inline void MatMulDispatchB(const TensorTypeA &a,
                              const TensorTypeB &b,
                              TensorTypeC &c, cudaStream_t stream,
                              float alpha, float beta)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    if (b.Stride(RANK - 1) == 1) {
      MatMulDispatchC<OrderA, MEM_ORDER_ROW_MAJOR>(a, b, c, stream, alpha,
                                                   beta);
    }
    else if (b.Stride(RANK - 2) == 1) {
      MatMulDispatchC<OrderA, MEM_ORDER_COL_MAJOR>(a, b, c, stream, alpha,
                                                   beta);
    }
    else {
      MATX_THROW(matxNotSupported,
                 "Matrix multiply on Affine Matrix Not supported");
    }
  }

  inline void MatMulDispatchA(const TensorTypeA &a,
                              const TensorTypeB &b,
                              TensorTypeC &c, cudaStream_t stream,
                              float alpha, float beta)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    if (a.Stride(RANK - 1) == 1) {
      MatMulDispatchB<MEM_ORDER_ROW_MAJOR>(a, b, c, stream, alpha, beta);
    }
    else if (a.Stride(RANK - 2) == 1) {
      MatMulDispatchB<MEM_ORDER_COL_MAJOR>(a, b, c, stream, alpha, beta);
    }
    else {
      MATX_THROW(matxNotSupported,
                 "Matrix multiply on Affine Matrix Not supported");
    }
  }
};

/**
 * Crude hash on GEMM to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common GEMM parameters change
 */
struct MatMulParamsKeyHash {
  std::size_t operator()(const MatMulParams_t &k) const noexcept
  {
    return std::hash<uint64_t>()(k.m) + std::hash<uint64_t>()(k.n) +
           std::hash<uint64_t>()(k.k) + std::hash<uint64_t>()(k.batch) +
           std::hash<uint64_t>()(k.prov) +
           std::hash<uint64_t>()((size_t)k.stream);
  }
};

/**
 * Test GEMM parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct MatMulParamsKeyEq {
  bool operator()(const MatMulParams_t &l, const MatMulParams_t &t) const
      noexcept
  {
    return l.m == t.m && l.n == t.n && l.k == t.k && l.a_rows == t.a_rows &&
           l.b_rows == t.b_rows && l.c_rows == t.c_rows &&
           l.a_cols == t.a_cols && l.b_cols == t.b_cols &&
           l.c_cols == t.c_cols && l.stream == t.stream && l.lda == t.lda &&
           l.ldb == t.ldb && l.ldc == t.ldc && l.batch == t.batch &&
           l.prov == t.prov && l.dtype == t.dtype && l.opA == t.opA &&
           l.opB == t.opB;
  }
};

// Static caches of GEMMs
static matxCache_t<MatMulParams_t, MatMulParamsKeyHash, MatMulParamsKeyEq>
    gemm_cache;

}

template <typename Op>
__MATX_INLINE__ auto getCublasSupportedTensor( const Op &in, cudaStream_t stream) {
  constexpr int RANK=Op::Rank();

  if constexpr ( !(is_tensor_view_v<Op>)) {
    return make_tensor<typename Op::scalar_type>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
  } else {
    bool supported = true;

    if(
    
      // either RANK-1 or RANK-2 stride must equal one in cublasLt
      (in.Stride(RANK-1) != 1 && in.Stride(RANK-2) != 1) || 
      // cublas allows 0 strides, but verify that the corresponding size is 1
      (in.Stride(RANK-1) == 0 && in.Size(RANK-1) != 1) ||
      (in.Stride(RANK-2) == 0 && in.Size(RANK-2) != 1)
      ) {
      supported = false;
    }

    if(supported) {
      return in;
    } else {
      return make_tensor<typename Op::scalar_type>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
    }
  }
}

/**
 * Run a GEMM without a plan
 *
 * Creates a new GEMM plan in the cache if none exists, and uses that to execute
 * the GEMM. This function is preferred over creating a plan directly for both
 * efficiency and simpler code. Since it only uses the signature of the GEMM to
 * decide if a plan is cached, it may be able to reused plans for different
 * A/B/C matrices as long as they were configured with the same dimensions.
 *
 * @tparam TensorTypeC
 *    Data type of C tensor or operator
 * @tparam TensorTypeA
 *    Data type of A tensor or operator
 * @tparam TensorTypeB
 *    Data type of B tensor or operator
 * @tparam PROV
 *    Provider type chosen from MatXMatMulProvider_t type
 *
 * @param C
 *   C A Tensor or Operator
 * @param A
 *   A A Tensor or Operator
 * @param B
 *   B A Tensor or Operator
 * @param stream
 *   CUDA stream
 * @param alpha
 *   Scalar multiplier to apply to operator A
 * @param beta
 *   Scalar multiplier to apply to operator C on input
 */
template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, 
          MatXMatMulProvider_t PROV = PROVIDER_TYPE_CUBLASLT>
void matmul(TensorTypeC C, const TensorTypeA A,
            const TensorTypeB B, cudaStream_t stream = 0,
            float alpha = 1.0, float beta = 0.0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  constexpr auto is_c_complex = is_complex_v<typename TensorTypeC::scalar_type>;

  if constexpr (is_c_complex) {
    constexpr auto is_a_complex = is_complex_v<typename TensorTypeA::scalar_type>;
    constexpr auto is_b_complex = is_complex_v<typename TensorTypeB::scalar_type>;
    static_assert(is_a_complex || is_b_complex, "If C is complex then either A or B should be complex ");
  }

  // promote A and B to the type of C
  auto A_ = as_type<typename TensorTypeC::scalar_type>(A);
  auto B_ = as_type<typename TensorTypeC::scalar_type>(B);

  // CublasLt does not support operators and certain transpose modes.
  // Grab a suppported tensor here and copy in if necessary.
  auto c = getCublasSupportedTensor(C, stream);
  auto a = getCublasSupportedTensor(A_, stream);
  auto b = getCublasSupportedTensor(B_, stream);

  typedef decltype(c) ctype;
  typedef decltype(a) atype;
  typedef decltype(b) btype;

  if(!a.isSameView(A_)) {
    (a = A_).run(stream);
  }

  if(!b.isSameView(B_)) {
    (b = B_).run(stream);
  }

#if MATX_ENABLE_CUTLASS != 1
  // cublasLt does not allow transpose modes on C.  Thus we need to make sure that the right most dimension has a stride of 1.
  // Use the identity CT = BT * AT to do the transpose through the gemm automatically.  Note we only want to do this transpose if
  // the rightmost stride is !=1 or this function will be an infinite recursion.
  if ( c.Stride(c.Rank()-2) == 1 && c.Stride(c.Rank()-1) > 1 ) {  // column major check
    // Column major
    matmul(transpose(c), transpose(b), transpose(a), stream, alpha, beta);
  } else 
#endif
  {
    // Get parameters required by these tensors
    auto params =
      detail::matxMatMulHandle_t<ctype, atype, btype, PROV>::GetGemmParams(c, a, b);
    params.stream = stream;

    // Get cache or new GEMM plan if it doesn't exist
    auto ret = detail::gemm_cache.Lookup(params);
    if (ret == std::nullopt) {
      auto tmp = new detail::matxMatMulHandle_t<ctype, atype, btype, PROV>{c, a, b};
      detail::gemm_cache.Insert(params, static_cast<void *>(tmp));

      // Set the stream on this plan once on creation
      tmp->Exec(c, a, b, stream, alpha, beta);
    }
    else {
      auto gemm_type =
        static_cast<detail::matxMatMulHandle_t<ctype, atype, btype, PROV> *>(ret.value());
      gemm_type->Exec(c, a, b, stream, alpha, beta);
    }
  }

  // if c and C are not the same then we need to copy results out.
  if(!c.isSameView(C)) {
    (C = c).run(stream);
  }
}

/**
 * Run a GEMM without a plan
 *
 * Creates a new GEMM plan in the cache if none exists, and uses that to execute
 * the GEMM. This function is preferred over creating a plan directly for both
 * efficiency and simpler code. Since it only uses the signature of the GEMM to
 * decide if a plan is cached, it may be able to reused plans for different
 * A/B/C matrices as long as they were configured with the same dimensions.
 *
 * @tparam TensorTypeC
 *    Data type of C tensor or operator
 * @tparam TensorTypeA
 *    Data type of A tensor or operator
 * @tparam TensorTypeB
 *    Data type of B tensor or operator
 * @tparam PROV
 *    Provider type chosen from MatXMatMulProvider_t type
 *
 * @param C
 *   C output tensor or operator
 * @param A
 *   A input tensor or operator
 * @param B
 *   B input tensor or operator
 * @param axis
 *   the axis of the tensor or operator to perform the gemm along
 * @param stream
 *   CUDA stream
 * @param alpha
 *   Scalar multiplier to apply to operator A
 * @param beta
 *   Scalar multiplier to apply to operator C on input
 */
template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, 
          MatXMatMulProvider_t PROV = PROVIDER_TYPE_CUBLASLT>
__MATX_INLINE__ void matmul(TensorTypeC C, const TensorTypeA A,
            const TensorTypeB B, const int32_t (&axis)[2],
            cudaStream_t stream = 0,
            float alpha = 1.0, float beta = 0.0)
{
  MATX_STATIC_ASSERT(TensorTypeA::Rank() == TensorTypeB::Rank(), "matmul: inputs must have same rank to use matmul with axis parameter");
  MATX_STATIC_ASSERT(TensorTypeA::Rank() == TensorTypeC::Rank(), "matmul: inputs and outputs must have same rank to use matmul with axis parameter");

  auto perm = detail::getPermuteDims<TensorTypeC::Rank()>(axis);
  auto out = permute(C, perm);

  auto in1 = permute(A, perm);
  auto in2 = permute(B, perm);

  matmul<TensorTypeC, TensorTypeA, TensorTypeB, PROV>(out, in1, in2, stream, alpha, beta);
}

/**
 * Run a GEMV without a plan
 *
 * Performs the GEMV:  C = beta*C + alpha*A*B where A is a matrix and B and C are vectors.
 *
 * Creates a new GEMM plan in the cache if none exists, and uses that to execute
 * the GEMM. This function is preferred over creating a plan directly for both
 * efficiency and simpler code. Since it only uses the signature of the GEMM to
 * decide if a plan is cached, it may be able to reused plans for different
 * A/B/C matrices as long as they were configured with the same dimensions.
 *
 * @tparam TensorTypeC
 *    Data type of C tensor or operator
 * @tparam TensorTypeA
 *    Data type of A tensor or operator
 * @tparam TensorTypeB
 *    Data type of B tensor or operator
 * @tparam PROV
 *    Provider type chosen from MatXMatMulProvider_t type
 *
 * @param C
 *   C output tensor or operator
 * @param A
 *   A input tensor or operator
 * @param B
 *   B input tensor or operator
 * @param stream
 *   CUDA stream
 * @param alpha
 *   Scalar multiplier to apply to operator A
 * @param beta
 *   Scalar multiplier to apply to operator C on input
 */
template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, 
          MatXMatMulProvider_t PROV = PROVIDER_TYPE_CUBLASLT>
__MATX_INLINE__ void matvec(TensorTypeC C, const TensorTypeA A,
            const TensorTypeB B,
            cudaStream_t stream = 0,
            float alpha = 1.0, float beta = 0.0)
{
  MATX_STATIC_ASSERT(TensorTypeA::Rank() == TensorTypeB::Rank()+1, "matvec: A rank must be one larger than B rank");
  
  MATX_ASSERT_STR(C.Size(TensorTypeB::Rank()-1) == A.Size(TensorTypeA::Rank()-2), matxInvalidDim, "matvec: C last size must match A second last Size");
  MATX_ASSERT_STR(B.Size(TensorTypeB::Rank()-1) == A.Size(TensorTypeA::Rank()-1), matxInvalidDim, "matvec: B last size must match A last size");

  // need to clone c and b 1 along inner dim to use cublas
  std::array<index_t, TensorTypeC::Rank()+1> shape;
  for(int i = 0; i < TensorTypeC::Rank(); i++) {
    shape[i] = matxKeepDim;
  }
  // clone last dim by 1 to create an Nx1 matrix
  shape[TensorTypeC::Rank()]=1;

  auto c = clone<TensorTypeC::Rank()+1>(C, shape);
  auto b = clone<TensorTypeB::Rank()+1>(B, shape);

  matmul<decltype(c), decltype(A), decltype(b), PROV>(c, A, b, stream, alpha, beta);
}

} // end namespace matx
