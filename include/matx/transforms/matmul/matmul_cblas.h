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

#include "matx/core/error.h"
#include "matx/core/make_tensor.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/executors/host.h"
#include "matx/executors/support.h"
#include "matx/transforms/matmul/matmul_common.h"

#include <cstdio>
#include <numeric>

#ifdef MATX_EN_NVPL
  #ifndef nvpl_scomplex_t
    #define nvpl_scomplex_t cuda::std::complex<float>
    #define nvpl_dcomplex_t cuda::std::complex<double>
  #endif
  #include <nvpl_blas_cblas.h>
  using cblas_int_t = nvpl_int_t;
#elif defined(MATX_EN_OPENBLAS)
  #include <cblas.h>
  using cblas_int_t = blasint;
#elif defined(MATX_EN_BLIS)
  #ifdef MATX_BLIS_64_HEADER
    #include <cblas64.h>
    #include <blis64.h>
  #else
    #include <cblas.h>
    #include <blis.h>
  #endif
  using cblas_int_t = f77_int;
#endif

namespace matx {

namespace detail {

template <typename OpA, typename OpB, typename OpC>
constexpr bool CompatibleGemmCBLASTypes() {
  // All 3 should be the same type
  if constexpr (!std::is_same_v<typename OpA::value_type, typename OpB::value_type> ||
                !std::is_same_v<typename OpB::value_type, typename OpC::value_type>) {
    return false;
  }

  // List of accepted types when A/B/C match
  return std::is_same_v<typename OpA::value_type, float> ||
         std::is_same_v<typename OpA::value_type, double> ||
         std::is_same_v<typename OpA::value_type, cuda::std::complex<float>> ||
         std::is_same_v<typename OpA::value_type, cuda::std::complex<double>>;
}

#if MATX_EN_CPU_MATMUL
/**
 * Parameters needed to execute a CBLAS GEMM. For the most part, these are very
 * similar to that of a standard GEMM call
 */
struct MatMulCBLASParams_t {
  cblas_int_t m;
  cblas_int_t n;
  cblas_int_t k;
  cblas_int_t lda;
  cblas_int_t ldb;
  cblas_int_t ldc;
  int rank;
  cblas_int_t batch;
  cblas_int_t astride;
  cblas_int_t bstride;
  cblas_int_t cstride;
  MatXDataType_t dtype;
  CBLAS_TRANSPOSE opA;
  CBLAS_TRANSPOSE opB;
};

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
static MatMulCBLASParams_t GetGemmParams(TensorTypeC &c,
                                         const TensorTypeA &a,
                                         const TensorTypeB &b)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  static constexpr int RANK = TensorTypeC::Rank();

  /* If a user passes in a tensor where the last two dimensions are transposed
     we retain the original size parameters, but tell the underlying libraries
     that the tensors are in column-major ordering.
  */
  MatMulCBLASParams_t params;
  params.dtype = TypeToInt<typename TensorTypeC::value_type>();
  params.rank = c.Rank();

  // Batches
  params.batch = 1;
  params.astride = 0;
  params.bstride = 0;
  params.cstride = 0;

  // If we have a 3D or above tensor, the upper dims are batch dimensions.
  if constexpr (RANK >= 3) {
    params.batch = static_cast<cblas_int_t>(c.Size(RANK - 3));
    if constexpr (TensorTypeA::Rank() == RANK) {
      params.astride = static_cast<cblas_int_t>(a.Stride(TensorTypeA::Rank() - 3));
    } else {
      params.astride = 0;
    }

    if constexpr (TensorTypeB::Rank() == RANK) {
      params.bstride = static_cast<cblas_int_t>(b.Stride(TensorTypeB::Rank() - 3));
    } else {
      params.bstride = 0;
    }

    params.cstride = static_cast<cblas_int_t>(c.Stride(RANK - 3));
  }

  // At this point, the transpose mode on C case has already been handled
  if (a.Stride(TensorTypeA::Rank() - 1) > 1) { // last stride > 1
    params.opA = CblasTrans;
  } else { // otherwise row major
    params.opA = CblasNoTrans;
  }

  if (b.Stride(TensorTypeB::Rank() - 1) > 1) { // last stride > 1
    params.opB = CblasTrans;
  } else { // otherwise row major
    params.opB = CblasNoTrans;
  }

  // set lda/ldb according to transpose modes. If we pass in a cloned tensor
  // the second stride can be 0 (such as for an outer product), which cblas
  // doesn't like even though it's unused. Set it to something that it would be
  // if the matrix had more than 1 row.
  if (params.opB == CblasTrans) {
    params.ldb = static_cast<cblas_int_t>(b.Stride(TensorTypeB::Rank() - 1));
  } else {
    params.ldb = static_cast<cblas_int_t>(b.Stride(TensorTypeB::Rank() - 2));
    params.ldb = (params.ldb == 0) ? static_cast<cblas_int_t>(b.Size(TensorTypeB::Rank() - 1)) : params.ldb;
  }

  if (params.opA == CblasTrans) {
    params.lda = static_cast<cblas_int_t>(a.Stride(TensorTypeA::Rank() - 1));
  } else {
    params.lda = static_cast<cblas_int_t>(a.Stride(TensorTypeA::Rank() - 2));
    params.lda = (params.lda == 0) ? static_cast<cblas_int_t>(a.Size(TensorTypeA::Rank() - 1)) : params.lda;
  }

  params.ldc = static_cast<cblas_int_t>(c.Stride(RANK - 2));

  params.m = static_cast<cblas_int_t>(a.Size(TensorTypeA::Rank() - 2));
  params.n = static_cast<cblas_int_t>(b.Size(TensorTypeB::Rank() - 1));
  params.k = static_cast<cblas_int_t>(a.Size(TensorTypeA::Rank() - 1));

  return params;
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
 * @tparam TensorTypeC
 *   Data type of C tensor or operator
 * @tparam TensorTypeA
 *   Data type of A tensor or operator
 * @tparam TensorTypeB
 *   Data type of B tensor or operator
 * @tparam MODE
 *   Threading policy
 *
 * @param c
 *   Output tensor C
 * @param a
 *   Input tensor A
 * @param b
 *   Input tensor B
 * @param params
 *   GEMM params
 * @param alpha
 *   Alpha value
 * @param beta
 *   Beta value
 * @param exec
 *   Host executor
 *
 */
template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, ThreadsMode MODE>
__MATX_INLINE__ void matmul_exec(TensorTypeC &c,
                                 const TensorTypeA &a,
                                 const TensorTypeB &b,
                                 MatMulCBLASParams_t &params,
                                 const float alpha,
                                 const float beta,
                                 [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static constexpr int RANK = TensorTypeC::Rank();
  using value_type = typename TensorTypeC::value_type;

  // Prep for batch looping
  using shape_type = typename TensorTypeA::desc_type::shape_type;
  [[maybe_unused]] cuda::std::array<shape_type, TensorTypeA::Rank()> a_idx{0};
  [[maybe_unused]] cuda::std::array<shape_type, TensorTypeB::Rank()> b_idx{0};
  [[maybe_unused]] cuda::std::array<shape_type, TensorTypeC::Rank()> c_idx{0};
  [[maybe_unused]] size_t total_iter = 1;

  if constexpr (RANK > 3) {
    // Get total number of batches
    auto c_shape = c.Shape();
    total_iter = std::accumulate(c_shape.begin(),
                                 c_shape.begin() + TensorTypeC::Rank() - 3, 1,
                                 std::multiplies<shape_type>());
  }

  value_type salpha;
  value_type sbeta;
  if constexpr (std::is_same_v<value_type, cuda::std::complex<float>> ||
                std::is_same_v<value_type, cuda::std::complex<double>>) {
    salpha = {alpha, 0};
    sbeta = {beta, 0};
  }
  else if constexpr (std::is_same_v<value_type, float> ||
                     std::is_same_v<value_type, double>) {
    salpha = alpha;;
    sbeta = beta;
  }

#ifdef MATX_EN_NVPL
  if constexpr (RANK <= 3) {
    auto a_ptr = a.Data();
    auto b_ptr = b.Data();
    auto c_ptr = c.Data();

    if constexpr (std::is_same_v<value_type, float>) {
      cblas_sgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                params.m, params.n, params.k, salpha,
                                a_ptr, params.lda, params.astride,
                                b_ptr, params.ldb, params.bstride, sbeta,
                                c_ptr, params.ldc, params.cstride, params.batch);
    } else if constexpr (std::is_same_v<value_type, double>) {
      cblas_dgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                params.m, params.n, params.k, salpha,
                                a_ptr, params.lda, params.astride,
                                b_ptr, params.ldb, params.bstride, sbeta,
                                c_ptr, params.ldc, params.cstride, params.batch);
    } else if constexpr (std::is_same_v<value_type, cuda::std::complex<float>>) {
      cblas_cgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                params.m, params.n, params.k, (void *)&salpha,
                                (void *)a_ptr, params.lda, params.astride,
                                (void *)b_ptr, params.ldb, params.bstride, (void *)&sbeta,
                                (void *)c_ptr, params.ldc, params.cstride, params.batch);
    } else if constexpr (std::is_same_v<value_type, cuda::std::complex<double>>) {
      cblas_zgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                params.m, params.n, params.k, (void *)&salpha,
                                (void *)a_ptr, params.lda, params.astride,
                                (void *)b_ptr, params.ldb, params.bstride, (void *)&sbeta,
                                (void *)c_ptr, params.ldc, params.cstride, params.batch);
    }
  } else {
    for (size_t iter = 0; iter < total_iter; iter++) {

      // Get pointers into A/B/C for this round
      auto ap = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, a_idx);
      auto bp = cuda::std::apply([&b](auto... param) { return b.GetPointer(param...); }, b_idx);
      auto cp = cuda::std::apply([&c](auto... param) { return c.GetPointer(param...); }, c_idx);

      if constexpr (std::is_same_v<value_type, float>) {
        cblas_sgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                  params.m, params.n, params.k, salpha,
                                  ap, params.lda, params.astride,
                                  bp, params.ldb, params.bstride, sbeta,
                                  cp, params.ldc, params.cstride, params.batch);
      } else if constexpr (std::is_same_v<value_type, double>) {
        cblas_dgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                  params.m, params.n, params.k, salpha,
                                  ap, params.lda, params.astride,
                                  bp, params.ldb, params.bstride, sbeta,
                                  cp, params.ldc, params.cstride, params.batch);
      } else if constexpr (std::is_same_v<value_type, cuda::std::complex<float>>) {
        cblas_cgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                  params.m, params.n, params.k, (void *)&salpha,
                                  (void *)ap, params.lda, params.astride,
                                  (void *)bp, params.ldb, params.bstride, (void *)&sbeta,
                                  (void *)cp, params.ldc, params.cstride, params.batch);
      } else if constexpr (std::is_same_v<value_type, cuda::std::complex<double>>) {
        cblas_zgemm_batch_strided(CblasRowMajor, params.opA, params.opB,
                                  params.m, params.n, params.k, (void *)&salpha,
                                  (void *)ap, params.lda, params.astride,
                                  (void *)bp, params.ldb, params.bstride, (void *)&sbeta,
                                  (void *)cp, params.ldc, params.cstride, params.batch);
      }

      // Update all but the last 3 indices
      UpdateIndices<TensorTypeA, shape_type, TensorTypeA::Rank()>(a, a_idx, 3);
      UpdateIndices<TensorTypeB, shape_type, TensorTypeB::Rank()>(b, b_idx, 3);
      UpdateIndices<TensorTypeC, shape_type, TensorTypeC::Rank()>(c, c_idx, 3);
    }
  }
#else
  // The batch api is a new addition to BLIS and OpenBLAS, so it may not be present.
  // Thus, we default to the standard gemm api and loop over anything above the 2nd dimension.

  #ifdef MATX_EN_OPENBLAS
  openblas_set_num_threads(exec.GetNumThreads());
  #elif defined(MATX_EN_BLIS)
  bli_thread_set_num_threads(exec.GetNumThreads());
  #endif
  
  total_iter *= params.batch;
  for (size_t iter = 0; iter < total_iter; iter++) {
    // Get pointers into A/B/C for this round
    auto ap = cuda::std::apply([&a](auto... param) { return a.GetPointer(param...); }, a_idx);
    auto bp = cuda::std::apply([&b](auto... param) { return b.GetPointer(param...); }, b_idx);
    auto cp = cuda::std::apply([&c](auto... param) { return c.GetPointer(param...); }, c_idx);

    if constexpr (std::is_same_v<value_type, float>) {
      cblas_sgemm(CblasRowMajor, params.opA, params.opB,
                  params.m, params.n, params.k, salpha,
                  ap, params.lda,
                  bp, params.ldb, sbeta,
                  cp, params.ldc);
    } else if constexpr (std::is_same_v<value_type, double>) {
      cblas_dgemm(CblasRowMajor, params.opA, params.opB,
                  params.m, params.n, params.k, salpha,
                  ap, params.lda,
                  bp, params.ldb, sbeta,
                  cp, params.ldc);
    } else if constexpr (std::is_same_v<value_type, cuda::std::complex<float>>) {
      cblas_cgemm(CblasRowMajor, params.opA, params.opB,
                  params.m, params.n, params.k, (void *)&salpha,
                  (const void **)ap, params.lda,
                  (const void **)bp, params.ldb, (void *)&sbeta,
                  (void **)cp, params.ldc);
    } else if constexpr (std::is_same_v<value_type, cuda::std::complex<double>>) {
      cblas_zgemm(CblasRowMajor, params.opA, params.opB,
                  params.m, params.n, params.k, (void *)&salpha,
                  (const void **)ap, params.lda,
                  (const void **)bp, params.ldb, (void *)&sbeta,
                  (void **)cp, params.ldc);
    }

    // Update all but the last 2 indices
    UpdateIndices<TensorTypeA, shape_type, TensorTypeA::Rank()>(a, a_idx, 2);
    UpdateIndices<TensorTypeB, shape_type, TensorTypeB::Rank()>(b, b_idx, 2);
    UpdateIndices<TensorTypeC, shape_type, TensorTypeC::Rank()>(c, c_idx, 2);
  }
#endif
}

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, ThreadsMode MODE>
__MATX_INLINE__ void matmul_dispatch(TensorTypeC &c, 
                                     const TensorTypeA &a,
                                     const TensorTypeB &b,
                                     const float alpha,
                                     const float beta,
                                     const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  static constexpr int RANK = TensorTypeC::Rank();

  // We allow a batch stride of 0 on one of the tensors, so only make sure C's
  // rank matches one of them
  MATX_STATIC_ASSERT_STR(RANK == TensorTypeB::Rank() || TensorTypeB::Rank() == 2, matxInvalidDim, "Tensor ranks do not match");
  MATX_STATIC_ASSERT_STR(RANK == TensorTypeA::Rank() || TensorTypeA::Rank() == 2, matxInvalidDim, "Tensor ranks do not match");

  MATX_STATIC_ASSERT_STR(RANK >= 2, matxInvalidDim, "Output tensor must be rank 2 or higher");
  MATX_ASSERT(a.Size(TensorTypeA::Rank() - 1) == b.Size(TensorTypeB::Rank() - 2), matxInvalidSize);
  MATX_ASSERT(c.Size(RANK - 1) == b.Size(TensorTypeB::Rank() - 1), matxInvalidSize);
  MATX_ASSERT(c.Size(RANK - 2) == a.Size(TensorTypeA::Rank() - 2), matxInvalidSize);

  // Ensure batch dimensions are equal
  for (int i = 0; i < RANK - 2; i++) {
    if constexpr (RANK == TensorTypeA::Rank()) {
      MATX_ASSERT(a.Size(i) == c.Size(i), matxInvalidSize);
    }
    if constexpr (RANK == TensorTypeB::Rank()) {
      MATX_ASSERT(b.Size(i) == c.Size(i), matxInvalidSize);
    }
  }

  auto params = GetGemmParams(c, a, b);
  matmul_exec(c, a, b, params, alpha, beta, exec);
}
#endif

} // end namespace detail

template <typename Op>
__MATX_INLINE__ auto getCBLASSupportedTensor(const Op &in) {
  constexpr int RANK = Op::Rank();

  if constexpr (!(is_tensor_view_v<Op>)) {
    return make_tensor<typename Op::value_type>(in.Shape(), MATX_HOST_MALLOC_MEMORY);
  } else {
    bool supported = true;

    if (
        // either RANK-1 or RANK-2 stride must equal one in cblas
        (in.Stride(RANK - 1) != (index_t)1 && in.Stride(RANK - 2) != (index_t)1) ||
        // verify that the corresponding size of a 0 stride dim is 1.
        // otherwise, it means a vector was repeated along a dimension, requiring a new tensor
        (in.Stride(RANK - 1) == (index_t)0 && in.Size(RANK - 1) != (index_t)1) ||
        (in.Stride(RANK - 2) == (index_t)0 && in.Size(RANK - 2) != (index_t)1)) {
      supported = false;
    }

    if (supported) {
      return in;
    } else {
      return make_tensor<typename Op::value_type>(in.Shape(), MATX_HOST_MALLOC_MEMORY);
    }
  }
}

/**
 * Run a CBLAS GEMM
 *
 * @tparam TensorTypeC
 *   Data type of C tensor or operator
 * @tparam TensorTypeA
 *   Data type of A tensor or operator
 * @tparam TensorTypeB
 *   Data type of B tensor or operator
 * @tparam MODE
 *   Threading policy
 *
 * @param C
 *   C A Tensor or Operator
 * @param A
 *   A A Tensor or Operator
 * @param B
 *   B A Tensor or Operator
 * @param exec
 *   Host executor
 * @param alpha
 *   Scalar multiplier to apply to operator A
 * @param beta
 *   Scalar multiplier to apply to operator C on input
 */
template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, ThreadsMode MODE>
void matmul_impl([[maybe_unused]] TensorTypeC C,
                 [[maybe_unused]] const TensorTypeA A,
                 [[maybe_unused]] const TensorTypeB B,
                 [[maybe_unused]] const HostExecutor<MODE> &exec,
                 [[maybe_unused]] float alpha = 1.0,
                 [[maybe_unused]] float beta = 0.0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(MATX_EN_CPU_MATMUL, matxInvalidExecutor, "Trying to run MatMul on host executor but host MatMul support is not configured");

#if MATX_EN_CPU_MATMUL
  constexpr auto is_c_complex = is_complex_v<typename TensorTypeC::value_type>;

  if constexpr (is_c_complex) {
    constexpr auto is_a_complex = is_complex_v<typename TensorTypeA::value_type>;
    constexpr auto is_b_complex = is_complex_v<typename TensorTypeB::value_type>;
    MATX_STATIC_ASSERT_STR((is_a_complex || is_b_complex), matxInvalidType, "If C is complex, then either A or B must be complex");
  }

  // promote A and B to the type of C
  auto A_ = as_type<typename TensorTypeC::value_type>(A);
  auto B_ = as_type<typename TensorTypeC::value_type>(B);

  MATX_STATIC_ASSERT_STR((detail::CompatibleGemmCBLASTypes<decltype(A_), decltype(B_), TensorTypeC>()), matxInvalidType,
      "Combination of A/B/C types are not supported for host MatMul");

  // cblas does not support operators and certain transpose modes.
  // Grab a suppported tensor here and copy in if necessary.
  auto c = getCBLASSupportedTensor(C);
  auto a = getCBLASSupportedTensor(A_);
  auto b = getCBLASSupportedTensor(B_);

  typedef decltype(c) ctype;
  typedef decltype(a) atype;
  typedef decltype(b) btype;

  if (!a.isSameView(A_)) {
    (a = A_).run(exec);
  }

  if (!b.isSameView(B_)) {
    (b = B_).run(exec);
  }

  if (beta != 0 && !c.isSameView(C)) {
    (c = C).run(exec);
  }

  // cblas does not allow transpose modes on C.  Thus we need to make sure that the right most dimension has a stride of 1.
  // Use the identity CT = BT * AT to do the transpose through the gemm automatically.  Note we only want to do this transpose if
  // the rightmost stride is !=1 or this function will be an infinite recursion.
  if (c.Stride(c.Rank() - 2) == 1 && c.Stride(c.Rank() - 1) > 1) { // column major check
    matmul_impl(transpose_matrix(c), transpose_matrix(b), transpose_matrix(a), exec, alpha, beta);
  } else {
    detail::matmul_dispatch(c, a, b, alpha, beta, exec);
  }

  // if c and C are not the same then we need to copy results out.
  if (!c.isSameView(C)) {
    (C = c).run(exec);
  }
#endif
}

}; // end namespace matx
