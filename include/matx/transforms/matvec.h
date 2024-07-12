////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// COpBright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above cOpBright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above cOpBright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the cOpBright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace matx {

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
 *    Provider type chosen from MatMulCUDAProvider_t type
 *
 * @param C
 *   C output tensor or operator
 * @param A
 *   A input tensor or operator
 * @param B
 *   B input tensor or operator
 * @param exec
 *   CUDA executor
 * @param alpha
 *   Scalar multiplier to apply to operator A
 * @param beta
 *   Scalar multiplier to apply to operator C on input
 */
template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB, typename Executor,
          MatMulCUDAProvider_t PROV = PROVIDER_TYPE_CUBLASLT>
__MATX_INLINE__ void matvec_impl(TensorTypeC C, const TensorTypeA A,
            const TensorTypeB B,
            const Executor &exec,
            float alpha = 1.0, float beta = 0.0)
{
  MATX_STATIC_ASSERT(TensorTypeA::Rank() == TensorTypeB::Rank()+1, "matvec: A rank must be one larger than B rank");
  
  MATX_ASSERT_STR(C.Size(TensorTypeB::Rank()-1) == A.Size(TensorTypeA::Rank()-2), matxInvalidDim, "matvec: C last size must match A second last Size");
  MATX_ASSERT_STR(B.Size(TensorTypeB::Rank()-1) == A.Size(TensorTypeA::Rank()-1), matxInvalidDim, "matvec: B last size must match A last size");

  // need to clone c and b 1 along inner dim to use cublas
  cuda::std::array<index_t, TensorTypeC::Rank()+1> shape;
  for(int i = 0; i < TensorTypeC::Rank(); i++) {
    shape[i] = matxKeepDim;
  }
  // clone last dim by 1 to create an Nx1 matrix
  shape[TensorTypeC::Rank()]=1;

  auto c = clone<TensorTypeC::Rank()+1>(C, shape);
  auto b = clone<TensorTypeB::Rank()+1>(B, shape);

  if constexpr (is_cuda_executor_v<Executor>) {
    matmul_impl<decltype(c), decltype(A), decltype(b), PROV>(c, A, b, exec, alpha, beta);
  }
  else {
    matmul_impl<decltype(c), decltype(A), decltype(b)>(c, A, b, exec, alpha, beta);
  }
}

};