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
   * Run an outer product on two vectors
   *
   * Performs an outer product where each element of vector A is multiplied by each
   * element of vector B to create a new matrix C. If A is length M and B is length M,
   * C is length NxM. A and B can be batched, where each dimension other than the
   * right-most is a batching dimension.
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
__MATX_INLINE__ void outer_impl(TensorTypeC C, const TensorTypeA A,
            const TensorTypeB B,
            const Executor &exec,
            float alpha = 1.0, float beta = 0.0)
{
  MATX_STATIC_ASSERT_STR(TensorTypeA::Rank() == TensorTypeB::Rank(), matxInvalidDim, "outer: A and B ranks must match");
  MATX_STATIC_ASSERT_STR(C.Rank() == A.Rank() + 1, matxInvalidDim,
    "outer: C tensor must be 1 rank higher than A");
  MATX_STATIC_ASSERT_STR(C.Rank() == B.Rank() + 1, matxInvalidDim,
    "outer: B tensor must be 1 rank higher than A");    
  MATX_ASSERT_STR(C.Size(C.Rank() - 2) == A.Size(A.Rank() - 1), matxInvalidSize,
      "outer: second-to-last dimension of C must match last dimension of A");
  MATX_ASSERT_STR(C.Size(C.Rank() - 1) == B.Size(B.Rank() - 1), matxInvalidSize,
      "outer: last dimension of C must match last dimension of B");

  cuda::std::array<index_t, TensorTypeA::Rank() + 1> ac;
  cuda::std::array<index_t, TensorTypeB::Rank() + 1> bc;

  ac.fill(matxKeepDim);
  bc.fill(matxKeepDim);

  for (int r = 0; r < cuda::std::min(A.Rank(), B.Rank()) - 1; r++) {
    MATX_ASSERT_STR(A.Size(r) == B.Size(r), matxInvalidSize, "A and B tensors must match batch sizes");
  }

  ac[ac.size() - 1] = 1;
  bc[bc.size() - 2] = 1;

  auto act = clone<TensorTypeA::Rank() + 1>(A, ac);
  auto bct = clone<TensorTypeB::Rank() + 1>(B, bc);

  if constexpr (is_cuda_executor_v<Executor>) {
    matmul_impl<decltype(C), decltype(act), decltype(bct), PROV>(C, act, bct, exec, alpha, beta);
  } else {
    matmul_impl<decltype(C), decltype(act), decltype(bct)>(C, act, bct, exec, alpha, beta);
  }
}

};
