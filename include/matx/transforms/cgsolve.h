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

#include "matx/transforms/reduce.h"
#include "matx/core/nvtx.h"
#include "matx/core/type_utils.h"
#include "matx/operators/all.h"
#include "matx/operators/if.h"
#include "matx/operators/clone.h"

namespace matx
{
  /**
   * Performs a complex gradient solve on a square matrix.  
   *
   * @param X
   *   Tensor To Solve out
   * @param A
   *   Tensor A
   * @param B
   *   Tensor B 
   * @param tol
   *   tolerance to solve to  
   * @param max_iters
   *   max iterations for solve
   * @param stream
   *   cuda Stream to execute on
   *
   */
  template <typename XType, typename AType, typename BType>
    __MATX_INLINE__ void cgsolve_impl(XType X, AType A, BType B, double tol=1e-6, int max_iters=4, cudaStream_t stream=0)
    {
      using value_type = typename XType::value_type;
      const int VRANK = XType::Rank();
      const int SRANK = XType::Rank() - 1;
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
      // TODO clone A,b if necessary.  x cannot be cloned
      
      MATX_ASSERT_STR(A.Rank() -1 == X.Rank(), matxInvalidDim, "cgsolve:  A rank must be one larger than X rank");
      MATX_ASSERT_STR(X.Rank() == B.Rank(), matxInvalidDim, "cgsole: X rank and B rank must match");

      cudaStream_t d2h;
      cudaEvent_t event;
      

      if(tol>0.0f) {
        cudaStreamCreateWithFlags(&d2h,cudaStreamNonBlocking);
        cudaEventCreate(&event);
      }

      int converged_host = false;

      // Construct 3 temporary vectors
      auto r0 = make_tensor<value_type>(X.Shape(),  MATX_ASYNC_DEVICE_MEMORY, stream);
      auto p = make_tensor<value_type>(X.Shape(),  MATX_ASYNC_DEVICE_MEMORY, stream);
      auto Ap = make_tensor<value_type>(X.Shape(),  MATX_ASYNC_DEVICE_MEMORY, stream);
  
      // create aliases to reuse memory
      auto r1 = r0;
      
      // Drop last dim of X
      cuda::std::array<index_t, SRANK> scalar_shape;
      for(int i = 0 ; i < SRANK; i++) {
        scalar_shape[i] = X.Size(i);
      }
      value_type N = value_type(X.Size(SRANK));

      // Construct temporary scalars
      auto r0r0 = make_tensor<value_type>(scalar_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
      auto r1r1 = make_tensor<value_type>(scalar_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
      auto pAp = make_tensor<value_type>(scalar_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
      auto norm = make_tensor<value_type>(scalar_shape, MATX_ASYNC_DEVICE_MEMORY, stream);

      auto converged = make_tensor<int>({}, MATX_ASYNC_DEVICE_MEMORY, stream);

      // create aliases to reuse memory
      //auto b = a;
      
      // batched scalars will need to be cloned back to vector to make math work below
      // auto cloning clones along the left most dims and we need right most
      cuda::std::array<index_t, VRANK> clone_shape;
      for(int i = 0 ; i < SRANK; i++) {
        clone_shape[i] = matxKeepDim;
      }
      clone_shape[SRANK] = X.Size(SRANK);

      auto r0r0c = clone<VRANK>(r0r0, clone_shape);
      auto r1r1c = clone<VRANK>(r1r1, clone_shape);
      auto pApc = clone<VRANK>(pAp, clone_shape);
    
      // A*X
      (Ap = matvec(A, X)).run(stream);
      // r0 = B - A*X   
      // p = r0 
      (p = r0 = B - Ap).run(stream);  
      
      (r0r0 = sum(r0*r0)).run(stream);
      
      if(tol>0.0f) {
        (converged = matx::all(as_int(sqrt(r0r0) < tol))).run(stream);
      
        cudaEventRecord(event, stream);
        cudaStreamWaitEvent(d2h, event);
      }

      int i;
      for (i = 0 ; i < max_iters; i++) {
        // Ap = matvec(A, p) 
        (Ap = matvec(A, p)).run(stream);

        // pAp = dot(p,Ap)  
        (pAp = sum(p*Ap)).run(stream);
        
        // if pAp is zero then we have exactly numerically converged.
        // However, this is batched so we may iterate more.  Iterating
        // further will result in NANs so we will guard with IFs.

        // Fuse these into 1 kernel with comma op
        // r1 = r0 - a * Ap 
        // x = x + a * p
        auto updateOp = ( r1 = r0 - (r0r0c/pApc) * Ap,
             X = X + (r0r0c/pApc) * p);

        (IF( pApc != value_type(0), updateOp)).run(stream);
        
        // r1r1 = dot(r1, r1)
        (r1r1 = sum(r1*r1)).run(stream);
        
        if(tol>0.0f) {
          // copy convergence criteria to host.  
          // This is in unpinned memory and cannot on most systems run asynchronously.  
          // We do this here to hide the copy/sync behind prior launch latency/execution.
          cudaMemcpyAsync(&converged_host, converged.Data(), sizeof(int), cudaMemcpyDeviceToHost, d2h);
          cudaStreamSynchronize(d2h);

          if(converged_host == true) {
            break;
          }

          (converged = matx::all(as_int(sqrt(r1r1) < tol))).run(stream);
        
          cudaEventRecord(event, stream);
          cudaStreamWaitEvent(d2h, event);
        }
        
        // p = r1 + b * p 
        auto updateP = ( p = r1 + (r1r1c/r0r0c) * p);
        (IF( pApc != value_type(0), updateP)).run(stream);

        // Advance residual
        swap(r0r0, r1r1);  
        swap(r0r0c, r1r1c);
      }

      if(tol>0.0f) {
        cudaEventDestroy(event);
        cudaStreamDestroy(d2h);
      }
    }
  
} // end namespace matx
