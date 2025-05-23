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

#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{

  /**
   * Returns cross product of two operators when the last dimensions are 2 or 3
   */
  namespace detail {
    template <typename OpA, typename OpB>
    class CrossOp : public BaseOp<CrossOp<OpA, OpB>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        typename detail::base_type_t<OpB> b_;

        static constexpr int32_t out_rank = cuda::std::max(OpA::Rank(), OpB::Rank());
        static constexpr int32_t min_rank = cuda::std::min(OpA::Rank(), OpB::Rank());

        cuda::std::array<index_t, out_rank> out_dims_;

        //helpers to simplify later checks
        bool isA2D_ = a_.Size(a_.Rank() - 1) == 2 ? true : false;
        bool isB2D_ = b_.Size(b_.Rank() - 1) == 2 ? true : false;

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;

        __MATX_INLINE__ std::string str() const { return "cross()"; }
        __MATX_INLINE__ CrossOp(const OpA &A, const OpB &B) : a_(A), b_(B) {
          MATX_STATIC_ASSERT_STR(OpA::Rank() >= 1 && OpB::Rank() >= 1, matxInvalidDim, "Operators to cross() must have rank GTE one.");

          //dims other than the last are batched, so count R-->L, beginning one-left of the right-most dim
          for (int32_t i = 1; i < min_rank; i++) {
            MATX_ASSERT_STR(a_.Size(a_.Rank() - 1 - i) == b_.Size(b_.Rank() - 1 - i), matxInvalidSize, "A and B tensors must match batch sizes.");
          }

          MATX_ASSERT_STR(a_.Size(a_.Rank() - 1) == 3 || a_.Size(a_.Rank() - 1) == 2, matxInvalidSize, "Last dimension of A must have size 2 or 3.")
          MATX_ASSERT_STR(b_.Size(b_.Rank() - 1) == 3 || b_.Size(b_.Rank() - 1) == 2, matxInvalidSize, "Last dimension of B must have size 2 or 3.")
        
          for (int32_t i = 0; i < out_rank - 1; i++) {
            if (i < a_.Rank()){
              out_dims_[i] = a_.Size(i);
            }
            else{
              out_dims_[i] = b_.Size(i);
            }
          }
          
          //mimic NumPy cross as closely as possible
          if(isA2D_ && isB2D_){
            out_dims_[out_dims_.size() - 1] = 1;
          }
          else{
            out_dims_[out_dims_.size() - 1] = 3;
          }
        };

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          // Only support EPT == ONE for now
          if constexpr (EPT == ElementsPerThread::ONE) {
            cuda::std::array idx{indices...};
              auto idxOut = idx[idx.size() - 1];

            //create references to individual slices for ease of notation
            cuda::std::array idx0{idx};
            cuda::std::array idx1{idx};
            cuda::std::array idx2{idx};

            idx0[idx0.size() - 1] = 0LL;
            idx1[idx1.size() - 1] = 1LL;
            idx2[idx2.size() - 1] = 2LL;

            auto a0 = get_value<EPT>(a_, idx0);
            auto a1 = get_value<EPT>(a_, idx1);
            
            auto b0 = get_value<EPT>(b_, idx0);
            auto b1 = get_value<EPT>(b_, idx1);

            //lots of if-elses, but similar to numpy implementation
            
            if (idxOut == 2 || (isA2D_ && isB2D_)){
              return a0 * b1 - a1 * b0;
            }

            if (!isA2D_ && !isB2D_){
              auto a2 = get_value<EPT>(a_, idx2);
              auto b2 = get_value<EPT>(b_, idx2);
              if (idxOut == 0){
                  return a1 * b2 - a2 * b1;
              }
              //idxOut == 1
              return a2 * b0 - a0 * b2;
              
            }
            else if (isA2D_ && !isB2D_){
              auto b2 = get_value<EPT>(b_, idx2);
              if (idxOut == 0){
                  return a1 * b2;
              }
              //idxOut == 1
              return -a0 * b2;
            }
            else{// !isA2D_ && isB2D_, case of both 2D are covered in the first if statement
              auto a2 = get_value<EPT>(a_, idx2);
              if (idxOut == 0){
                  return -a2 * b1;
              }
              //idxOut == 1
              return a2 * b0;
            }
          } else {
            return Vector<value_type, static_cast<size_t>(EPT)>();
          }
          
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            return ElementsPerThread::ONE;
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
            detail::get_operator_capability<Cap>(a_),
              detail::get_operator_capability<Cap>(b_)
            );
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return out_rank;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          return out_dims_[dim];
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_matx_op<OpB>()) {
            b_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }          
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }     

          if constexpr (is_matx_op<OpB>()) {
            b_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          } 
        }    
    };
  }


  /**
   * @brief Evaluate a cross product
   *  
   * @tparam OpA Type of input tensor 1
   * @tparam OpB Type of input tensor 2
   * @param A Input tensor 1
   * @param B Input tensor 2
   * @return cross operator 
   */
  template <typename OpA, typename OpB>
  __MATX_INLINE__ auto cross(const OpA &A, const OpB &B) {
    return detail::CrossOp(A, B);
  }
} // end namespace matx
