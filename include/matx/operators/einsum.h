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


#include "matx/core/type_utils.h"
#include "matx/core/nvtx.h"
#include "matx/core/operator_utils.h"
#include "matx/transforms/einsum.h"

namespace matx
{


/* Einsum operator does not allow async allocations yet since that requires determining the output rank */
namespace detail {
  template<typename... OpA>
  class EinsumOp : public BaseOp<EinsumOp<OpA...>>
  {
    private:
      cuda::std::tuple<typename detail::base_type_t<OpA> ...> a_;
      std::string subscripts_;

    public:
      using matxop = bool;
      using value_type = void;
      using matx_transform_op = bool;
      using einsum_xform_op = bool;

      __MATX_INLINE__ std::string str() const { return "einsum()"; }
      __MATX_INLINE__ EinsumOp(const std::string &subscripts, const OpA&... ops) : subscripts_(subscripts), a_(ops...) { };

      // This should never be called
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const = delete;

      template <typename Out, typename Executor>
      void Exec(Out &&out, Executor &&ex) const {
        static_assert(is_cuda_executor_v<Executor>, "einsum() only supports the CUDA executor currently");   

        cuda::std::apply([&](auto... args) {
          ::matx::cutensor::einsum_impl(cuda::std::get<0>(out), subscripts_, ex.getStream(), args...);
        }, a_);
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return matxNoRank;
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        // Maybe do something here later if we take operators as input        
      }

      template <int I, typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (I < sizeof...(OpA)-1) {
          if constexpr (is_matx_op<cuda::std::tuple_element_t<I,cuda::std::tuple<OpA...>>>()) {
            cuda::std::get<I>(a_).PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            PreRun<I+1, ShapeType, Executor>(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        } else if constexpr (I == sizeof...(OpA)-1) {
          if constexpr (is_matx_op<cuda::std::tuple_element_t<I,cuda::std::tuple<OpA...>>>()) {
            cuda::std::get<I>(a_).PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            // This was the last ops_ element, so stop recursion
          }
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        PreRun<0, ShapeType, Executor>(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
      }          


      // Size is not relevant in einsum() since there are multiple return values and it
      // is not allowed to be called in larger expressions
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return 0;
      }

  };
}

namespace cutensor {
  /**
   * @brief Evaluates the Einstein summation on the operands
   * 
   * einsum() is a multi-purpose tool capable of performing various operations on tensors in a compact
   * syntax. A non-exhaustive list of operations are: tensor contractions, GEMMs, dot products, and tranposes.
   * Because einsum is extremely powerful, not all features are supported or tested in MatX yet. Currently only
   * tensor contractions are tested. Other operations may work, but they're not tested yet.
   * 
   * MatX uses a syntax very similar to NumPy's einsum syntax:
   * https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
   * 
   * Ellipses are not supported yet, but a variadic list of tensors for contraction is supported. The output
   * operator '->' is required in MatX currently, and serves to provide error checking on the output tensor size.
   *  
   * 
   * @tparam InT Types of input operators
   * @param subscripts String containing Einstein notation of operation to perform
   * @param ops List of input operators
   */
  template <typename... InT>
  __MATX_INLINE__ auto einsum(const std::string &subscripts, const InT&... ops) {
    return detail::EinsumOp(subscripts, ops...);
  }
}

}