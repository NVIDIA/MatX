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

#include "matx/core/iterator.h"
#include "matx/core/type_utils.h"
#include "matx/operators/collapse.h"

namespace matx {

  template <bool ConvertType, typename Func, typename OutputOp, typename InputOp, typename BeginIter, typename EndIter>
  __MATX_HOST__ __MATX_INLINE__ auto ReduceOutput(Func &&func, OutputOp &&out, InputOp &&in, BeginIter &&bi, EndIter &&ei) {
    if constexpr (out.Rank() <= 1 && is_tensor_view_v<OutputOp>) {
      if (out.IsContiguous()) {
        if constexpr(ConvertType) {
          return func(  in, 
                        reinterpret_cast<detail::convert_matx_type_t<typename remove_cvref_t<OutputOp>::scalar_type> *>(out.Data()), 
                        bi, 
                        ei);
        }
        else {
          return func(  in, 
                        reinterpret_cast<typename remove_cvref_t<OutputOp>::scalar_type *>(out.Data()), 
                        bi, 
                        ei);
        }
      }
    }
    
    detail::base_type_t<OutputOp> out_base = out;
    auto iter = RandomOperatorOutputIterator<decltype(out_base), ConvertType>{out_base};
    return func(in, iter, bi, ei);
  }  

  template <typename Func, typename OutputOp, typename InputOp, bool ConvertType = true>
  __MATX_HOST__ __MATX_INLINE__ auto ReduceInput(Func &&func, OutputOp &&out, InputOp &&in) {
    typename detail::base_type_t<InputOp> in_base = in;
    if constexpr (in_base.Rank() < 2 && is_tensor_view_v<InputOp>) {
      if (in_base.IsContiguous()) {
        if constexpr (ConvertType) {
          return ReduceOutput<ConvertType>( std::forward<Func>(func), 
                                            std::forward<OutputOp>(out), 
                                            reinterpret_cast<detail::convert_matx_type_t<typename remove_cvref_t<InputOp>::scalar_type> *>(in_base.Data()), 
                                            BeginOffset{in_base}, 
                                            EndOffset{in_base});
        }
        else {
          return ReduceOutput<ConvertType>( std::forward<Func>(func), 
                                            std::forward<OutputOp>(out), 
                                            reinterpret_cast<typename remove_cvref_t<InputOp>::scalar_type *>(in_base.Data()), 
                                            BeginOffset{in_base}, 
                                            EndOffset{in_base});
        }
      }
    }

    // Collapse the right-most dimensions by the difference in ranks for the reduction dimension,
    // then collapse the left size by the output rank to get the batch dimensions  
    auto collapsed = matx::lcollapse<out.Rank()>(rcollapse<in.Rank() - out.Rank()>(in_base));
    const auto &iter = matx::RandomOperatorIterator<decltype(collapsed), ConvertType>{collapsed};

  
    return ReduceOutput<ConvertType>(std::forward<Func>(func), std::forward<OutputOp>(out), iter, BeginOffset{iter}, EndOffset{iter});   
  } 

  template <typename Func, typename OutputOp, typename InputOp>
  __MATX_HOST__ __MATX_INLINE__ auto ReduceInputNoConvert(Func &&func, OutputOp &&out, InputOp &&in) {
    return ReduceInput<Func, OutputOp, InputOp, false>(std::forward<Func>(func), std::forward<OutputOp>(out), std::forward<InputOp>(in));
  }
}; 
