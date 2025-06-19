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
#ifndef __CUDACC_RTC__

#pragma once

#include <source_location>
#include "matx/core/iterator.h"
#include "matx/core/type_utils.h"
#include "matx/operators/collapse.h"

namespace matx {

  template <bool ConvertType, typename Func, typename OutputOp, typename InputOp, typename BeginIter, typename EndIter>
  __MATX_HOST__ __MATX_INLINE__ auto ReduceOutput(Func &&func, OutputOp &&out, InputOp &&in, BeginIter &&bi, EndIter &&ei) {
    if constexpr (remove_cvref_t<decltype(out)>::Rank() <= 1 && is_tensor_view_v<OutputOp>) {
      if (out.IsContiguous()) {
        if constexpr(ConvertType) {   
          return func(  in, 
                        reinterpret_cast<detail::convert_matx_type_t<typename remove_cvref_t<OutputOp>::value_type> *>(out.Data()), 
                        bi, 
                        ei);
        }
        else {
          return func(  in, 
                        reinterpret_cast<typename remove_cvref_t<OutputOp>::value_type *>(out.Data()), 
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
                                            reinterpret_cast<detail::convert_matx_type_t<typename remove_cvref_t<InputOp>::value_type> *>(in_base.Data()), 
                                            BeginOffset{in_base}, 
                                            EndOffset{in_base});
        }
        else {
          return ReduceOutput<ConvertType>( std::forward<Func>(func), 
                                            std::forward<OutputOp>(out), 
                                            reinterpret_cast<typename remove_cvref_t<InputOp>::value_type *>(in_base.Data()), 
                                            BeginOffset{in_base}, 
                                            EndOffset{in_base});
        }
      }
    }

    // Collapse the right-most dimensions by the difference in ranks for the reduction dimension,
    // then collapse the left size by the output rank to get the batch dimensions  
    auto collapsed = matx::lcollapse<remove_cvref_t<decltype(out)>::Rank()>(rcollapse<remove_cvref_t<decltype(in)>::Rank() - 
                                                                                      remove_cvref_t<decltype(out)>::Rank()>(in_base));
    const auto &iter = matx::RandomOperatorIterator<decltype(collapsed), ConvertType>{collapsed};
    return ReduceOutput<ConvertType>(std::forward<Func>(func), std::forward<OutputOp>(out), iter, BeginOffset{iter}, EndOffset{iter});   
  } 

  template <typename Func, typename OutputOp, typename InputOp>
  __MATX_HOST__ __MATX_INLINE__ auto ReduceInputNoConvert(Func &&func, OutputOp &&out, InputOp &&in) {
    return ReduceInput<Func, OutputOp, InputOp, false>(std::forward<Func>(func), std::forward<OutputOp>(out), std::forward<InputOp>(in));
  }

  constexpr bool RankGTE(int32_t rank1, int32_t rank2) {
    return rank1 >= rank2 || rank1 == matxNoRank;
  }

  constexpr bool RankGT(int32_t rank1, int32_t rank2) {
    return rank1 > rank2 || rank1 == matxNoRank;
  }

  template <typename Op>
  cuda::std::array<index_t, Op::Rank()> Shape(const Op &op) {
    cuda::std::array<index_t, Op::Rank()> shape;
    for (int r = 0; r < Op::Rank(); r++) {
      shape[r] = op.Size(r);
    }

    return shape;
  }
  namespace detail {
    // Used inside of transforms to allocate temporary output
    template <typename TensorType, typename Executor, typename ShapeType> 
    __MATX_HOST__ __MATX_INLINE__ void AllocateTempTensor(TensorType &tensor, Executor &&ex, ShapeType &&shape, typename TensorType::value_type **ptr) {
      const auto ttl_size = cuda::std::accumulate(shape.begin(), shape.end(), static_cast<index_t>(1),
                                  std::multiplies<index_t>()) * sizeof(typename TensorType::value_type);      
      if constexpr (is_cuda_executor_v<Executor>) {
        matxAlloc((void**)ptr, ttl_size, MATX_ASYNC_DEVICE_MEMORY, ex.getStream());
        make_tensor(tensor, *ptr, shape);
      }
      else {        
        matxAlloc((void**)ptr, ttl_size, MATX_HOST_MEMORY);
        make_tensor(tensor, *ptr, shape);        
      }  
    }

    template <typename Op, typename ValidFunc>
    __MATX_INLINE__ auto GetSupportedTensor(const Op &in, const ValidFunc &fn, matxMemorySpace_t space, cudaStream_t stream = 0) {
      if constexpr (is_matx_transform_op<Op>()) {
        // We can assume that if a transform is passed to the input then PreRun has already completed
        // on the transform and we can use the internal pointer
        return make_tensor<typename Op::value_type>(in.Data(), Shape(in));
      }
      else if constexpr (!is_tensor_view_v<Op>) {
        return make_tensor<typename Op::value_type>(in.Shape(), space, stream);
      }   
      else {
        bool supported = fn();

        if(supported) {
          return make_tensor<typename Op::value_type>(in.Data(), in.Descriptor());
        } else {
          return make_tensor<typename Op::value_type>(in.Shape(), space, stream);
        }
      }      
    }

    template <typename CapType, typename T, typename Func>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto ApplyGeneratorVecFunc(const Func &func, index_t index) {
      if constexpr (CapType::ept == ElementsPerThread::ONE) {
        return func(index);
      } else {
        Vector<T, static_cast<index_t>(CapType::ept)> result;
        #pragma unroll
        for (int i = 0; i < static_cast<index_t>(CapType::ept); i++) {
          result.data[i] = func(index * static_cast<index_t>(CapType::ept) + i);
        }
        return result;
      }
    }

    template <typename CapType, typename OutType, typename Func, typename... Vals>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto ApplyVecFunc(const Func &func, const Vals &...vals) {
      if constexpr (CapType::ept == ElementsPerThread::ONE) {
        return func(vals...);
      } else {
        Vector<OutType, static_cast<index_t>(CapType::ept)> result;
        #pragma unroll
        for (int i = 0; i < static_cast<index_t>(CapType::ept); i++) {
          result.data[i] = func(vals.data[i]...);
        }
        return result;
      }
    }

  }
}; 
#endif

// namespace matx{
//   namespace detail {
//     template<unsigned N>
//     struct reader {
//         #ifdef __GNUC__
//         #pragma GCC diagnostic push
//         #pragma GCC diagnostic ignored "-Wnon-template-friend"
//         #endif
//         #ifdef __NVCC__
//         #pragma nv_diag_suppress 1301
//         #endif
//         friend auto counted_flag(reader<N>);
//         #ifdef __NVCC__
//         #pragma nv_diag_default 1301
//         #endif
//         #ifdef __GNUC__
//         #pragma GCC diagnostic pop
//         #endif
//     };


//     template<unsigned N>
//     struct setter {
//         friend auto counted_flag(reader<N>) {}

//         static constexpr unsigned n = N;
//     };


//     template<
//         auto Tag,
//         unsigned NextVal = 0
//     >
//     [[nodiscard]]
//     consteval auto counter_impl() {
//         constexpr bool counted_past_value = requires(reader<NextVal> r) {
//             counted_flag(r);
//         };

//         if constexpr (counted_past_value) {
//             return counter_impl<Tag, NextVal + 1>();
//         }
//         else {
//             setter<NextVal> s;
//             return s.n;
//         }
//     }


//     template<
//         auto Tag = []{},
//         auto Val = counter_impl<Tag>()
//     >
//     constexpr auto counter = Val;

//       constexpr auto get_counter() {
//         constexpr auto count = counter<>;
//         return count;
//       }
//     }  
// }

#include <string_view>

namespace matx {
  namespace detail {

template <auto>
struct UniqueTag {};

template <typename T>
constexpr std::string_view get_type_name() {
#if defined(__clang__) || defined(__GNUC__)
    return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    return __FUNCSIG__;
#else
#error "Compiler not supported for type name retrieval"
#endif
}

    // FNV-1a compile-time hash
    constexpr uint64_t fnv1a_64(const std::string_view str) {
        uint64_t hash = 0xcbf29ce484222325;
        const uint64_t prime = 0x100000001b3;

        for (char c : str) {
            hash ^= static_cast<uint64_t>(c);
            hash *= prime;
        }
        return hash;
    }
  }
}