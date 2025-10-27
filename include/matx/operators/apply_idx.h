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
#include "matx/operators/base_operator.h"

namespace matx
{
  /**
   * ApplyIdxOp applies a custom lambda function to one or more input operators,
   * passing the indices as a cuda::std::array along with the operators themselves.
   * The lambda function is called for each element during operator evaluation.
   * The rank of the operator matches the rank of the first input operator, and
   * the size is also taken from the first input operator.
   */
  namespace detail {
    template <typename Func, typename... Ops>
    class ApplyIdxOp : public BaseOp<ApplyIdxOp<Func, Ops...>>
    {
      public:
        using matxop = bool;
        
        // Deduce value_type from the lambda function's return type
        using first_op_type = cuda::std::tuple_element_t<0, cuda::std::tuple<Ops...>>;
        static constexpr int RANK = first_op_type::Rank();
        using value_type = decltype(cuda::std::declval<Func>()(
            cuda::std::declval<cuda::std::array<index_t, RANK>>(),
            cuda::std::declval<Ops>()...));
        using self_type = ApplyIdxOp<Func, Ops...>;

        __MATX_INLINE__ std::string str() const { return "apply_idx()"; }

        __MATX_INLINE__ ApplyIdxOp(Func func, const Ops&... ops) : func_(func), ops_(detail::base_type_t<Ops>(ops)...)
        {
          static_assert(sizeof...(Ops) > 0, "ApplyIdxOp requires at least one input operator");
          
          // Initialize sizes from the first operator
          constexpr int rank = Rank();
          for (int i = 0; i < rank; i++) {
            sizes_[i] = cuda::std::get<0>(ops_).Size(i);
          }
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return apply_impl<CapType>(cuda::std::index_sequence_for<Ops...>{}, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return 
                combine_capabilities<Cap>(my_cap, get_combined_ops_capability<Cap>(in, ops_));
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, get_combined_ops_capability<Cap>(in, ops_));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          pre_run_impl(cuda::std::forward<ShapeType>(shape), cuda::std::forward<Executor>(ex), 
                       cuda::std::index_sequence_for<Ops...>{});
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          post_run_impl(cuda::std::forward<ShapeType>(shape), cuda::std::forward<Executor>(ex), 
                        cuda::std::index_sequence_for<Ops...>{});
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return first_op_type::Rank();
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          return sizes_[dim];
        }

        ~ApplyIdxOp() = default;
        ApplyIdxOp(const ApplyIdxOp &rhs) = default;

      private:
        Func func_;
        cuda::std::tuple<typename detail::base_type_t<Ops>...> ops_;
        cuda::std::array<index_t, first_op_type::Rank()> sizes_;
        
        // Helper to apply the lambda function with indices and operators
        template <typename CapType, size_t... Is, typename... Indices>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) apply_impl(
            cuda::std::index_sequence<Is...>, Indices... indices) const
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            // Scalar case: single element access
            cuda::std::array<index_t, sizeof...(Indices)> idx_array{static_cast<index_t>(indices)...};
            return func_(idx_array, cuda::std::get<Is>(ops_)...);
          } else {
            // Vector case: multiple elements per thread
            // Deduce the result element type by calling func_ once
            // Note that this path is disabled for now since apply_idx can't swizzle with vectors
            cuda::std::array<index_t, sizeof...(Indices)> idx_array{static_cast<index_t>(indices)...};
            using result_element_type = decltype(func_(idx_array, cuda::std::get<Is>(ops_)...));
            Vector<result_element_type, static_cast<int>(CapType::ept)> result;

            // Adjust the last index to the EPT value. The user's lambda function does not know whether we're
            // using vectorization or not so we need to adjust the index ourselves.
            idx_array[sizeof...(Indices) - 1] *= static_cast<int>(CapType::ept);
            
            // Unroll loop to call func_ on each element of the vector
            // For multi-dimensional indices, only the last index varies
            MATX_LOOP_UNROLL
            for (int i = 0; i < static_cast<int>(CapType::ept); i++) {
              result.data[i] = func_(idx_array, cuda::std::get<Is>(ops_)...);
              if constexpr (sizeof...(Indices) > 0) {
                idx_array[sizeof...(Indices) - 1]++;
              }
            }
            return result;
          }
        }

        // Helper to call PreRun on all operators
        template <typename ShapeType, typename Executor, size_t... Is>
        __MATX_INLINE__ void pre_run_impl(ShapeType &&shape, Executor &&ex, 
                                          cuda::std::index_sequence<Is...>) const noexcept
        {
          ((is_matx_op<Ops>() && (cuda::std::get<Is>(ops_).PreRun(cuda::std::forward<ShapeType>(shape), 
                                                                   cuda::std::forward<Executor>(ex)), true)), ...);
        }

        // Helper to call PostRun on all operators
        template <typename ShapeType, typename Executor, size_t... Is>
        __MATX_INLINE__ void post_run_impl(ShapeType &&shape, Executor &&ex, 
                                           cuda::std::index_sequence<Is...>) const noexcept
        {
          ((is_matx_op<Ops>() && (cuda::std::get<Is>(ops_).PostRun(cuda::std::forward<ShapeType>(shape), 
                                                                    cuda::std::forward<Executor>(ex)), true)), ...);
        }

        // Helper to combine capabilities from all operators
        template <OperatorCapability Cap, typename CapType, typename OpsTuple, size_t... Is>
        __MATX_INLINE__ __MATX_HOST__ auto combine_capabilities_tuple(
            CapType self_has_cap, const OpsTuple& ops_tuple, cuda::std::index_sequence<Is...>) const
        {
          return combine_capabilities<Cap>(self_has_cap, 
                                           detail::get_operator_capability<Cap>(cuda::std::get<Is>(ops_tuple))...);
        }
    };
  }

  /**
   * @brief Apply a custom lambda function or functor to one or more operators with index access
   * 
   * The apply_idx operator allows applying a custom lambda function or functor to one or more input
   * operators, where the lambda receives the current indices as a cuda::std::array along with
   * the operators themselves. This allows the lambda to access elements at any position, not just
   * the current element position.
   * 
   * The resulting operator has the same rank as the first input operator, and its
   * size matches the size of the first input operator. The value type is deduced from
   * the return type of the lambda function.
   * 
   * @tparam Func Lambda function or functor type
   * @tparam Ops Input operator types (one or more operators)
   * 
   * @param func Lambda function or functor to apply. Can be __host__, __device__, or both.
   *             The function signature should accept a cuda::std::array<index_t, RANK> followed
   *             by the input operators themselves (not their values).
   *             Note: Inline __device__ lambdas work in regular code (e.g., main()) but NOT in
   *             Google Test fixtures due to private method restrictions. Use functors for tests.
   *             Requires --extended-lambda compiler flag.
   * @param ops Input operators (one or more)
   * 
   * @return ApplyIdxOp operator that applies the function element-wise
   * 
   * Example using an inline lambda (works in main(), not in test fixtures):
   * @code
   * auto t_in = make_tensor<float>({10});
   * auto t_out = make_tensor<float>({10});
   * 
   * auto stencil = [] __device__ (auto idx, auto op) {
   *   auto i = idx[0];
   *   // Access current and neighboring elements
   *   if (i == 0 || i == op.Size(0) - 1) return op(i);
   *   return (op(i-1) + op(i) + op(i+1)) / 3.0f;
   * };
   * (t_out = apply_idx(stencil, t_in)).run();
   * @endcode
   * 
   * Example using a functor (works everywhere including tests):
   * @code
   * struct StencilFunctor {
   *   template<typename Op>
   *   __host__ __device__ auto operator()(cuda::std::array<index_t, 1> idx, const Op& op) const {
   *     auto i = idx[0];
   *     if (i == 0 || i == op.Size(0) - 1) return op(i);
   *     return (op(i-1) + op(i) + op(i+1)) / 3.0f;
   *   }
   * };
   * auto t_in = make_tensor<float>({10});
   * auto t_out = make_tensor<float>({10});
   * (t_out = apply_idx(StencilFunctor{}, t_in)).run();
   * @endcode
   */
  template <typename Func, typename... Ops>
  auto __MATX_INLINE__ apply_idx(Func func, const Ops&... ops)
  {
    return detail::ApplyIdxOp<Func, Ops...>(func, ops...);
  }

} // end namespace matx


