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
   * ApplyOp applies a custom lambda function to one or more input operators.
   * The lambda function is called for each element during operator evaluation.
   * The rank of the operator matches the rank of the first input operator, and
   * the size is also taken from the first input operator.
   */
  namespace detail {
    template <typename Func, typename... Ops>
    class ApplyOp : public BaseOp<ApplyOp<Func, Ops...>>
    {
      public:
        using matxop = bool;
        
        // Deduce value_type from the lambda function's return type
        using first_op_type = cuda::std::tuple_element_t<0, cuda::std::tuple<Ops...>>;
        using value_type = decltype(cuda::std::declval<Func>()(cuda::std::declval<typename Ops::value_type>()...));
        using self_type = ApplyOp<Func, Ops...>;

        __MATX_INLINE__ std::string str() const { return "apply()"; }

        __MATX_INLINE__ ApplyOp(Func func, const Ops&... ops) : func_(func), ops_(ops...)
        {
          static_assert(sizeof...(Ops) > 0, "ApplyOp requires at least one input operator");
          
          // Initialize sizes from the first operator
          constexpr int rank = Rank();
          for (int i = 0; i < rank; i++) {
            sizes_[i] = cuda::std::get<0>(ops_).Size(i);
          }
        }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return apply_impl<EPT>(cuda::std::index_sequence_for<Ops...>{}, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities_tuple<Cap>(self_has_cap, ops_, cuda::std::index_sequence_for<Ops...>{});
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

        ~ApplyOp() = default;
        ApplyOp(const ApplyOp &rhs) = default;

      private:
        Func func_;
        cuda::std::tuple<typename detail::base_type_t<Ops>...> ops_;
        cuda::std::array<index_t, first_op_type::Rank()> sizes_;
        // Helper to apply the lambda function to all operators
        template <ElementsPerThread EPT, size_t... Is, typename... Indices>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) apply_impl(
            cuda::std::index_sequence<Is...>, Indices... indices) const
        {
          using out_type = decltype(cuda::std::get<0>(ops_).template operator()<EPT>(indices...));
          if constexpr (is_vector_v<out_type>) {
            // Each operator returns a vector, so call operator() once per operator to get the vectors
            auto op_results = cuda::std::make_tuple(cuda::std::get<Is>(ops_).template operator()<EPT>(indices...)...);
            
            // Deduce the result type by calling func_ on scalar elements
            using result_element_type = decltype(func_(cuda::std::get<Is>(op_results).data[0]...));
            Vector<result_element_type, static_cast<int>(EPT)> result;
            
            // Unroll loop to call func_ on each element of the vectors
            MATX_LOOP_UNROLL
            for (int i = 0; i < static_cast<int>(EPT); i++) {
              result.data[i] = func_(cuda::std::get<Is>(op_results).data[i]...);
            }
            return result;
          } else {
            return func_(cuda::std::get<Is>(ops_).template operator()<EPT>(indices...)...);
          }
        }

        // Helper to call PreRun on all operators
        template <typename ShapeType, typename Executor, size_t... Is>
        __MATX_INLINE__ void pre_run_impl(ShapeType &&shape, Executor &&ex, 
                                          cuda::std::index_sequence<Is...>) const noexcept
        {
          int dummy[] = {
            (is_matx_op<Ops>() ? 
              (cuda::std::get<Is>(ops_).PreRun(cuda::std::forward<ShapeType>(shape), 
                                               cuda::std::forward<Executor>(ex)), 0) : 0)...
          };
          (void)dummy;
        }

        // Helper to call PostRun on all operators
        template <typename ShapeType, typename Executor, size_t... Is>
        __MATX_INLINE__ void post_run_impl(ShapeType &&shape, Executor &&ex, 
                                           cuda::std::index_sequence<Is...>) const noexcept
        {
          int dummy[] = {
            (is_matx_op<Ops>() ? 
              (cuda::std::get<Is>(ops_).PostRun(cuda::std::forward<ShapeType>(shape), 
                                                cuda::std::forward<Executor>(ex)), 0) : 0)...
          };
          (void)dummy;
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
   * @brief Apply a custom lambda function or functor to one or more operators
   * 
   * The apply operator allows applying a custom lambda function or functor to one or more input
   * operators. The function is called for each element position, and receives
   * the values from all input operators at that position.
   * 
   * The resulting operator has the same rank as the first input operator, and its
   * size matches the size of the first input operator. The value type is deduced from
   * the return type of the lambda function.
   * 
   * @tparam Func Lambda function or functor type
   * @tparam Ops Input operator types (one or more operators)
   * 
   * @param func Lambda function or functor to apply. Can be __host__, __device__, or both.
   *             The function signature should accept value_type from each input operator.
   *             Note: Using __host__ __device__ lambdas requires the --extended-lambda compiler flag.
   *             For complex scenarios, consider using functors instead of lambdas.
   * @param ops Input operators (one or more)
   * 
   * @return ApplyOp operator that applies the function element-wise
   * 
   * Example using a lambda:
   * @code
   * auto t1 = make_tensor<float>({10, 10});
   * auto t2 = make_tensor<float>({10, 10});
   * auto result = make_tensor<float>({10, 10});
   * 
   * // Apply a custom function that adds and squares
   * auto my_func = [] __device__ (float a, float b) { return (a + b) * (a + b); };
   * (result = apply(my_func, t1, t2)).run();
   * @endcode
   * 
   * Example using a functor:
   * @code
   * struct SquareFunctor {
   *   template<typename T>
   *   __host__ __device__ auto operator()(T x) const { return x * x; }
   * };
   * 
   * auto t_in = make_tensor<float>({10});
   * auto t_out = make_tensor<float>({10});
   * (t_out = apply(SquareFunctor{}, t_in)).run();
   * @endcode
   */
  template <typename Func, typename... Ops>
  auto __MATX_INLINE__ apply(Func func, const Ops&... ops)
  {
    return detail::ApplyOp<Func, Ops...>(func, ops...);
  }

} // end namespace matx

