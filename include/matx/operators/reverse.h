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
   * Reverse the indexing of a View or operator on a single dimension
   *
   * Allows a view or operator to be indexed in reverse order. After applying the
   * operator, index 0 is the last element in the selected dimension, index 1 is
   * second to last, etc.
   *
   */
  namespace detail {
    template <int DIM, typename T1>
      class ReverseOp : public BaseOp<ReverseOp<DIM, T1>>
    {
      private:
        mutable typename detail::base_type_t<T1> op_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;
        using value_type = typename T1::value_type;
        using self_type = ReverseOp<DIM, T1>;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITReverse_dim{}", DIM);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < Rank(); ++i) {
            out_dims_[i] = Size(i);
          }
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = typename T::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int DIM_ = {};\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      if constexpr (Rank_ == 0) {{\n"
                "        return op_.template operator()<CapType>();\n"
                "      }} else {{\n"
                "        cuda::std::array idx{{indices...}};\n"
                "        idx[DIM_] = out_dims_[DIM_] - idx[DIM_] - 1;\n"
                "        return get_value<CapType>(op_, idx);\n"
                "      }}\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, DIM, Rank(), detail::array_to_string(out_dims_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "reverse(" + op_.str() + ")"; }

        __MATX_INLINE__ ReverseOp(const T1 &op) : op_(op){
          MATX_LOG_TRACE("{} constructor: rank={}", str(), DIM);
        };

        template <typename CapType, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            if constexpr (Rank() == 0) {
              return op.template operator()<CapType>();
            } 
            else {
              cuda::std::array idx{indices...};
              idx[DIM] = op.Size(DIM) - idx[DIM] - 1;
              return get_value<CapType>(cuda::std::forward<Op>(op), idx);
            }            
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return get_impl<CapType>(cuda::std::as_const(op_), indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return op_.Size(dim);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }   

        ~ReverseOp() = default;
        ReverseOp(const ReverseOp &rhs) = default;

        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R> 
        __MATX_INLINE__ auto operator=(const R &rhs) { 
          return set(*this, rhs); 
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            return std::format("{}<{}>", get_jit_class_name(), op_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
            const auto [key, value] = get_jit_op_str();
            if (in.find(key) == in.end()) {
              in[key] = value;
            }
            detail::get_operator_capability<Cap>(op_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(op_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          } 
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return detail::get_operator_capability<Cap>(op_, in_copy);
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }
    };
  }

  /**
   * @brief Operator to logically reverse elements of an operator. Base case for variadic template.
   *
   * @tparam DIM Dimension to apply the reverse
   * @tparam Op Input operator/tensor type
   * @param t Input operator
   */
  template <int DIM, typename Op>
  auto __MATX_INLINE__ reverse(const Op &t)
  {
    return detail::ReverseOp<DIM, Op>(t);
  };

  /**
   * @brief Operator to logically reverse elements of an operator.
   *
   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam DIM Dimension to apply the reverse
   * @tparam DIMS... list of multiple dimensions to reverse along
   * @tparam Op Input operator/tensor type
   * @param t Input operator
   */
    template <int DIM1, int DIM2, int... DIMS, typename Op>
    auto __MATX_INLINE__ reverse(const Op &t)
  {
    // recursively call remap on remaining bits
    auto op = reverse<DIM2, DIMS...>(t);

    // construct remap op
    return detail::ReverseOp<DIM1, decltype(op)>(op);
  };

  /**
   * Flip the vertical axis of a tensor.
   */
  template <typename T1>
  auto __MATX_INLINE__ flipud(const T1 &t)
  {
    constexpr int dim = std::max(0, T1::Rank() - 2);
    return detail::ReverseOp<dim, T1>(t);
  };

  /**
   * Flip the horizontal axis of a tensor.
   */
  template <typename T1>
  auto __MATX_INLINE__ fliplr(const T1 &t)
  {
    if constexpr (T1::Rank() == 1)
    {
      return detail::ReverseOp<T1::Rank() - 1, T1>(t);
    }

    return detail::ReverseOp<T1::Rank() - 1, T1>(t);
  };
} // end namespace matx
