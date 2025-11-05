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
  namespace detail {
    template <typename T1>
      class FlattenOp : public BaseOp<FlattenOp<T1>>
    {
      private:
        mutable typename detail::base_type_t<T1> op1_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op1_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op1_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITFlatten_size{}", Size(0));
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = typename T::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static index_t size_ = {};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op1_;\n"
                "  template <typename CapType, typename Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is id0) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      const auto arr = detail::GetIdxFromAbs(op1_, id0);\n"
                "      return value_type(get_value<CapType>(op1_, arr));\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return 1; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const {{ return size_; }}\n"
                "}};\n",
                func_name, Size(0))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "flatten(" + op1_.str() + ")"; }
 
        __MATX_INLINE__ FlattenOp(const T1 &op1) : op1_(op1)
        {
          static_assert(T1::Rank() > 1, "flatten has no effect on tensors of rank 0 and 1");
          MATX_LOG_TRACE("{} constructor: input_rank={}, output_rank=1", str(), T1::Rank());
        }

        template <typename CapType, typename Op, typename Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_impl(Op&& op, Is id0) {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            const auto arr = detail::GetIdxFromAbs(op, id0);
            return value_type(get_value<CapType>(cuda::std::forward<Op>(op), arr));
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename CapType, typename Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is id0) const 
        {
          return get_impl<CapType>(cuda::std::as_const(op1_), id0);
        }

        template <typename Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is id0) const 
        {
          return this->operator()<DefaultCapabilities>(id0);
        }

        template <typename CapType, typename Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is id0) 
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op1_)>(op1_), id0);
        }

        template <typename Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is id0) 
        {
          return this->operator()<DefaultCapabilities>(id0);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return 1;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          index_t size = 1;
          if (dim == 0) {
            for (int r = 0; r < op1_.Rank(); r++) {
              size *= op1_.Size(r);
            }
          }

          return size;
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op1_, in);
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
            detail::get_operator_capability<Cap>(op1_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(op1_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op1_, in));
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op1_, in));
          }
        }
    };
  }

  /**
   * Flatten operator
   *
   * The flatten operator takes an operator of rank 2 or higher and flattens every dimension
   * into a single 1D tensor. 
   *
   * @tparam T1
   *   Operator type
   *
   * @returns
   *   Operator of flattened input
   */
  template <typename T1>
    auto __MATX_INLINE__ flatten(const T1 &a)
    {
      if constexpr (T1::Rank() <= 1) {
        return a;
      }
      else {
        return detail::FlattenOp<T1>(a);
      }
    };

} // end namespace matx
