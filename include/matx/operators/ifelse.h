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
#include "matx/core/tensor_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{
  namespace detail {
  /**
   * Conditionally execute an operator, otherwise execute a different operator
   *
   * Compares two operators or views and conditionally executes the second
   * statement if the first is true, otherwise executes the third statement.
   * Values from an operator are executed individually, and the only requirement
   * for the conditional is the comparison operator must be defined for the
   * particular type. For example, operator< on two integers is okay, but the same
   * operator on two complex numbers will give a compiler error.
   *
   */
  template <typename C1, typename T1, typename T2>
    class IFELSEOp : public BaseOp<IFELSEOp<C1, T1, T2>>
  {
    private:
      mutable typename detail::base_type_t<C1> cond_;
      mutable typename detail::base_type_t<T1> op1_;
      mutable typename detail::base_type_t<T2> op2_;
      cuda::std::array<index_t, detail::matx_max(detail::get_rank<C1>(), detail::get_rank<T1>(), detail::get_rank<T2>())> size_;

    public:
      using value_type = void; ///< Scalar type for type extraction

#ifdef MATX_EN_JIT
      struct JIT_Storage {
        typename detail::inner_storage_or_self_t<detail::base_type_t<C1>> cond_;
        typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op1_;
        typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> op2_;
      };

      JIT_Storage ToJITStorage() const {
        return JIT_Storage{detail::to_jit_storage(cond_), detail::to_jit_storage(op1_), detail::to_jit_storage(op2_)};
      }

      __MATX_INLINE__ std::string get_jit_class_name() const {
        return "JITIFELSE";
      }

      __MATX_INLINE__ auto get_jit_op_str() const {
        std::string func_name = get_jit_class_name();
        cuda::std::array<index_t, Rank()> out_dims_;
        for (int i = 0; i < Rank(); ++i) {
          out_dims_[i] = Size(i);
        }
        
        return cuda::std::make_tuple(
          func_name,
          std::format("template <typename C1, typename T1, typename T2> struct {} {{\n"
              "  using value_type = void;\n"
              "  constexpr static int Rank_ = {};\n"
              "  constexpr static cuda::std::array<index_t, Rank_> size_ = {{ {} }};\n"
              "  typename detail::inner_storage_or_self_t<detail::base_type_t<C1>> cond_;\n"
              "  typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op1_;\n"
              "  typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> op2_;\n"
              "  template <typename CapType, typename... Is>\n"
              "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const\n"
              "  {{\n"
              "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
              "      if (get_value<CapType>(cond_, indices...)) {{\n"
              "        return get_value<CapType>(op1_, indices...);\n"
              "      }} else {{\n"
              "        return get_value<CapType>(op2_, indices...);\n"
              "      }}\n"
              "    }} else {{\n"
              "      return Vector<int, static_cast<index_t>(CapType::ept)>{{}};\n"
              "    }}\n"
              "  }}\n"
              "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
              "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return size_[dim]; }}\n"
              "}};\n",
              func_name, Rank(), detail::array_to_string(out_dims_))
        );
      }
#endif

      __MATX_INLINE__ std::string str() const {
        return  "if(" + detail::get_type_str(cond_) + ") then {" +  detail::get_type_str(op1_) + "} else {" + detail::get_type_str(op2_) + "}";
      }

      /**
       * @brief Constructor for an IFELSEOp statement
       *
       * @param cond Condition to perform the IF/ELSE on
       * @param op1 Operator if conditional branch is true
       * @param op2 Operator if conditional branch is false
       */
      __MATX_INLINE__ IFELSEOp(const C1 &cond, const T1 &op1, const T2 &op2) :
                              cond_(cond), op1_(op1), op2_(op2)
    {
      MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
      static_assert((!is_tensor_view_v<T1> && !is_tensor_view_v<T2>),
          "Only operator emmitters are allowed in IFELSE. Tensor views "
          "are not allowed");
      constexpr int32_t rank0 = detail::get_rank<C1>();
      constexpr int32_t rank1 = detail::get_rank<T1>();
      constexpr int32_t rank2 = detail::get_rank<T2>();
      static_assert(rank0 == matxNoRank || rank0 == Rank());
      static_assert(rank1 == matxNoRank || rank1 == Rank());
      static_assert(rank2 == matxNoRank || rank2 == Rank());

      if constexpr (Rank() > 0)
      {
        for (int i = 0; i < Rank(); i++)
        {
          index_t size0 = detail::get_expanded_size<Rank()>(cond_, i);
          index_t size1 = detail::get_expanded_size<Rank()>(op1, i);
          index_t size2 = detail::get_expanded_size<Rank()>(op2, i);
          size_[i] = detail::matx_max(size0, size1, size2);
        }
      }

      MATX_ASSERT_COMPATIBLE_OP_SIZES(op1_);
      MATX_ASSERT_COMPATIBLE_OP_SIZES(op2_);
      MATX_ASSERT_COMPATIBLE_OP_SIZES(cond_);
    }

      /**
       * @brief Operator() for getting values of an if/else
       *
       * @tparam EPT ElementsPerThread
       * @tparam Is Index types
       * @param indices Index values
       */
      template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(Is... indices) const {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            if (get_value<DefaultCapabilities>(cond_, indices...)) {
              return get_value<DefaultCapabilities>(op1_, indices...);
            }
            else {
              return get_value<DefaultCapabilities>(op2_, indices...);
            }
          } else {
            return Vector<int, static_cast<index_t>(CapType::ept)>{};
          }
        }

      /**
       * @brief Operator() for getting values of an if/else
       *
       * @tparam Is Index types
       * @param indices Index values
       */
      template <typename... Is>
        __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(Is... indices) const {
          return this->operator()<DefaultCapabilities>(indices...);
        }

      /**
       * @brief Rank of IF/ELSE operator
       *
       * @return Rank
       */
      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return detail::matx_max(detail::get_rank<C1>(), detail::get_rank<T1>(), detail::get_rank<T2>());
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T1>()) {
          op1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<T2>()) {
          op2_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<C1>()) {
          cond_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
      {
        if constexpr (is_matx_op<T1>()) {
          op1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<T2>()) {
          op2_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }

        if constexpr (is_matx_op<C1>()) {
          cond_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
        }
      }

      /**
       * @brief Size of dimension of operator
       *
       * @param dim Dimension to get size of
       * @return Size of dimension
       */
      constexpr index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const
      {
        return size_[dim];
      }

      template <OperatorCapability Cap, typename InType>
      __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
        if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
          const auto cond_jit_name = detail::get_operator_capability<Cap>(cond_, in);
          const auto op1_jit_name = detail::get_operator_capability<Cap>(op1_, in);
          const auto op2_jit_name = detail::get_operator_capability<Cap>(op2_, in);
          return std::format("{}<{},{},{}>", get_jit_class_name(), cond_jit_name, op1_jit_name, op2_jit_name);
#else
          return "";
#endif
        }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true, 
              detail::get_operator_capability<Cap>(cond_, in),
              detail::get_operator_capability<Cap>(op1_, in),
              detail::get_operator_capability<Cap>(op2_, in));
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
          const auto [key, value] = get_jit_op_str();
          if (in.find(key) == in.end()) {
            in[key] = value;
          }
          detail::get_operator_capability<Cap>(cond_, in);
          detail::get_operator_capability<Cap>(op1_, in);
          detail::get_operator_capability<Cap>(op2_, in);
          return true;
#else
          return false;
#endif
        }
        else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
          return detail::get_operator_capability<Cap>(cond_, in) +
                 detail::get_operator_capability<Cap>(op1_, in) +
                 detail::get_operator_capability<Cap>(op2_, in);
        }
        else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
          const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
          return combine_capabilities<Cap>(
              my_cap,
            detail::get_operator_capability<Cap>(cond_, in),
            detail::get_operator_capability<Cap>(op1_, in),
            detail::get_operator_capability<Cap>(op2_, in)
          );
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(
              self_has_cap,
            detail::get_operator_capability<Cap>(cond_, in),
            detail::get_operator_capability<Cap>(op1_, in),
            detail::get_operator_capability<Cap>(op2_, in)
          );
        }
      }
  };

  } // end namespace detail

  /**
   *
   * @brief Compares two operators or views and conditionally executes the second
   * statement if the first is true. Values from an operator are executed
   * individually, and the only requirement for the conditional is the comparison
   * operator must be defined for the particular type. For example, operator< on
   * two integers is okay, but the same operator on two complex numbers will give
   * a compiler error.
   *
   * @param cond Condition to perform the IF/ELSE on
   * @param t1 op1
   *
   * @param t2 op2
   */
  template <typename C1, typename T1, typename T2>
    auto IFELSE(C1 cond, T1 t1, T2 t2) {
      return detail::IFELSEOp<C1,T1,T2>(cond,t1,t2);
    }  
} // end namespace matx
