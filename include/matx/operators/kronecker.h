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
   * Kronecker tensor product
   *
   * Performs a Kronecker tensor product on two matrices. For input tensors A
   * (MxN) and B (PxQ), A is repeated and multiplied by each element in B to
   * create a new matrix of size M*P x N*Q.
   */
  namespace detail {
    template <typename T1, typename T2, int DIM>
      class KronOp : public BaseOp<KronOp<T1, T2, DIM>>
    {
      private:
        mutable typename detail::base_type_t<T1> op1_;
        mutable typename detail::base_type_t<T2> op2_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op1_;
          typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> op2_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op1_), detail::to_jit_storage(op2_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return "JITKron";
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < Rank(); ++i) {
            out_dims_[i] = Size(i);
          }
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T1, typename T2> struct {} {{\n"
                "  using value_type = typename T1::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op1_;\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T2>> op2_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      cuda::std::array idx1{{indices...}};\n"
                "      cuda::std::array idx2{{indices...}};\n"
                "      idx2[Rank_ - 2] = pp_get<Rank_ - 2>(indices...) % op2_.Size(Rank_ - 2);\n"
                "      idx2[Rank_ - 1] = pp_get<Rank_ - 1>(indices...) % op2_.Size(Rank_ - 1);\n"
                "      idx1[Rank_ - 2] = pp_get<Rank_ - 2>(indices...) / op2_.Size(Rank_ - 2);\n"
                "      idx1[Rank_ - 1] = pp_get<Rank_ - 1>(indices...) / op2_.Size(Rank_ - 1);\n"
                "      return get_value<CapType>(op2_, idx2) * get_value<CapType>(op1_, idx1);\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, Rank(), detail::array_to_string(out_dims_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "kron(" + op1_.str() + "," + op2_.str() + ")"; }

        __MATX_INLINE__ KronOp(const T1 &op1, const T2 &op2) : op1_(op1), op2_(op2)
        {
          MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
          static_assert(RankGTE(Rank(), 2), "Kronecker product must be used on tensors with rank 2 or higher");
        }        

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            cuda::std::array idx1{indices...};
            cuda::std::array idx2{indices...};

            idx2[Rank() - 2] = pp_get<Rank() - 2>(indices...) % op2_.Size(Rank() - 2);
            idx2[Rank() - 1] = pp_get<Rank() - 1>(indices...) % op2_.Size(Rank() - 1);

            idx1[Rank() - 2] = pp_get<Rank() - 2>(indices...) / op2_.Size(Rank() - 2);
            idx1[Rank() - 1] = pp_get<Rank() - 1>(indices...) / op2_.Size(Rank() - 1);

            return get_value<CapType>(op2_, idx2) * get_value<CapType>(op1_, idx1);
          }
          else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
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
        }          

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return op1_.Size(dim) * op2_.Size(dim);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op1_jit_name = detail::get_operator_capability<Cap>(op1_, in);
            const auto op2_jit_name = detail::get_operator_capability<Cap>(op2_, in);
            return std::format("{}<{},{}>", get_jit_class_name(), op1_jit_name, op2_jit_name);
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
            detail::get_operator_capability<Cap>(op2_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(op1_, in) +
                   detail::get_operator_capability<Cap>(op2_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(
              my_cap,
              detail::get_operator_capability<Cap>(op1_, in),
              detail::get_operator_capability<Cap>(op2_, in)
            );
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
              detail::get_operator_capability<Cap>(op1_, in),
              detail::get_operator_capability<Cap>(op2_, in)
            );
          }
        }
    };
  }

  /**
   * Kronecker tensor product
   *
   * The Kronecker tensor product is formed by the matrix b by ever element in the
   * matrix a. The resulting matrix has the number of rows and columns equal to
   * the product of the rows and columns of matrices a and b, respectively.
   *
   * @tparam T1
   *   Type of first input
   * @tparam T2
   *   Type of second input
   * @param a
   *   Operator or view for first input
   * @param b
   *   Operator or view for second input
   *
   * @returns
   *   New operator of the kronecker product
   */
  template <typename T1, typename T2>
    auto __MATX_INLINE__ kron(const T1 &a, const T2 &b)
    {
      return detail::KronOp<T1, T2, T1::Rank()>(a, b);
    };
} // end namespace matx
