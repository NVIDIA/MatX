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
   * Returns the polynomial evaluated at each point
   */
  namespace detail {
    template <typename Op, typename Coeffs>
    class PolyvalOp : public BaseOp<PolyvalOp<Op, Coeffs>>
    {
      private:
        mutable typename detail::base_type_t<Op> op_;
        mutable Coeffs coeffs_;

      public:
        using matxop = bool;
        using value_type = typename Op::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<Op>> op_;
          typename detail::inner_storage_or_self_t<Coeffs> coeffs_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_), detail::to_jit_storage(coeffs_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITPolyval_ncoeffs{}", coeffs_.Size(0));
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename Op, typename Coeffs> struct {} {{\n"
                "  using value_type = typename Op::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static index_t ncoeffs_ = {};\n"
                "  constexpr static index_t size_ = {};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<Op>> op_;\n"
                "  typename detail::inner_storage_or_self_t<Coeffs> coeffs_;\n"
                "  template <typename CapType>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(index_t idx) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      value_type ttl{{get_value<CapType>(coeffs_, 0)}};\n"
                "      for(int i = 1; i < ncoeffs_; i++) {{\n"
                "        ttl = ttl * get_value<CapType>(op_, idx) + get_value<CapType>(coeffs_, i);\n"
                "      }}\n"
                "      return ttl;\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return 1; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const {{ return size_; }}\n"
                "}};\n",
                func_name, coeffs_.Size(0), op_.Size(0))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "polyval()"; }
        __MATX_INLINE__ PolyvalOp(const Op &op, const Coeffs &coeffs) : op_(op), coeffs_(coeffs) {
          MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
          MATX_STATIC_ASSERT_STR(Coeffs::Rank() == 1, matxInvalidDim, "Coefficient must be rank 1");
          MATX_STATIC_ASSERT_STR(Op::Rank() == 1, matxInvalidDim, "Input operator must be rank 1");
        };

        template <typename CapType>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(index_t idx) const
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            // Horner's method for computing polynomial
            value_type ttl{get_value<CapType>(coeffs_, 0)};
            for(int i = 1; i < coeffs_.Size(0); i++) {
                ttl = ttl * get_value<CapType>(op_, idx) + get_value<CapType>(coeffs_, i);
            }

            return ttl;
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(index_t idx) const
        {
          return this->operator()<DefaultCapabilities>(idx);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return 1;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          return op_.Size(0);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<Op>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            const auto coeffs_jit_name = detail::get_operator_capability<Cap>(coeffs_, in);
            return std::format("{}<{},{}>", get_jit_class_name(), op_jit_name, coeffs_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
            return combine_capabilities<Cap>(true, detail::get_operator_capability<Cap>(op_, in));
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
            detail::get_operator_capability<Cap>(op_, in);
            detail::get_operator_capability<Cap>(coeffs_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(op_, in) +
                   detail::get_operator_capability<Cap>(coeffs_, in);
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(
              my_cap,
              detail::get_operator_capability<Cap>(op_, in),
              detail::get_operator_capability<Cap>(coeffs_, in)
            );
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
              detail::get_operator_capability<Cap>(op_, in),
              detail::get_operator_capability<Cap>(coeffs_, in)
            );
          }
        }
    };
  }


  /**
   * @brief Evaluate a polynomial
   * 
   * Currently only allows 1D input and coefficients
   * 
   * @tparam Op Type of input values to evaluate
   * @tparam Coeffs Type of coefficients
   * @param op Input values to evaluate
   * @param coeffs Coefficient values
   * @return polyval operator 
   */
  template <typename Op, typename Coeffs>
  __MATX_INLINE__ auto polyval(const Op &op, const Coeffs &coeffs) {
    return detail::PolyvalOp(op, coeffs);
  }
} // end namespace matx
