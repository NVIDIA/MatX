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
#include <cuda/std/__algorithm/min.h>

namespace matx
{
  /**
   * Returns elements on the diagonal
   *
   * Returns elements on the diagonal of a 2D tensor. Any dimensions above 2 will
   * be considered batch dimension and the size of those match the size of the
   * input operator. The last dimension is always sized to be the minimum of the
   * last two dimension of the input operator
   */
  namespace detail {
    template <typename T1, int RANK>
      class DiagOp : public BaseOp<DiagOp<T1, RANK>>
    {
      private:
        mutable typename detail::base_type_t<T1> op_;
        index_t k_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITDiag_k{}_rank{}", k_, RANK);
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
                "  constexpr static int RANK_ = {};\n"
                "  constexpr static int OutRank_ = {};\n"
                "  constexpr static index_t k_ = {};\n"
                "  constexpr static cuda::std::array<index_t, OutRank_> out_dims_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      if constexpr (RANK_ == 1) {{\n"
                "        cuda::std::array<index_t, 2> idx{{indices...}};\n"
                "        if (idx[0] == idx[1]) {{\n"
                "          return get_value<CapType>(op_, cuda::std::array<index_t, 1>{{idx[0]}});\n"
                "        }}\n"
                "        else {{\n"
                "          return static_cast<value_type>(0);\n"
                "        }}\n"
                "      }}\n"
                "      else {{\n"
                "        cuda::std::array<index_t, sizeof...(Is)> idx{{indices...}};\n"
                "        cuda::std::array<index_t, RANK_> tmp;\n"
                "        for (int i = 0; i < RANK_ - 2; i++) {{\n"
                "          tmp[i] = idx[i];\n"
                "        }}\n"
                "        if (k_ < 0) {{\n"
                "          tmp[RANK_ - 1] = idx[RANK_ - 2];\n"
                "          tmp[RANK_ - 2] = idx[RANK_ - 2] - k_;\n"
                "        }}\n"
                "        else {{\n"
                "          tmp[RANK_ - 2] = idx[RANK_ - 2];\n"
                "          tmp[RANK_ - 1] = idx[RANK_ - 2] + k_;\n"
                "        }}\n"
                "        return get_value<CapType>(op_, tmp);\n"
                "      }}\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return OutRank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, RANK, Rank(), k_, detail::array_to_string(out_dims_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "diag(" + op_.str() + ")"; }

        __MATX_INLINE__ DiagOp(const T1 &op, index_t k) : op_(op), k_(k) {
          MATX_LOG_TRACE("{} constructor: k={}", str(), k);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            static_assert(RANK != 0, "Cannot make get diagonals from 0D tensor");
            using tt = typename cuda::std::common_type_t<Is...>;

            if constexpr (RANK == 1) {
              static_assert(sizeof...(Is) == 2, "Indexing of diag() on a 1D input must be 2 indices");
              if (((pp_get<0>(indices...) == indices) && ...)) {
                return get_value<CapType>(op_, pp_get<0>(indices...));
              }
              else {
                return (value_type)(0);
              }
            }
            else {
              static_assert(sizeof...(Is) == RANK - 1, "Diagonal operator must have one fewer op() index than rank of operator");

              // Offset either the rows or columns by k_, depending on if it's negative
              if (k_ < 0) {
                cuda::std::array<tt, sizeof...(Is) + 1> tmp{indices...};
                tmp[RANK - 1] = pp_get<RANK-2>(indices...);
                //cuda::std::get<RANK - 1>(tup) = pp_get<RANK-2>(indices...) ;
  MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
                tmp[RANK - 2] -= k_;
                //cuda::std::get<RANK - 2>(tup) = cuda::std::get<RANK - 2>(tup) - k_;
  MATX_IGNORE_WARNING_POP_GCC
                return get_value<CapType>(op_, tmp);
              }
              else {
                cuda::std::array<tt, sizeof...(Is) + 1> tmp{indices...};
                //auto tup = cuda::std::make_tuple(indices..., static_cast<tt>(0));
  MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
                tmp[RANK - 1] = pp_get<RANK-2>(indices...) + k_;
                //cuda::std::get<RANK - 1>(tup) = pp_get<RANK-2>(indices...) + k_;
  MATX_IGNORE_WARNING_POP_GCC
                return get_value<CapType>(op_, tmp);
              }
            }
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

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            return std::format("{}<{}>", get_jit_class_name(), op_jit_name);
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
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          if constexpr (RANK == 1) {
            return 2;
          }
          else {
            return RANK - 1;
          }
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          if constexpr (RANK == 1) {
            return op_.Size(0);
          }
          else {
            if (dim < RANK - 2) {
              return op_.Size(dim);
            }
            else {
              if (k_ == 0) {
                return cuda::std::min(op_.Size(RANK - 1), op_.Size(RANK-2));
              }
              else {
                // If k is off the main diagonal we need to adjust the sizes
                if (k_ > 0) {
                  return cuda::std::min(op_.Size(RANK - 1), op_.Size(RANK-2) - k_);
                }
                else {
                  return cuda::std::min(op_.Size(RANK - 1) + k_, op_.Size(RANK-2));
                }
              }
            }
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }

  /**
   * Get the elements on the diagonal (2D inputs and above), or generate a diagonal matrix (1D input)
   *
   * @param t
   *   Input operator
   * @param k
   *   Diagonal to pull (0 is the main diagonal). Only used for 2D tensors and above
   */
#ifdef DOXYGEN_ONLY
  auto __MATX_INLINE__ diag(const T1 &t, index_t k = 0) {
#else
  template <typename T1>
    requires is_matx_op_c<T1>
    auto __MATX_INLINE__ diag(T1 t, index_t k = 0) {
#endif
      MATX_ASSERT_STR(T1::Rank() != 1 || k == 0, matxInvalidParameter,
          "k parameter in diag() can only be used for 2D tensors and above");
      return detail::DiagOp<T1, T1::Rank()>(t, k);
    }
} // end namespace matx
