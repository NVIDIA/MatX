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
   * permutes dimensions of a tensor/operator
   */
  namespace detail {
    template <typename T>
      class PermuteOp : public BaseOp<PermuteOp<T>>
    {
      public:
        using value_type = typename T::value_type;
        using self_type = PermuteOp<T>;

      private:
        mutable typename detail::base_type_t<T> op_;
        cuda::std::array<int32_t, T::Rank()> dims_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        // Propagate dynamic tensor marker through expression tree
        using dynamic_tensor_expr = cuda::std::bool_constant<
          is_dynamic_tensor_v<T> || is_dynamic_rank_op_v<T>>;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string dims_str;
          for (int i = 0; i < Rank(); i++) {
            dims_str += std::to_string(dims_[i]);
            if (i < Rank() - 1) dims_str += "_";
          }
          return "JITPermute_dims" + dims_str;
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          const int actual_rank = jit_rank();
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < actual_rank; ++i) {
            out_dims_[i] = Size(i);
          }

          std::string dims_array = "{ ";
          for (int i = 0; i < actual_rank; i++) {
            dims_array += std::to_string(dims_[i]);
            if (i < actual_rank - 1) dims_array += ", ";
          }
          dims_array += " }";

          return cuda::std::make_tuple(
            func_name,
            std::string("template <typename T> struct " + func_name + " {\n") +
                "  using value_type = typename T::value_type;\n" +
                "  using matxop = bool;\n" +
                "  constexpr static int Rank_ = " + std::to_string(actual_rank) + ";\n" +
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = { " +
                detail::array_to_string(out_dims_, actual_rank) + " };\n" +
                "  constexpr static cuda::std::array<int32_t, Rank_> dims_ = " + dims_array + ";\n" +
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n" +
                "  template <size_t K, typename Dims, typename Inds>\n" +
                "  static __MATX_INLINE__ __MATX_DEVICE__ index_t lookup_for_axis_(const Dims &dims, const Inds &inds) {\n" +
                "    index_t result = 0;\n" +
                "    for(int32_t j = 0; j < Rank_; j++) {\n" +
                "      if(dims[j] == static_cast<int32_t>(K)) result = inds[j];\n" +
                "    }\n" +
                "    return result;\n" +
                "  }\n" +
                "  template <typename CapType, typename Op, typename Dims, typename Inds, size_t... K>\n" +
                "  static __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) apply_permuted_(Op &&op, const Dims &dims, const Inds &inds, cuda::std::index_sequence<K...>) {\n" +
                "    return get_value<CapType>(cuda::std::forward<Op>(op), cuda::std::array<index_t, Rank_>{lookup_for_axis_<K>(dims, inds)...});\n" +
                "  }\n" +
                "  template <typename CapType, typename... Is>\n" +
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n" +
                "  {\n" +
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {\n" +
                "      const cuda::std::array<index_t, Rank_> inds{indices...};\n" +
                "      return apply_permuted_<CapType>(op_, dims_, inds, cuda::std::make_index_sequence<Rank_>{});\n" +
                "    } else {\n" +
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{};\n" +
                "    }\n" +
                "  }\n" +
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() { return Rank_; }\n" +
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int32_t dim) const { return out_dims_[dim]; }\n" +
                "};\n"
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "permute(" + op_.str() + ")"; }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T::Rank();
        }

        static_assert(Rank() > 0, "PermuteOp: Rank of operator must be greater than 0.");

        __MATX_INLINE__ PermuteOp(const T &op, const cuda::std::array<int32_t, T::Rank()> &dims) : op_(op) {

          cuda::std::array<bool, Rank()> seen{};
          for(int32_t i = 0; i < Rank(); i++) {
            const int32_t dim = dims[i];
            MATX_ASSERT_STR(dim < Rank() && dim >= 0, matxInvalidDim, "PermuteOp:  Invalid permute index.");
            MATX_ASSERT_STR(!seen[dim], matxInvalidDim, "PermuteOp:  Duplicate permute index.");
            seen[dim] = true;

            dims_[i] = dims[i];
          }
          MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
        }

        // For permuted-output axis K, find the input axis j with dims[j]==K
        // and return inds[j]. The accumulator form with a conditional store
        // (rather than an early return) is intentional: ptxas keeps `result`
        // in a register and the unrolled loop becomes a predicated chain. An
        // early-return inside the unrolled loop spills the inds array to
        // local memory because each iteration becomes a real exit edge that
        // inhibits the optimization. With the constructor's range +
        // uniqueness checks, exactly one j matches per K.
        template <size_t K, typename Dims, typename Inds>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ index_t
        lookup_for_axis_(const Dims &dims, const Inds &inds)
        {
          index_t result = 0;
          MATX_LOOP_UNROLL
          for (int32_t j = 0; j < Rank(); j++) {
            if (dims[j] == static_cast<int32_t>(K)) {
              result = inds[j];
            }
          }
          return result;
        }

        // Aggregate-initialize the permuted-index array via the parameter pack
        // expansion of lookup_for_axis_<K>(...) and forward it to op. Avoids a
        // mutable local array, which was the source of register spills (when
        // built via direct indexed store) and warnings (when conditionally
        // written in a double loop).
        template <typename CapType, typename Op, typename Dims, typename Inds, size_t... K>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto)
        apply_permuted_(Op &&op, const Dims &dims, const Inds &inds,
                        cuda::std::index_sequence<K...>)
        {
          return get_value<CapType>(cuda::std::forward<Op>(op),
                                    cuda::std::array<index_t, Rank()>{
                                        lookup_for_axis_<K>(dims, inds)...});
        }

        template <typename CapType, typename Op, typename Dims, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, const Dims &dims, Is... indices)
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            const cuda::std::array<index_t, Rank()> inds{indices...};
            return apply_permuted_<CapType>(cuda::std::forward<Op>(op),
                                            dims, inds,
                                            cuda::std::make_index_sequence<Rank()>{});
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<CapType>(cuda::std::as_const(op_), dims_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), dims_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          return op_.Size(dims_[dim]);
        }

        __MATX_INLINE__ __MATX_HOST__ int32_t DynRank() const {
          return detail::get_dyn_rank(op_);
        }

        __MATX_INLINE__ __MATX_HOST__ int32_t jit_rank() const {
          if constexpr (is_dynamic_rank_op_v<self_type>) return DynRank();
          else return Rank();
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            return get_jit_class_name() + "<" + op_jit_name + ">";
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

        ~PermuteOp() = default;
        PermuteOp(const PermuteOp &rhs) = default;

        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R>
        __MATX_INLINE__ auto operator=(const R &rhs) {
          return set(*this, rhs);
        }
    };
  }

  /**
   * @brief Operator to permute the dimensions of a tensor or operator.
   *
   * The each dimension must appear in the dims array once.
   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param dims the reordered dimensions of the operator.
   * @return permuted operator
   */
  template <typename T>
    __MATX_INLINE__ auto permute( const T &op,
        const cuda::std::array<int32_t, T::Rank()> &dims) {
      if constexpr (is_tensor_view_v<T>) {
        return op.Permute(dims);
      } else {
        return detail::PermuteOp<T>(op, dims);
      }
    }


  /**
   * @brief Operator to permute the dimensions of a tensor or operator.
   *
   * The each dimension must appear in the dims array once.

   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param dims the reordered dimensions of the operator.
   * @return permuted operator
   */
  template <typename T>
    __MATX_INLINE__ auto permute( const T &op,
        const int32_t (&dims)[T::Rank()]) {
      return permute(op, detail::to_array(dims));
    }


} // end namespace matx
