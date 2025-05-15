////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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
#include "matx/transforms/normalize.h"

namespace matx 
{
  namespace detail
  {
    template <typename OpA, int DIM>
    class NormalizeOp: public BaseOp<NormalizeOp<OpA, DIM>>
    {
      private:
        typename detail::base_type_t<OpA> op_;
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        NORMALIZE_RANGE normalize_method;
        float p_ = -1.0f;
        float a_ = 0.0f;
        float b_ = 1.0f;
        using ttype = std::conditional_t<is_complex_v<typename OpA::value_type>, 
                                      typename OpA::value_type, 
                                      typename scalar_to_complex<typename OpA::value_type>::ctype>;
        mutable ::matx::detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;
        mutable bool prerun_done_ = false;

        __MATX_INLINE__ void InitNormalize() {
          static_assert(DIM <= OpA::Rank(), "Normalize DIM must be less than the rank of operator");
          static_assert(DIM >= -1, "Normalize DIM must be non-negative or -1 for normalizing first non-singular dimension");
          for (int r = 0; r < OpA::Rank(); r++) {
            out_dims_[r] = op_.Size(r);
          }
        }

      public:
        using matxop = bool;
        using matx_transform_op = bool; 
        using value_type = typename OpA::value_type;
        using self_type = NormalizeOp<OpA, DIM>;

        __MATX_INLINE__ NormalizeOp(const OpA &op, const NORMALIZE_RANGE method): op_(op), normalize_method(method) {
          InitNormalize();
        }

        __MATX_INLINE__ NormalizeOp(const OpA &op, const NORMALIZE_RANGE method, const float p): op_(op), normalize_method(method),  p_(p){
          MATX_ASSERT_STR(normalize_method == NORMALIZE_RANGE::NORM, matxInvalidParameter, "p value can be specified for only p-norm");
          InitNormalize();
        }

        __MATX_INLINE__ NormalizeOp(const OpA &op, const NORMALIZE_RANGE method, const float a, const float b): op_(op), normalize_method(method),  a_(a), b_(b){
          MATX_ASSERT_STR(normalize_method == NORMALIZE_RANGE::RANGE, matxInvalidParameter, "a and b values specify the range for range normalize method");
          InitNormalize();
        }

        __MATX_INLINE__ std::string str() const { return "normalize(" + op_.str() + ")"; }
        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif
          return tmp_out_.template operator()<CapType>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return OpA::Rank(); }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          normalize_impl<typename cuda::std::tuple_element<0, Out>::type, detail::base_type_t<OpA>, DIM, Executor>(
            cuda::std::get<0>(out), op_, normalize_method, p_, a_, b_, ex
          );
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if (prerun_done_) {
            return;
          }

          InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));

          detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

          prerun_done_ = true;
          Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          matxFree(ptr);
        }
    };
  } // end namespace detail

  /**
   * @brief Normalize operates along the first dimension of A whose size does not equal 1.
   *
   * For a matrix, it normalizes along the column by default
   *
   * @tparam OpA Type of input value to normalize
   * @param op Input value to evaluate
   * @param normalize_method Method of normalization to use: ZSCORE, NORM, SCALE, RANGE, CENTER
   * @return normalized operator
   */
  template<int DIM=-1, typename OpA>
  __MATX_INLINE__ auto normalize(const OpA &op, const NORMALIZE_RANGE normalize_method) {
    MATX_NVTX_START("normalize(" + get_type_str(op) + ")", matx::MATX_NVTX_LOG_API)
    return detail::NormalizeOp<OpA, DIM>(op, normalize_method);
  }

  /**
   * @brief Normalize operates along the first dimension of A whose size does not equal 1.
   *
   * For a matrix, it normalizes along the column by default
   *
   * @tparam OpA Type of input value to normalize
   * @param op Input value to evaluate
   * @param normalize_method Method of normalization to use: ZSCORE, NORM, SCALE, RANGE, CENTER
   * @param p for method="NORM" specify p for Lp-norm, if unspecified max norm (infinite norm) is applied
   * @return normalized operator
   */
  template<int DIM=-1, typename OpA>
  __MATX_INLINE__ auto normalize(const OpA &op, const NORMALIZE_RANGE normalize_method, const float p) {
    MATX_NVTX_START("normalize(" + get_type_str(op) + ")", matx::MATX_NVTX_LOG_API)
    return detail::NormalizeOp<OpA, DIM>(op, normalize_method, p);
  }

  /**
   * @brief Normalize operates along the first dimension of A whose size does not equal 1.
   *
   * For a matrix, it normalizes along the column by default
   *
   * @tparam OpA Type of input value to normalize
   * @param op Input value to evaluate
   * @param normalize_method Method of normalization to use: ZSCORE, NORM, SCALE, RANGE, CENTER
   * @param a start interval for range
   * @param b end interval for range
   * @return normalized operator
   */
  template<int DIM=-1, typename OpA>
  __MATX_INLINE__ auto normalize(const OpA &op, const NORMALIZE_RANGE normalize_method, const float a, const float b) {
    MATX_NVTX_START("normalize(" + get_type_str(op) + ")", matx::MATX_NVTX_LOG_API)
    return detail::NormalizeOp<OpA, DIM>(op, normalize_method, a, b);
  }
}