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
        using ttype = std::conditional_t<is_complex_v<typename OpA::value_type>, 
                                      typename OpA::value_type, 
                                      typename scalar_to_complex<typename OpA::value_type>::ctype>;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr;
      public:
        using matxop = bool;
        using matx_transform_op = bool; 
        using value_type = typename OpA::value_type;
        using shape_type = index_t;
        using self_type = NormalizeOp<OpA, DIM>;

        __MATX_INLINE__ NormalizeOp(const OpA &op, const NORMALIZE_RANGE method): op_(op), normalize_method(method) {
          static_assert(DIM <= OpA::Rank(), "Normalize DIM must be less than the rank of operator");
          static_assert(DIM >= -1, "Normalize DIM must be non-negative or -1 for normalizing first non-singular dimension");
          for (int r = 0; r < OpA::Rank(); r++) {
            out_dims_[r] = op_.Size(r);
          }
        }

        __MATX_INLINE__ std::string str() const { return "normalize(" + op_.str() + ")"; }
        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return OpA::Rank(); }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          normalize_impl(cuda::std::get<0>(out), op_, normalize_method, ex);
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
          InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));

          detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

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
   * @param normalize_method Method of normalization to use: ZSCORE, NORM, SCALE, RANGE
   * @return normalized operator
   */
  template<int DIM=-1, typename OpA>
  __MATX_INLINE__ auto normalize(const OpA &op, const NORMALIZE_RANGE normalize_method) {
    MATX_NVTX_START("normalize(" + get_type_str(op) + ")", matx::MATX_NVTX_LOG_API)
    return detail::NormalizeOp<OpA, DIM>(op, normalize_method);
  }
}