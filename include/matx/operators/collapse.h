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
    template <int DIM, typename T1>
      class LCollapseOp : public BaseOp<LCollapseOp<DIM, T1>>
    {
      private:
        mutable typename detail::base_type_t<T1> op_;
        index_t size_;  // size of collapsed dim

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;
        using matxoplvalue = bool;
        using self_type = LCollapseOp<DIM, T1>;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITLCollapse_dim{}_size{}", DIM, size_);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < Rank(); ++i) {
            out_dims_[i] = Size(i);
          }

          cuda::std::array<index_t, T1::Rank()> op_sizes_;
          for (int i = 0; i < T1::Rank(); ++i) {
            op_sizes_[i] = op_.Size(i);
          }
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = typename T::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int DIM_ = {};\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static int OpRank_ = {};\n"
                "  constexpr static index_t size_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
                "  constexpr static cuda::std::array<index_t, OpRank_> op_sizes_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      cuda::std::array<index_t, Rank_> in{{indices...}};\n"
                "      cuda::std::array<index_t, OpRank_> out;\n"
                "      for(int i = 1; i < Rank_; i++) out[DIM_ + i - 1] = in[i];\n"
                "      auto ind = in[0];\n"
                "      for(int i = 0; i < DIM_; i++) {{\n"
                "        int d = DIM_ - i - 1;\n"
                "        out[d] = ind % op_sizes_[d];\n"
                "        ind /= op_sizes_[d];\n"
                "      }}\n"
                "      return get_value<CapType>(op_, out);\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, DIM, Rank(), T1::Rank(), size_, detail::array_to_string(out_dims_), detail::array_to_string(op_sizes_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "lcollapse<" + std::to_string(DIM) + ">(" + op_.str() + ")"; }
        __MATX_INLINE__ LCollapseOp(const T1 &op) : op_(op)
        {
          static_assert(DIM <= T1::Rank(),  "Collapse DIM must be less than or equal to Rank() of operator");
          static_assert(DIM > 1, "Must collapse multiple dims");
          MATX_LOG_TRACE("{} constructor: input_rank={}, output_rank={}", str(), T1::Rank(), T1::Rank() - DIM + 1);
          static_assert(T1::Rank() >= 2, "Collapse must be called on operators with rank >= 2");

          // compute size of collapsed dimension
          size_ = 1;

          // Collapse left-most dims
  MATX_LOOP_UNROLL
          for(int i = 0 ; i < DIM; i++) {
            size_ *= op_.Size(i);
          }
        }

        template <typename CapType, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            // indices coming in
            cuda::std::array<index_t, Rank()> in{indices...};  // index coming in
            cuda::std::array<index_t, T1::Rank()> out;         // index going out

  MATX_LOOP_UNROLL
            for(int i = 1; i < Rank(); i++) {
              // copy all but first input index into out array
              out[DIM + i - 1] = in[i];
            }

            // expand first input index into DIM indices
            auto ind = in[0];
  MATX_LOOP_UNROLL
            for(int i = 0; i < DIM; i++) {
              int d = DIM - i - 1;
              out[d] = ind % op.Size(d);
              ind /= op.Size(d);
            }

            return get_value<CapType>(cuda::std::forward<Op>(op), out);
          }
          else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
        {
          return get_impl<DefaultCapabilities>(cuda::std::forward<Op>(op), indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return get_impl<CapType>(cuda::std::as_const(op_), indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<DefaultCapabilities>(cuda::std::as_const(op_), indices...);
        }    

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<DefaultCapabilities>(cuda::std::forward<decltype(op_)>(op_), indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T1::Rank() - DIM + 1;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          if(dim == 0)  // if asking for the first dim, return collapsed size
            return size_;
          else // otherwise return the un-collapsed size from operator
            return op_.Size(DIM + dim - 1);
        }

        ~LCollapseOp() = default;
        LCollapseOp(const LCollapseOp &rhs) = default;

        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R>
        __MATX_INLINE__ auto operator=(const R &rhs) {
          return set(*this, rhs);
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
    };
  }
  /**
   * lcollapse operator
   *
   * The lcollapse operator takes a tensor and collapses the left most dimensions into a single dimension.
   *
   * @tparam DIM
   *   The number of input dimensions to collapse
   * @tparam T1
   *   Operator type
   *
   * @param a
   *   The operator being collapsed
   *
   * @returns
   *   Operator with collapsed input
   */
  template <int DIM, typename T1>
    auto __MATX_INLINE__ lcollapse(const T1 &a)
    {
      if constexpr (DIM <= 1) {
        return a;
      }
      else {
        return detail::LCollapseOp<DIM, T1>(a);
      }
    }

  namespace detail {
    template <int DIM, typename T1>
      class RCollapseOp : public BaseOp<RCollapseOp<DIM, T1>>
    {
      private:
        mutable typename detail::base_type_t<T1> op_;
        index_t size_;  // size of collapsed dim

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;
        using matxoplvalue = bool;
        using self_type = RCollapseOp<DIM, T1>;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T1>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITRCollapse_dim{}_size{}", DIM, size_);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < Rank(); ++i) {
            out_dims_[i] = Size(i);
          }

          cuda::std::array<index_t, T1::Rank()> op_sizes_;
          for (int i = 0; i < T1::Rank(); ++i) {
            op_sizes_[i] = op_.Size(i);
          }
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = typename T::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int DIM_ = {};\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static int OpRank_ = {};\n"
                "  constexpr static index_t size_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
                "  constexpr static cuda::std::array<index_t, OpRank_> op_sizes_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      cuda::std::array<index_t, Rank_> in{{indices...}};\n"
                "      cuda::std::array<index_t, OpRank_> out;\n"
                "      for(int i = 0 ; i < Rank_ - 1; i++) out[i] = in[i];\n"
                "      auto ind = in[Rank_ - 1];\n"
                "      for(int i = 0; i < DIM_; i++) {{\n"
                "        int d = OpRank_ - 1 - i;\n"
                "        out[d] = ind % op_sizes_[d];\n"
                "        ind /= op_sizes_[d];\n"
                "      }}\n"
                "      return get_value<CapType>(op_, out);\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, DIM, Rank(), T1::Rank(), size_, detail::array_to_string(out_dims_), detail::array_to_string(op_sizes_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "rcollapse<" + std::to_string(DIM) + ">(" + op_.str() + ")"; }

        __MATX_INLINE__ RCollapseOp(const T1 &op) : op_(op)
        {
          static_assert(DIM <= T1::Rank(),  "Collapse DIM must be less than or equal to Rank() of operator");
          static_assert(DIM > 1, "Collapse DIM must have be greater than 1");
          static_assert(T1::Rank() >= 2, "Collapse must be called on operators with rank >= 2");

          // comptue size of collapsed dimension
          size_ = 1;

          // Collapse right-most dims
  MATX_LOOP_UNROLL
          for(int i = 0 ; i < DIM; i++) {
            size_ *= op_.Size(T1::Rank() - 1 - i);
          }
        }

        template <typename CapType, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
        {      
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            // indices coming in
            cuda::std::array<index_t, Rank()> in{indices...};  // index coming in
            cuda::std::array<index_t, T1::Rank()> out;         // index going out

MATX_LOOP_UNROLL
            for(int i = 0 ; i < Rank() - 1; i++) {
              // copy all but last index into out array
              out[i] = in[i];
            }

            // expand last index into DIM indices
            auto ind = in[Rank() - 1];
MATX_LOOP_UNROLL
            for(int i = 0; i < DIM; i++) {
              int d = T1::Rank() - 1 - i;
              out[d] = ind % op.Size(d);
              ind /= op.Size(d);
            }

            return get_value<CapType>(cuda::std::forward<Op>(op), out);
          }
          else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
        {
          return get_impl<DefaultCapabilities>(cuda::std::forward<Op>(op), indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return get_impl<CapType>(cuda::std::as_const(op_), indices...);
        }    

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), indices...);
        } 

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<DefaultCapabilities>(cuda::std::as_const(op_), indices...);
        }    

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<DefaultCapabilities>(cuda::std::forward<decltype(op_)>(op_), indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T1::Rank() - DIM + 1;
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          if(dim == Rank()-1)  // if asking for the last dim, return collapsed size
            return size_;
          else // otherwise return the un-collapsed size from operator
            return op_.Size(dim);
        }

        ~RCollapseOp() = default;
        RCollapseOp(const RCollapseOp &rhs) = default;

        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R>
        __MATX_INLINE__ auto operator=(const R &rhs) {
          return set(*this, rhs);
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
    };
  }
  /**
   * rcollapse operator
   *
   * The rcollapse operator takes a tensor and collapses the right most dimensions into a single dimension.
   *
   * @tparam DIM
   *   The number of input dimensions to collapse
   * @tparam T1
   *   Operator type
   *
   * @param a
   *   The parameter being collapsed
   *
   * @returns
   *   Operator with collapsed input
   */
  template <int DIM, typename T1>
    auto __MATX_INLINE__ rcollapse(const T1 &a)
    {
      if constexpr (DIM <= 1) {
        return a;
      }
      else {
        return detail::RCollapseOp<DIM, T1>(a);
      }
    }
} // end namespace matx
