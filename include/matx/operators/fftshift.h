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
      class FFTShift1DOp : public BaseOp<FFTShift1DOp<T1>>
    {
      private:
        typename detail::base_type_t<T1> op_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type; 

        __MATX_INLINE__ std::string str() const { return "fftshift(" + op_.str() + ")"; }

        __MATX_INLINE__ FFTShift1DOp(const T1 &op) : op_(op){
          MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
          static_assert(Rank() >= 1, "1D FFT shift must have a rank 1 operator or higher");
        };

        template <typename CapType, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
        { 
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            cuda::std::array idx{indices...};
            idx[Rank() - 1] = (idx[Rank() - 1] + (op.Size(Rank()-1) + 1) / 2) % op.Size(Rank()-1);
            return get_value<CapType>(cuda::std::forward<Op>(op), idx);
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

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          }
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(op_, in_copy));
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          return op_.Size(dim);
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
   * Perform an FFTShift operation on the last dimension of a tensor
   *
   * Shifts the new indexing of the tensor's last dimension to begin at
   * Size()/2. MatX FFTs leave the sample order starting with DC, positive
   * frequencies, then negative frequencies last. FFTShift gives a shifted
   * view of a signal where the new order is negative frequencies, DC, then
   * positive frequencies.
   *
   * @tparam T1
   *   Type of View/Op
   * @param t
   *   View/Op to shift
   *
   */
  template <typename T1>
    auto fftshift1D(const T1 &t) { return detail::FFTShift1DOp<T1>(t); }


  namespace detail {
    template <typename T1>
      class FFTShift2DOp : public BaseOp<FFTShift2DOp<T1>>
    {
      private:
        typename detail::base_type_t<T1> op_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ std::string str() const { return "fftshift(" + op_.str() + ")"; }

        __MATX_INLINE__ FFTShift2DOp(const T1 &op) : op_(op){
          MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
          static_assert(Rank() >= 2, "2D FFT shift must have a rank 2 operator or higher");
        };

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            auto tup = cuda::std::make_tuple(indices...);
            cuda::std::get<Rank()-2>(tup) = (cuda::std::get<Rank()-2>(tup) + (Size(Rank()-2) + 1) / 2) % Size(Rank()-2);
            cuda::std::get<Rank()-1>(tup) = (cuda::std::get<Rank()-1>(tup) + (Size(Rank()-1) + 1) / 2) % Size(Rank()-1);
            return cuda::std::apply([&](auto &&...call_args){ return op_.template operator()<CapType>(call_args...); }, tup);
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          }
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(op_, in_copy));
          }          
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          return op_.Size(dim);
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
   * Perform an IFFTShift operation on a 2D tensor swapping the first quadrant
   * with the third, and the second with the fourth.
   *
   * Shifts the new indexing of the tensor's last dimension to begin at
   * Size()/2. MatX FFTs leave the sample order starting with DC, positive
   * frequencies, then negative frequencies last. IFFTShift gives a shifted
   * view of a signal where the new order is negative frequencies, DC, then
   * positive frequencies.
   *
   * @tparam T1
   *   Type of View/Op
   * @param t
   *   View/Op to shift
   *
   */
  template <typename T1>
    auto fftshift2D(T1 t) { return detail::FFTShift2DOp<T1>(t); }

  namespace detail {
    template <typename T1>
      class IFFTShift1DOp : public BaseOp<IFFTShift1DOp<T1>>
    {
      private:
        typename detail::base_type_t<T1> op_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ IFFTShift1DOp(const T1 &op) : op_(op) {
          static_assert(Rank() >= 1, "1D IFFT shift must have a rank 1 operator or higher");
        };

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        { 
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            auto tup = cuda::std::make_tuple(indices...);
            cuda::std::get<Rank()-1>(tup) = (cuda::std::get<Rank()-1>(tup) + (Size(Rank()-1)) / 2) % Size(Rank()-1);
            return cuda::std::apply([&](auto &&...call_args){ return op_.template operator()<CapType>(call_args...); }, tup);
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        } 

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          }
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(op_, in_copy));
          }          
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          return op_.Size(dim);
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
   * Perform an IFFTShift operation on the last dimension of a tensor
   *
   * Shifts the new indexing of the tensor's last dimension to begin at
   * Size()/2. MatX FFTs leave the sample order starting with DC, positive
   * frequencies, then negative frequencies last. IFFTShift gives a shifted
   * view of a signal where the new order is negative frequencies, DC, then
   * positive frequencies. Note that ifftshift is the same as fftshift if the
   * length of the signal is even.
   *
   * @tparam T1
   *   Type of View/Op
   * @param t
   *   View/Op to shift
   *
   */
  template <typename T1>
    auto ifftshift1D(T1 t) { return detail::IFFTShift1DOp<T1>(t); }

  namespace detail {
    template <typename T1>
      class IFFTShift2DOp : public BaseOp<IFFTShift2DOp<T1>>
    {
      private:
        typename detail::base_type_t<T1> op_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ IFFTShift2DOp(const T1 &op) : op_(op) {
          static_assert(Rank() >= 2, "2D IFFT shift must have a rank 2 operator or higher");
        };

        template <typename CapType, typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            auto tup = cuda::std::make_tuple(indices...);
            cuda::std::get<Rank()-2>(tup) = (cuda::std::get<Rank()-2>(tup) + (Size(Rank()-2)) / 2) % Size(Rank()-2);
            cuda::std::get<Rank()-1>(tup) = (cuda::std::get<Rank()-1>(tup) + (Size(Rank()-1)) / 2) % Size(Rank()-1);
            return cuda::std::apply([&](auto &&...call_args){ return op_.template operator()<CapType>(call_args...); }, tup);
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }   

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          }
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(op_, in_copy));
          }          
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          return op_.Size(dim);
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
   * Perform an IFFTShift operation on a 2D tensor swapping the first quadrant
   * with the third, and the second with the fourth.
   *
   * Shifts the new indexing of the tensor's last dimension to begin at
   * Size()/2. MatX FFTs leave the sample order starting with DC, positive
   * frequencies, then negative frequencies last. IFFTShift gives a shifted
   * view of a signal where the new order is negative frequencies, DC, then
   * positive frequencies. Note that ifftshift is the same as fftshift if the
   * length of the signal is even.
   *
   * @tparam T1
   *   Type of View/Op
   * @param t
   *   View/Op to shift
   *
   */
  template <typename T1>
    auto ifftshift2D(const T1 &t) { return detail::IFFTShift2DOp<T1>(t); }
} // end namespace matx
