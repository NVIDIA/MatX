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
#include "matx/transforms/fft.h"

namespace matx
{
  namespace detail {
    template <typename OpA, typename PermDims, typename FFTType>
    class FFTOp : public BaseOp<FFTOp<OpA, PermDims, FFTType>>
    {
      private:
        OpA a_;
        uint64_t fft_size_;
        PermDims perm_;
        FFTType type_;
        std::array<index_t, OpA::Rank()> out_dims_;
        matx::tensor_t<std::conditional_t<is_complex_v<typename OpA::scalar_type>, 
                                          typename OpA::scalar_type, 
                                          typename scalar_to_complex<OpA>::ctype>, OpA::Rank()> tmp_out_;

      public:
        using matxop = bool;
        using scalar_type = typename OpA::scalar_type;
        using matx_transform_op = bool;
        using fft_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          if constexpr (std::is_same_v<FFTType, detail::fft_t>) {
            return "fft(" + get_type_str(a_) + ")";
          }
          else {
            return "ifft(" + get_type_str(a_) + ")";
          }
        }

        __MATX_INLINE__ FFTOp(OpA a, uint64_t size, PermDims perm, FFTType t) : a_(a), fft_size_(size),  perm_(perm), type_(t) {
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
          }

          if (fft_size_ != 0) {
            if constexpr (std::is_same_v<PermDims, no_permute_t>) {
              out_dims_[Rank() - 1] = fft_size_;
            }
            else {
              out_dims_[perm_[0]] = fft_size_;
            }
          }
          else {
            if constexpr (!is_complex_v<typename OpA::scalar_type>) { // C2C uses the same input/output size. R2C is N/2+1
              if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[perm_[0]] = out_dims_[perm_[0]] / 2 + 1;
              }
              else {
                out_dims_[Rank() - 1] = out_dims_[Rank() - 1] / 2 + 1;
              }
            }
          }     
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const
        {
          return tmp_out_(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) {
          static_assert(is_device_executor_v<Executor>, "fft()/ifft() only supports the CUDA executor currently");
          if constexpr (std::is_same_v<PermDims, no_permute_t>) {
            if constexpr (std::is_same_v<FFTType, fft_t>) { 
              fft_impl(std::get<0>(out), a_, fft_size_, ex.getStream());
            }
            else {
              ifft_impl(std::get<0>(out), a_, fft_size_, ex.getStream());
            }
          }
          else {
            if constexpr (std::is_same_v<FFTType, fft_t>) { 
              fft_impl(permute(std::get<0>(out), perm_), permute(a_, perm_), fft_size_, ex.getStream());
            }
            else {
              ifft_impl(permute(std::get<0>(out), perm_), permute(a_, perm_), fft_size_, ex.getStream());
            }
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_device_executor_v<Executor>) {
            make_tensor(tmp_out_, out_dims_, MATX_ASYNC_DEVICE_MEMORY, ex.getStream());
          }

          Exec(std::make_tuple(tmp_out_), std::forward<Executor>(ex));
        }
    };
  }


  /**
   * Run a 1D FFT with a cached plan
   *
   * Creates a new FFT plan in the cache if none exists, and uses that to execute
   * the 1D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
   * match
   * Note: fft_size must be unsigned so that the axis overload does not match both 
   * prototypes with index_t. 
   *
   * @tparam OpA
   *   Input tensor or operator type
   * @param a
   *   input tensor or operator
   * @param fft_size
   *   Size of FFT. Setting to 0 uses the output size to figure out the FFT size.
   */
  template<typename OpA>
  __MATX_INLINE__ auto fft(OpA &&a, uint64_t fft_size = 0) {
    return detail::FFTOp(a, fft_size, detail::no_permute_t{}, detail::fft_t{});
  }

  /**
   * Run a 1D FFT with a cached plan
   *
   * Creates a new FFT plan in the cache if none exists, and uses that to execute
   * the 1D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
   * match
   * Note: fft_size must be unsigned so that the axis overload does not match both 
   * prototypes with index_t. 
   *
   * @tparam OpA
   *   Input tensor or operator type
   * @param a
   *   input tensor or operator
   * @param axis
   *   axis to perform FFT along
   * @param fft_size
   *   Size of FFT. Setting to 0 uses the output size to figure out the FFT size.
   */
  template<typename OpA>
  __MATX_INLINE__ auto fft(OpA &&a, const int32_t (&axis)[1], uint64_t fft_size = 0) {

    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFTOp(a, fft_size, perm, detail::fft_t{});
  }

  /**
   * Run a 1D IFFT with a cached plan
   *
   * Creates a new FFT plan in the cache if none exists, and uses that to execute
   * the 1D IFFT. Note that FFTs and IFFTs share the same plans if all dimensions
   * match
   *
   * @tparam OpA
   *   Input tensor or operator type
   * 
   * Note: fft_size must be unsigned so that the axis overload does not match both 
   * prototypes with index_t. 
   * @param a
   *   input tensor or operator
   * @param fft_size
   *   Size of FFT. Setting to 0 uses the output size to figure out the FFT size.
   */
  template<typename OpA>
  __MATX_INLINE__ auto ifft(OpA &&a, uint64_t fft_size = 0) {
    return detail::FFTOp(a, fft_size, detail::no_permute_t{}, detail::ifft_t{});
  }

  /**
   * Run a 1D IFFT with a cached plan
   *
   * Creates a new FFT Iplan in the cache if none exists, and uses that to execute
   * the 1D IFFT. Note that FFTs and IFFTs share the same plans if all dimensions
   * match
   * Note: fft_size must be unsigned so that the axis overload does not match both 
   * prototypes with index_t. 
   *
   * @tparam OpA
   *   Input tensor or operator type
   * @param a
   *   input tensor or operator
   * @param axis
   *   axis to perform FFT along
   * @param fft_size
   *   Size of FFT. Setting to 0 uses the output size to figure out the FFT size.
   */
  template<typename OpA>
  __MATX_INLINE__ auto ifft(OpA &&a, const int32_t (&axis)[1], uint64_t fft_size = 0) {
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFTOp(a, fft_size, perm, detail::ifft_t{});
  }  


  namespace detail {
    template <typename OpA, typename PermDims, typename FFTType>
    class FFT2Op : public BaseOp<FFT2Op<OpA, PermDims, FFTType>>
    {
      private:
        OpA a_;
        PermDims perm_;
        FFTType type_;
        std::array<index_t, OpA::Rank()> out_dims_;
        matx::tensor_t<std::conditional_t<is_complex_v<typename OpA::scalar_type>, 
                                          typename OpA::scalar_type, 
                                          typename scalar_to_complex<OpA>::ctype>, OpA::Rank()> tmp_out_;

      public:
        using matxop = bool;
        using scalar_type = typename OpA::scalar_type;
        using matx_transform_op = bool;
        using fft2_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          if constexpr (std::is_same_v<FFTType, detail::fft_t>) {
            return "fft2(" + get_type_str(a_) + ")";
          }
          else {
            return "ifft2(" + get_type_str(a_) + ")";
          }
        }

        __MATX_INLINE__ FFT2Op(OpA a, PermDims perm, FFTType t) : a_(a),  perm_(perm), type_(t) {
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
          }

          if constexpr (!is_complex_v<typename OpA::scalar_type>) { // C2C uses the same input/output size. R2C is N/2+1
            if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
              out_dims_[perm_[0]] = out_dims_[perm_[0]] / 2 + 1;
              out_dims_[perm_[1]] = out_dims_[perm_[1]] / 2 + 1;
            }
            else {
              out_dims_[Rank() - 1] = out_dims_[Rank() - 1] / 2 + 1;
              out_dims_[Rank() - 2] = out_dims_[Rank() - 2] / 2 + 1;
            }
          }  
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
          return tmp_out_(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() {
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) {
          static_assert(is_device_executor_v<Executor>, "fft()/ifft() only supports the CUDA executor currently");
          if constexpr (std::is_same_v<PermDims, no_permute_t>) {
            if constexpr (std::is_same_v<FFTType, fft_t>) { 
              fft2_impl(std::get<0>(out), a_, ex.getStream());
            }
            else {
              ifft2_impl(std::get<0>(out), a_, ex.getStream());
            }
          }
          else {
            if constexpr (std::is_same_v<FFTType, fft_t>) { 
              fft2_impl(permute(std::get<0>(out), perm_), permute(a_, perm_), ex.getStream());
            }
            else {
              ifft2_impl(permute(std::get<0>(out), perm_), permute(a_, perm_), ex.getStream());
            }
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          if constexpr (is_device_executor_v<Executor>) {
            make_tensor(tmp_out_, out_dims_, MATX_ASYNC_DEVICE_MEMORY, ex.getStream());
          }

          Exec(std::make_tuple(tmp_out_), std::forward<Executor>(ex));
        }
    };    
  }


/**
 * Run a 2D FFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 2D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam OpA
 *   Input operator or tensor
 * @param a
 *   input operator or tensor
 */
  template<typename OpA>
  __MATX_INLINE__ auto fft2(OpA &&a) {
    return detail::FFT2Op(a, detail::no_permute_t{}, detail::fft_t{});
  }

/**
 * Run a 2D FFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 2D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam OpA
 *   Input operator or tensor type
 * @param a
 *   input operator or tensor
 * @param axis
 *   axes to perform fft on
 */
  template<typename OpA>
  __MATX_INLINE__ auto fft2(OpA &&a, const int32_t (&axis)[2]) {

    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFT2Op(a, perm, detail::fft_t{});
  }

/**
 * Run a 2D IFFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 2D IFFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam OpA
 *   Input operator or tensor type
 * @param a
 *   Input operator
 */
  template<typename OpA>
  __MATX_INLINE__ auto ifft2(OpA &&a) {
    return detail::FFT2Op(a, detail::no_permute_t{}, detail::ifft_t{});
  }

/**
 * Run a 2D IFFT with a cached plan
 *
 * Creates a new IFFT plan in the cache if none exists, and uses that to execute
 * the 2D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam OpA
 *   Input operator or tensor data type
 * @param a
 *   Input operator or tensor
 * @param axis
 *   axes to perform ifft on
 */
  template<typename OpA>
  __MATX_INLINE__ auto ifft2(OpA &&a, const int32_t (&axis)[2]) {
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFT2Op(a, perm, detail::ifft_t{});
  }  

}