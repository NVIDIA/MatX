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
#include "matx/transforms/fft/fft_cuda.h"
#ifdef MATX_EN_CPU_FFT
  #include "matx/transforms/fft/fft_fftw.h"
#endif  

namespace matx
{
  namespace detail {
    template <typename OpA, typename PermDims, typename FFTType>
    class FFTOp : public BaseOp<FFTOp<OpA, PermDims, FFTType>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        uint64_t fft_size_;
        PermDims perm_;
        FFTType type_;
        FFTNorm norm_;
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        using ttype = std::conditional_t<is_complex_v<typename OpA::value_type>, 
                                          typename OpA::value_type, 
                                          typename scalar_to_complex<typename OpA::value_type>::ctype>;
        // This should be tensor_impl_t, but need to work around issues with temp types returned in fft
        mutable detail::tensor_impl_t<ttype, OpA::Rank()> tmp_out_;
        mutable ttype *ptr = nullptr;                                           

      public:
        using matxop = bool;
        using value_type = std::conditional_t<is_complex_v<typename OpA::value_type>,
          typename OpA::value_type,
          typename scalar_to_complex<typename OpA::value_type>::ctype>;
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

        __MATX_INLINE__ FFTOp(const OpA &a, uint64_t size, PermDims perm, FFTType t, FFTNorm norm) : 
            a_(a), fft_size_(size),  perm_(perm), type_(t), norm_(norm) {
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
          }

          if (fft_size_ != 0) {
            if constexpr (is_complex_v<typename OpA::value_type>) {
              if constexpr (std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[Rank() - 1] = fft_size_;
              }
              else {
                out_dims_[perm_[Rank()-1]] = fft_size_;
              }
            } else {
              // R2C transforms pack the results in fft_size_/2 + 1 complex elements
              if constexpr (std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[Rank() - 1] = fft_size_ / 2 + 1;
              }
              else {
                out_dims_[perm_[Rank()-1]] = fft_size_ / 2 + 1;
              }
            }
          }
          else { 
            if constexpr (!is_complex_v<typename OpA::value_type>) { // C2C uses the same input/output size. R2C is N/2+1
              if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[perm_[Rank()-1]] = out_dims_[perm_[Rank()-1]] / 2 + 1;
              }
              else {
                out_dims_[Rank() - 1] = out_dims_[Rank() - 1] / 2 + 1;
              }
            }
            // For R2C transforms, the output length could correspond to an input length of
            // either (out_dim-1)*2 or (out_dim-1)*2+1. The FFT transform will be unable to
            // deduce which is correct, so explicitly set the transform size here. For C2C
            // transforms, we do not want to implicitly zero-pad. Users can opt-in to
            // zero-padding by providing the desired padded fft_size_ as a parameter.
            // For C2R transforms, we do not know the fft size -- it could be either
            // (N-1)*2 or (N-1)*2+1. If the operator is used such that the output is a
            // real tensor, then the size will be set to the output tensor length. If we
            // create a temporary, then this will be a C2C transform and the output length
            // will match the input length.
            if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
              fft_size_ = a.Size(perm_[Rank()-1]);
            } else {
              fft_size_ = a.Size(a.Rank()-1);
            }
          }
        }

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }
                  
        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<InWidth, OutWidth>(indices...);
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
        void Exec(Out &&out, Executor &&ex) const {
          if constexpr (std::is_same_v<PermDims, no_permute_t>) {
            if constexpr (std::is_same_v<FFTType, fft_t>) {
              fft_impl(cuda::std::get<0>(out), a_, fft_size_, norm_, ex);
            }
            else {
              ifft_impl(cuda::std::get<0>(out), a_, fft_size_, norm_, ex);
            }
          }
          else {
            if constexpr (std::is_same_v<FFTType, fft_t>) { 
              fft_impl(permute(cuda::std::get<0>(out), perm_), permute(a_, perm_), fft_size_, norm_, ex);
            }
            else {
              ifft_impl(permute(cuda::std::get<0>(out), perm_), permute(a_, perm_), fft_size_, norm_, ex);
            }
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          matxFree(ptr); 
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
   * @param norm
   *   Normalization to apply to FFT
   */
  template<typename OpA>
  __MATX_INLINE__ auto fft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
    return detail::FFTOp(a, fft_size, detail::no_permute_t{}, detail::fft_t{}, norm);
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
   * @param norm
   *   Normalization to apply to FFT
   */
  template<typename OpA>
  __MATX_INLINE__ auto fft(const OpA &a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {

    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFTOp(a, fft_size, perm, detail::fft_t{}, norm);
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
   * @param norm
   *   Normalization to apply to IFFT
   */
  template<typename OpA>
  __MATX_INLINE__ auto ifft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
    return detail::FFTOp(a, fft_size, detail::no_permute_t{}, detail::ifft_t{}, norm);
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
   * @param norm
   *   Normalization to apply to IFFT
   */
  template<typename OpA>
  __MATX_INLINE__ auto ifft(const OpA &a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFTOp(a, fft_size, perm, detail::ifft_t{}, norm);
  }  


  namespace detail {
    template <typename OpA, typename PermDims, typename FFTType>
    class FFT2Op : public BaseOp<FFT2Op<OpA, PermDims, FFTType>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        PermDims perm_;
        FFTType type_;
        FFTNorm norm_;
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        using ttype = std::conditional_t<is_complex_v<typename OpA::value_type>, 
                                          typename OpA::value_type, 
                                          typename scalar_to_complex<typename OpA::value_type>::ctype>;
        // This should be tensor_impl_t, but need to work around issues with temp types returned in fft
        mutable detail::tensor_impl_t<ttype, OpA::Rank()> tmp_out_; 
        mutable ttype *ptr = nullptr;                                                

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
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

        __MATX_INLINE__ FFT2Op(const OpA &a, PermDims perm, FFTType t, FFTNorm norm) : a_(a),  perm_(perm), type_(t), norm_(norm) {
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
          }

          if constexpr (!is_complex_v<typename OpA::value_type>) { // C2C uses the same input/output size. R2C is N/2+1
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

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<InWidth, OutWidth>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() {
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          if constexpr (std::is_same_v<PermDims, no_permute_t>) {
            if constexpr (std::is_same_v<FFTType, fft_t>) { 
              fft2_impl(cuda::std::get<0>(out), a_, norm_, ex);
            }
            else {
              ifft2_impl(cuda::std::get<0>(out), a_, norm_, ex);
            }
          }
          else {
            if constexpr (std::is_same_v<FFTType, fft_t>) { 
              fft2_impl(permute(cuda::std::get<0>(out), perm_), permute(a_, perm_), norm_, ex);
            }
            else {
              ifft2_impl(permute(cuda::std::get<0>(out), perm_), permute(a_, perm_), norm_, ex);
            }
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
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
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          matxFree(ptr);           
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
 *   Input operator or tensor
 * @param norm
 *   Normalization to apply to FFT
 */
  template<typename OpA>
  __MATX_INLINE__ auto fft2(const OpA &a, FFTNorm norm = FFTNorm::BACKWARD) {
    return detail::FFT2Op(a, detail::no_permute_t{}, detail::fft_t{}, norm);
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
 * @param norm
 *   Normalization to apply to FFT
 */
  template<typename OpA>
  __MATX_INLINE__ auto fft2(const OpA &a, const int32_t (&axis)[2], FFTNorm norm = FFTNorm::BACKWARD) {

    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFT2Op(a, perm, detail::fft_t{}, norm);
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
 * @param norm
 *   Normalization to apply to IFFT
 */
  template<typename OpA>
  __MATX_INLINE__ auto ifft2(const OpA &a, FFTNorm norm = FFTNorm::BACKWARD) {
    return detail::FFT2Op(a, detail::no_permute_t{}, detail::ifft_t{}, norm);
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
 * @param norm
 *   Normalization to apply to IFFT
 */
  template<typename OpA>
  __MATX_INLINE__ auto ifft2(const OpA &a, const int32_t (&axis)[2], FFTNorm norm = FFTNorm::BACKWARD) {
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFT2Op(a, perm, detail::ifft_t{}, norm);
  }  

}