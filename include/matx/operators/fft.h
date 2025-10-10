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
    template <typename OpA, typename PermDims, FFTDirection Direction, FFTType Type>
    class FFTOp : public BaseOp<FFTOp<OpA, PermDims, Direction, Type>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        index_t fft_size_;
        PermDims perm_;
        FFTNorm norm_;
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        using ttype = std::conditional_t<is_complex_v<typename OpA::value_type>, 
                                          typename OpA::value_type, 
                                          typename scalar_to_complex<typename OpA::value_type>::ctype>;
        // This should be tensor_impl_t, but need to work around issues with temp types returned in fft
        mutable detail::tensor_impl_t<ttype, OpA::Rank()> tmp_out_;
        mutable ttype *ptr = nullptr;
        mutable bool prerun_done_ = false;                                       

      public:
        using matxop = bool;
        using value_type = std::conditional_t<is_complex_v<typename OpA::value_type>,
          typename OpA::value_type,
          typename scalar_to_complex<typename OpA::value_type>::ctype>;
        using matx_transform_op = bool;
        using fft_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          if constexpr (Direction == detail::FFTDirection::FORWARD) {
            return "fft(" + get_type_str(a_) + ")";
          }
          else {
            return "ifft(" + get_type_str(a_) + ")";
          }
        }

        __MATX_INLINE__ FFTOp(const OpA &a, index_t size, PermDims perm, FFTNorm norm) : 
            a_(a), fft_size_(size),  perm_(perm), norm_(norm) {
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
          }

          if (fft_size_ != 0) {
            if constexpr (Type == detail::FFTType::C2C) {
              if constexpr (std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[Rank() - 1] = fft_size_;
              }
              else {
                out_dims_[perm_[Rank()-1]] = fft_size_;
              }
            } 
            else if constexpr (Type == detail::FFTType::R2C) {
              // R2C transforms pack the results in fft_size_/2 + 1 complex elements
              if constexpr (std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[Rank() - 1] = fft_size_ / 2 + 1;
              }
              else {
                out_dims_[perm_[Rank()-1]] = fft_size_ / 2 + 1;
              }
            }
            else {
              if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[perm_[Rank()-1]] = 2 * (fft_size_ - 1);
              }
              else {
                out_dims_[Rank() - 1] = 2 * (fft_size_ - 1);
              }                
            }
          }
          else { 
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
            if constexpr (Type == detail::FFTType::C2C) { 
              if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
                fft_size_ = a.Size(perm_[Rank()-1]);
              } else {
                fft_size_ = a.Size(a.Rank()-1);
              }              
            }
            else if constexpr (Type == detail::FFTType::R2C) { // R2C is N/2+1
              if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[perm_[Rank()-1]] = out_dims_[perm_[Rank()-1]] / 2 + 1;
              }
              else {
                out_dims_[Rank() - 1] = out_dims_[Rank() - 1] / 2 + 1;
              }
            }
            else {
              // R2C is 2*(N-1) unless overridden
              if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
                out_dims_[perm_[Rank()-1]] = 2 * (out_dims_[perm_[Rank()-1]] - 1);
              }
              else {
                out_dims_[Rank() - 1] = 2 * (out_dims_[Rank() - 1] - 1);
              }              
            }
          }
        }

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }
                  
        template <detail::ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<EPT>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          // 1. Determine if the binary operation ITSELF intrinsically has this capability.
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_));
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          if constexpr (std::is_same_v<PermDims, no_permute_t>) {
            if constexpr (Direction == detail::FFTDirection::FORWARD) {
              fft_impl(cuda::std::get<0>(out), a_, fft_size_, norm_, ex);
            }
            else {
              ifft_impl(cuda::std::get<0>(out), a_, fft_size_, norm_, ex);
            }
          }
          else {
            if constexpr (Direction == detail::FFTDirection::FORWARD) { 
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
   * prototypes with index_t. However, the value of fft_size must still fit into an index_t.
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
  template <typename OpA>
  __MATX_INLINE__ auto fft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
    constexpr auto fft_type = detail::ComplexInType<OpA>();
    const index_t fft_size_ = static_cast<index_t>(fft_size);
    return detail::FFTOp<OpA, detail::no_permute_t, detail::FFTDirection::FORWARD, fft_type>(a, fft_size_, detail::no_permute_t{}, norm);
  }

  /**
   * Run a 1D FFT with a cached plan
   *
   * Creates a new FFT plan in the cache if none exists, and uses that to execute
   * the 1D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
   * match
   * Note: fft_size must be unsigned so that the axis overload does not match both 
   * prototypes with index_t. However, the value of fft_size must still fit into an index_t.
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
    constexpr auto fft_type = detail::ComplexInType<OpA>();
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);
    const index_t fft_size_ = static_cast<index_t>(fft_size);
    return detail::FFTOp<OpA, decltype(perm), detail::FFTDirection::FORWARD, fft_type>(a, fft_size_, perm, norm);
  }

  /**
   * R2C FFT
   *
   * Performs an R2C FFT.
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
   __MATX_INLINE__ auto rfft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
     static_assert(!is_complex_v<typename remove_cvref_t<OpA>::value_type>, "RFFT only supports real input");
     const index_t fft_size_ = static_cast<index_t>(fft_size);
     return detail::FFTOp<OpA, detail::no_permute_t, detail::FFTDirection::FORWARD, detail::FFTType::R2C>(a, fft_size_, detail::no_permute_t{}, norm);
   }
 
   /**
    * R2C FFT
    *
    * Performs an R2C FFT.
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
   __MATX_INLINE__ auto rfft(const OpA &a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
     static_assert(!is_complex_v<typename remove_cvref_t<OpA>::value_type>, "RFFT only supports real input");
     auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);
     const index_t fft_size_ = static_cast<index_t>(fft_size);
     return detail::FFTOp<OpA, decltype(perm), detail::FFTDirection::FORWARD, detail::FFTType::R2C>(a, fft_size_, perm, norm);
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
   * prototypes with index_t. However, the value of fft_size must still fit into an index_t.
   * @param a
   *   input tensor or operator
   * @param fft_size
   *   Size of FFT. Setting to 0 uses the output size to figure out the FFT size.
   * @param norm
   *   Normalization to apply to IFFT
   */
  template<typename OpA>
  __MATX_INLINE__ auto ifft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
    const index_t fft_size_ = static_cast<index_t>(fft_size);
    return detail::FFTOp<OpA, detail::no_permute_t, detail::FFTDirection::BACKWARD, detail::FFTType::C2C>(a, fft_size_, detail::no_permute_t{} , norm);
  }

  /**
   * Run a 1D IFFT with a cached plan
   *
   * Creates a new FFT Iplan in the cache if none exists, and uses that to execute
   * the 1D IFFT. Note that FFTs and IFFTs share the same plans if all dimensions
   * match
   * Note: fft_size must be unsigned so that the axis overload does not match both 
   * prototypes with index_t. However, the value of fft_size must still fit into an index_t.
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
    const index_t fft_size_ = static_cast<index_t>(fft_size);
    return detail::FFTOp<OpA, decltype(perm), detail::FFTDirection::BACKWARD, detail::FFTType::C2C>(a, fft_size_, perm, norm);
  }

  /**
   * C2R inverse FFT
   *
   * Performs an inverse FFT on a complex operator to produce a complex operator.
   *
   * @tparam OpA
   *   Input tensor or operator type
   * 
   * Note: fft_size must be unsigned so that the axis overload does not match both 
   * prototypes with index_t. However, the value of fft_size must still fit into an index_t.
   * @param a
   *   input tensor or operator
   * @param fft_size
   *   Size of FFT. Setting to 0 uses the output size to figure out the FFT size.
   * @param norm
   *   Normalization to apply to IFFT
   */
   template<typename OpA>
   __MATX_INLINE__ auto irfft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
     const index_t fft_size_ = static_cast<index_t>(fft_size);
     return detail::FFTOp<OpA, detail::no_permute_t, detail::FFTDirection::BACKWARD, detail::FFTType::C2R>(a, fft_size_, detail::no_permute_t{} , norm);
   }
 
   /**
    * C2R inverse FFT
    *
    * Performs an inverse FFT on a complex operator to produce a real operator.
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
   __MATX_INLINE__ auto irfft(const OpA &a, const int32_t (&axis)[1], uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
     auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);
     const index_t fft_size_ = static_cast<index_t>(fft_size);
     return detail::FFTOp<OpA, decltype(perm), detail::FFTDirection::BACKWARD, detail::FFTType::C2R>(a, fft_size_, perm, norm);
   }    


  namespace detail {
    template <typename OpA, typename PermDims, FFTDirection Direction, FFTType Type>
    class FFT2Op : public BaseOp<FFT2Op<OpA, PermDims, Direction, Type>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        PermDims perm_;
        FFTNorm norm_;
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        using ttype = std::conditional_t<is_complex_v<typename OpA::value_type>, 
                                          typename OpA::value_type, 
                                          typename scalar_to_complex<typename OpA::value_type>::ctype>;
        // This should be tensor_impl_t, but need to work around issues with temp types returned in fft
        mutable detail::tensor_impl_t<ttype, OpA::Rank()> tmp_out_; 
        mutable ttype *ptr = nullptr;
        mutable bool prerun_done_ = false;                                                

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
        using matx_transform_op = bool;
        using fft2_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          if constexpr (Direction == detail::FFTDirection::FORWARD) {
            return "fft2(" + get_type_str(a_) + ")";
          }
          else {
            return "ifft2(" + get_type_str(a_) + ")";
          }
        }

        __MATX_INLINE__ FFT2Op(const OpA &a, PermDims perm, FFTNorm norm) : a_(a),  perm_(perm), norm_(norm) {
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
          }

          if constexpr (Type == detail::FFTType::C2C) {
            if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
              out_dims_[perm_[0]] = out_dims_[perm_[0]];
              out_dims_[perm_[1]] = out_dims_[perm_[1]];
            }
          }
          else if constexpr (Type == detail::FFTType::R2C) {
            if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
              out_dims_[perm_[0]] = out_dims_[perm_[0]] / 2 + 1;
              out_dims_[perm_[1]] = out_dims_[perm_[1]];
            }
            else {
              out_dims_[Rank() - 1] = out_dims_[Rank() - 1] / 2 + 1;
              out_dims_[Rank() - 2] = out_dims_[Rank() - 2];
            }
          }
          else {
            if constexpr (!std::is_same_v<PermDims, no_permute_t>) {
              out_dims_[perm_[0]] = 2 * (out_dims_[perm_[0]] - 1);
              out_dims_[perm_[1]] = out_dims_[perm_[1]];
            }
            else {
              out_dims_[Rank() - 1] = 2 * (out_dims_[Rank() - 1] - 1);
              out_dims_[Rank() - 2] = out_dims_[Rank() - 2];
            }
          }
        }

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <detail::ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<EPT>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<detail::ElementsPerThread::ONE>(indices...);
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
            if constexpr (Direction == detail::FFTDirection::FORWARD) { 
              fft2_impl(cuda::std::get<0>(out), a_, norm_, ex);
            }
            else {
              ifft2_impl(cuda::std::get<0>(out), a_, norm_, ex);
            }
          }
          else {
            if constexpr (Direction == detail::FFTDirection::FORWARD) { 
              fft2_impl(permute(cuda::std::get<0>(out), perm_), permute(a_, perm_), norm_, ex);
            }
            else {
              ifft2_impl(permute(cuda::std::get<0>(out), perm_), permute(a_, perm_), norm_, ex);
            }
          }
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          // 1. Determine if the binary operation ITSELF intrinsically has this capability.
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_));
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
    constexpr auto fft_type = detail::ComplexInType<OpA>();
    return detail::FFT2Op<OpA, detail::no_permute_t, detail::FFTDirection::FORWARD, fft_type>(a, detail::no_permute_t{}, norm);
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
    constexpr auto fft_type = detail::ComplexInType<OpA>();
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFT2Op<OpA, decltype(perm), detail::FFTDirection::FORWARD, fft_type>(a, perm, norm);
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
    constexpr auto fft_type = detail::ComplexInType<OpA>();
    return detail::FFT2Op<OpA, detail::no_permute_t, detail::FFTDirection::BACKWARD, fft_type>(a, detail::no_permute_t{}, norm);
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
    constexpr auto fft_type = detail::ComplexInType<OpA>();
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);  
    return detail::FFT2Op<OpA, decltype(perm), detail::FFTDirection::BACKWARD, fft_type>(a, perm, norm);
  }  

/**
 * Run a 2D RFFT (real-to-complex FFT)
 *
 * Performs a 2D RFFT.
 *
 * @tparam OpA
 *   Input tensor or operator type (must be real-valued)
 * @param a
 *   Input tensor or operator
 * @param norm
 *   Normalization to apply to FFT
 */
template<typename OpA>
__MATX_INLINE__ auto rfft2(const OpA &a, FFTNorm norm = FFTNorm::BACKWARD) {
  static_assert(!is_complex_v<typename remove_cvref_t<OpA>::value_type>, "rfft2 only supports real input");
  return detail::FFT2Op<OpA, detail::no_permute_t, detail::FFTDirection::FORWARD, detail::FFTType::R2C>(a, detail::no_permute_t{}, norm);
}

/**
 * Run a 2D RFFT (real-to-complex FFT) along specified axes
 *
 * Performs a 2D RFFT along the given axes.
 *
 * @tparam OpA
 *   Input tensor or operator type (must be real-valued)
 * @param a
 *   Input tensor or operator
 * @param axis
 *   Axes to perform FFT along
 * @param norm
 *   Normalization to apply to FFT
 */
template<typename OpA>
__MATX_INLINE__ auto rfft2(const OpA &a, const int32_t (&axis)[2], FFTNorm norm = FFTNorm::BACKWARD) {
  static_assert(!is_complex_v<typename remove_cvref_t<OpA>::value_type>, "rfft2 only supports real input");
  auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);
  return detail::FFT2Op<OpA, decltype(perm), detail::FFTDirection::FORWARD, detail::FFTType::R2C>(a, perm, norm);
}

/**
 * Run a 2D IRFFT (complex-to-real inverse FFT)
 *
 * Performs a 2D IRFFT.
 *
 * @tparam OpA
 *   Input tensor or operator type (must be complex-valued)
 * @param a
 *   Input tensor or operator
 * @param norm
 *   Normalization to apply to IFFT
 */
template<typename OpA>
__MATX_INLINE__ auto irfft2(const OpA &a, FFTNorm norm = FFTNorm::BACKWARD) {
  static_assert(is_complex_v<typename remove_cvref_t<OpA>::value_type>, "irfft2 only supports complex input");
  return detail::FFT2Op<OpA, detail::no_permute_t, detail::FFTDirection::BACKWARD, detail::FFTType::C2R>(a, detail::no_permute_t{}, norm);
}

/**
 * Run a 2D IRFFT (complex-to-real inverse FFT) along specified axes
 *
 * Performs a 2D IRFFT along the given axes.
 *
 * @tparam OpA
 *   Input tensor or operator type (must be complex-valued)
 * @param a
 *   Input tensor or operator
 * @param axis
 *   Axes to perform IFFT along
 * @param norm
 *   Normalization to apply to IFFT
 */
template<typename OpA>
__MATX_INLINE__ auto irfft2(const OpA &a, const int32_t (&axis)[2], FFTNorm norm = FFTNorm::BACKWARD) {
  static_assert(is_complex_v<typename remove_cvref_t<OpA>::value_type>, "irfft2 only supports complex input");
  auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);
  return detail::FFT2Op<OpA, decltype(perm), detail::FFTDirection::BACKWARD, detail::FFTType::C2R>(a, perm, norm);
}
}