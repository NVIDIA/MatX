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

#include <unordered_map>
#include <string>
#include "matx/core/type_utils.h"
#include "matx/core/utils.h"
#include "matx/operators/base_operator.h"
#include "matx/core/operator_options.h"

#ifndef __CUDACC_RTC__
  #include "matx/transforms/fft/fft_cuda.h"
  #ifdef MATX_EN_CPU_FFT
    #include "matx/transforms/fft/fft_fftw.h"
  #endif  
#endif

#ifdef MATX_EN_MATHDX
  #include "cuComplex.h"
  #include "matx/transforms/fft/fft_cufftdx.h"
#endif


namespace matx
{
  namespace detail {

    template <typename OpA, typename PermDims, typename FFTDirection>
    class FFTOp : public BaseOp<FFTOp<OpA, PermDims, FFTDirection>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        index_t fft_size_;
        PermDims perm_;
        FFTDirection direction_;
        FFTNorm norm_;    
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        using ttype = std::conditional_t<is_complex_v<typename OpA::value_type>, 
                                          typename OpA::value_type, 
                                          typename scalar_to_complex<typename OpA::value_type>::ctype>;
        // This should be tensor_impl_t, but need to work around issues with temp types returned in fft
        mutable ::matx::detail::tensor_impl_t<ttype, OpA::Rank()> tmp_out_;
        mutable ttype *ptr = nullptr;    

      public:
        using matxop = bool;
        using input_type = typename OpA::value_type;        
        using value_type = std::conditional_t<is_complex_v<input_type>,
          input_type,
          typename scalar_to_complex<input_type>::ctype>;
        using matx_transform_op = bool;
        using fft_xform_op = bool;    

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_t<detail::base_type_t<OpA>> a_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{a_.ToJITStorage()};
        }
#endif             
        
        __MATX_INLINE__ std::string str() const { 
          if constexpr (std::is_same_v<FFTDirection, detail::fft_t>) {
            return "fft(" + get_type_str(a_) + ")";
          }
          else {
            return "ifft(" + get_type_str(a_) + ")";
          }
        }


        __MATX_INLINE__ FFTOp(const OpA &a, index_t size, PermDims perm, FFTDirection direction, FFTNorm norm) : 
            a_(a), fft_size_(size),  perm_(perm), direction_(direction), norm_(norm) {
          
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

#ifdef MATX_EN_JIT
        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string symbol_name = "JITFFTOp_";
          symbol_name += std::to_string(fft_size_);
          symbol_name += "_T";
          symbol_name += std::to_string(static_cast<int>(DeduceFFTTransformType<typename scalar_to_complex<typename OpA::value_type>::ctype, value_type>()));
          symbol_name += "_D";
          symbol_name += std::is_same_v<FFTDirection, detail::fft_t> ? std::string("F") : std::string("B");

          return symbol_name;
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          const std::string class_name = get_jit_class_name();
          return cuda::std::make_tuple(
             class_name, 
             std::string("template <typename OpA> class " + class_name + "  {\n") + 
                 "  struct JIT_Storage {\n" +
                 "    typename detail::inner_storage_t<detail::base_type_t<OpA>> a_;\n" +
                 "  };\n" +
                 "  JIT_Storage storage_;\n" +
                 "  constexpr static cuda::std::array<index_t, " + std::to_string(Rank()) + "> out_dims_ = { " + 
                 detail::array_to_string(out_dims_) + " };\n" +
                 "  constexpr static unsigned int fft_size_ = " + std::to_string(fft_size_) + ";\n" +
                 "  constexpr static bool fft_forward_ = " + std::to_string(static_cast<bool>(std::is_same_v<FFTDirection, detail::fft_t>)) + ";\n" +
                 "  template <typename CapType, typename... Is>\n" +
                 "  __MATX_INLINE__ __MATX_DEVICE__  decltype(auto) operator()(Is... indices) const\n" +
                 "  {\n" +
                 "    return detail::RunDxFFT1D<input_type, CapType, fft_size, fft_forward>(storage_.a_, indices...);\n" +
                 "  }\n" +
                 "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n" +
                 "  {\n" +
                 "    return OpA::Rank();\n" +
                 "  }\n" +
                 "  constexpr __MATX_INLINE__  __MATX_DEVICE__ index_t Size(int dim) const\n" +
                 "  {\n" +
                 "    constexpr index_t dim_ = dim;\n" +
                 "    return out_dims_[dim_];\n " +
                 "  }\n" +    
                 "};\n"
          );
        }
#endif

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
#ifdef __CUDA_ARCH__
        if constexpr (CapType::jit) {
          if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
            printf("threadIdx.x %d, CapType::ept %d, Size(Rank() - 1) %lld\n", threadIdx.x, static_cast<int>(CapType::ept), Size(Rank() - 1));
            return detail::GetJitSentinelValue<CapType, value_type>();
          }
        }
#endif
          // cuFFTDx Doesn't support CapType::ept == 1
          if constexpr (CapType::ept == detail::ElementsPerThread::ONE) {
            return tmp_out_.template operator()<CapType>(indices...);
          }
          else {
#if defined(__CUDA_ARCH__) && defined(__CUDACC_RTC__)
            return detail::RunDxFFT1D<input_type, CapType>(a_, indices...);
            //return tmp_out_.template operator()<CapType>(indices...);
#else
            return tmp_out_.template operator()<CapType>(indices...);
#endif
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }


        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          [[maybe_unused]] detail::FFTDirection dir = std::is_same_v<FFTDirection, detail::fft_t> ? 
                                            detail::FFTDirection::FORWARD : 
                                            detail::FFTDirection::BACKWARD;      

          if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)            
            return combine_capabilities<Cap>(cuFFTDxHelper<typename OpA::value_type>::GetShmRequired(fft_size_, FFTType::C2C, dir, in.ept), detail::get_operator_capability<Cap>(a_, in));
#else
            return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
            bool supported = true;
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
            if (((fft_size_ & (fft_size_ - 1)) != 0 || fft_size_ == 0) // Only support power-of-2 FFT sizes for JIT support
                || is_complex_half_v<typename OpA::value_type>  // No half support in MatX for fusion yet
                || !is_complex_v<typename OpA::value_type>) // Only support C2C for JIT support
            {
              supported = false;
            } 
            else {
              supported = cuFFTDxHelper<typename OpA::value_type>::IsSupported(fft_size_, FFTType::C2C, dir);
            }
#else
            supported = false;
#endif
            return combine_capabilities<Cap>(supported, detail::get_operator_capability<Cap>(a_, in));      
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {           
            // Get the capability string and add to map
            const auto [key, value] = get_jit_op_str();
      
            // Insert into the map if the key doesn't exist
            if (in.find(key) == in.end()) {
              in[key] = value;
            }
            
            // Also handle child operators
            detail::get_operator_capability<Cap>(a_, in);
            
            // Always return true for now
            return true;
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
            if (in.jit) {
              // Currently MatX only attempts to use the "best" EPT as returned by cuFFTDx. In the future we may
              // try other EPT values that yield different SHM values.
              if (cuFFTDxHelper<typename OpA::value_type>::IsSupported(fft_size_, FFTType::C2C, dir)) {
                return combine_capabilities<Cap>(cuFFTDxHelper<typename OpA::value_type>::GetEPTs(fft_size_, FFTType::C2C, dir), 
                        detail::get_operator_capability<Cap>(a_, in));
              }
              else {
                // If we're asking for JIT and the parameters aren't supported, return invalid EPT
                const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::INVALID, ElementsPerThread::INVALID};
                return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));                
              }
            }
            else {
              return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
            }
#else
            return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
          }
          else if constexpr (Cap == OperatorCapability::BLOCK_DIM) {
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
            return combine_capabilities<Cap>(cuFFTDxHelper<typename OpA::value_type>::GetBlockDim(fft_size_, FFTType::C2C, dir, in.ept), detail::get_operator_capability<Cap>(a_, in));
#else
            return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
          }
          else if constexpr (Cap == OperatorCapability::GENERATE_LTOIR) {
            printf("GENERATE_LTOIR\n");
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__) && !defined(__CUDACC_RTC__) && !defined(__CUDA_ARCH__)
            return combine_capabilities<Cap>(
                cuFFTDxHelper<typename OpA::value_type>::GenerateLTOIR(fft_size_, FFTType::C2C, dir, in.ept), 
                detail::get_operator_capability<Cap>(a_, in));
#else
            return combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
#endif
          }    
          else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
            // No need to use combine_capabilities here since we're just returning a string.
            const auto inner_op_jit_name = detail::get_operator_capability<Cap>(a_, in);
            return get_jit_class_name() + "<" + inner_op_jit_name + ">";
          }                
          else {
            // 1. Determine if the binary operation ITSELF intrinsically has this capability.
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
          }
        }

#ifndef __CUDACC_RTC__  
        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          if constexpr (std::is_same_v<PermDims, no_permute_t>) {
            if constexpr (std::is_same_v<FFTDirection, fft_t>) {
              fft_impl(cuda::std::get<0>(out), a_, fft_size_, norm_, ex);
            }
            else {
              ifft_impl(cuda::std::get<0>(out), a_, fft_size_, norm_, ex);
            }
          }
          else {
            if constexpr (std::is_same_v<FFTDirection, fft_t>) { 
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
          printf("prerun\n");
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
#endif          
    };
  }

#ifndef __CUDACC_RTC__
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
  template<typename OpA>
  __MATX_INLINE__ auto fft(const OpA &a, uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD) {
    const index_t fft_size_ = static_cast<index_t>(fft_size);

    return detail::FFTOp(a, fft_size_, detail::no_permute_t{}, detail::fft_t{}, norm);
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
    auto perm = detail::getPermuteDims<remove_cvref_t<OpA>::Rank()>(axis);
    const index_t fft_size_ = static_cast<index_t>(fft_size);
    return detail::FFTOp(a, fft_size_, perm, detail::fft_t{}, norm);
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
    return detail::FFTOp(a, fft_size_, detail::no_permute_t{} , detail::ifft_t{}, norm);
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
    return detail::FFTOp(a, fft_size_, perm, detail::ifft_t{}, norm);
  }  
#endif


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
        mutable ::matx::detail::tensor_impl_t<ttype, OpA::Rank()> tmp_out_; 
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
          return tmp_out_.template operator()<DefaultCapabilities>(indices...);
        }        

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() {
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
          return out_dims_[dim];
        }

#ifndef __CUDACC_RTC__
        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

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

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          // 1. Determine if the binary operation ITSELF intrinsically has this capability.
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
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
#endif
    };    
  }

#ifndef __CUDACC_RTC__
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
#endif
}

