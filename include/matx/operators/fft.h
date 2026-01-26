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
#include "matx/core/log.h"

#include "matx/transforms/fft/fft_cuda.h"
#ifdef MATX_EN_CPU_FFT
  #include "matx/transforms/fft/fft_fftw.h"
#endif  

#if defined(MATX_EN_MATHDX) && defined (__CUDACC__)
  #include "cuComplex.h"
  #include "matx/transforms/fft/fft_cufftdx.h"
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
#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
        mutable cuFFTDxHelper<typename OpA::value_type> dx_fft_helper_;
#endif

      public:
        using matxop = bool;
        using input_type = typename OpA::value_type;        
        using value_type = std::conditional_t<is_complex_v<input_type>,
          input_type,
          typename scalar_to_complex<input_type>::ctype>;
        using matx_transform_op = bool;
        using fft_xform_op = bool;
        using can_alias = bool; // FFTs can use same input/output memory

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(a_)};
        }
#endif             
        
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
          MATX_LOG_TRACE("{} constructor: fft_size={}, norm={}", str(), size, static_cast<int>(norm));
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

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
            int major = 0;
            int minor = 0;
            int device;
            cudaGetDevice(&device);            
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
            int cc = major * 100 + minor;          
            dx_fft_helper_.set_fft_size(fft_size_);
            dx_fft_helper_.set_fft_type(DeduceFFTTransformType<typename scalar_to_complex<typename OpA::value_type>::ctype, value_type>());
            dx_fft_helper_.set_direction(Direction);
            dx_fft_helper_.set_cc(cc);
            // if (fft_size_ <= 32) {
            //   dx_fft_helper_.set_method(cuFFTDxMethod::REGISTER);
            // } else {
              dx_fft_helper_.set_method(cuFFTDxMethod::SHARED);
            //}            

            bool contiguous = false;
            if constexpr (is_tensor_view_v<OpA>) {
              contiguous = a_.IsContiguous();
            }
            dx_fft_helper_.set_contiguous_input(contiguous);
#endif
        }

#if defined(MATX_EN_MATHDX) && defined (__CUDACC__)
        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string symbol_name = "JITFFTOp_";
          symbol_name += std::to_string(fft_size_);
          symbol_name += "_T";
          symbol_name += std::to_string(static_cast<int>(DeduceFFTTransformType<typename scalar_to_complex<typename OpA::value_type>::ctype, value_type>()));
          symbol_name += "_D";
          symbol_name += Direction == detail::FFTDirection::FORWARD ? std::string("F") : std::string("B");

          return symbol_name;
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          const std::string class_name = get_jit_class_name();

          const std::string fft_func_name = std::string(FFT_DX_FUNC_PREFIX) + "_" + dx_fft_helper_.GetSymbolName();
         
          return cuda::std::make_tuple(
             class_name, 
             std::string(
                 " extern \"C\" __device__ void " + fft_func_name + "(" + detail::type_to_string<input_type>() + "*);\n" +
                 " template <typename OpA> struct " + class_name + "  {\n" +
                 "  using input_type = typename OpA::value_type;\n" +
                 "  using matxop = bool;\n" +
                 "  using value_type = cuda::std::conditional_t<is_complex_v<input_type>, input_type, typename scalar_to_complex<input_type>::ctype>;\n" +
                 "  typename detail::inner_storage_or_self_t<detail::base_type_t<OpA>> a_;\n" +
                 "  constexpr static cuda::std::array<index_t, " + std::to_string(Rank()) + "> out_dims_ = { " + 
                 detail::array_to_string(out_dims_) + " };\n" +             
                 "  template <typename CapType, typename... Is>\n" +
                 "  __MATX_INLINE__ __MATX_DEVICE__  decltype(auto) operator()(Is... indices) const\n" +
                 "  {\n" +
                 "    " + dx_fft_helper_.GetFuncStr(fft_func_name, static_cast<int>(norm_)) + "\n" +
                 "  }\n" +
                 "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n" +
                 "  {\n" +
                 "    return OpA::Rank();\n" +
                 "  }\n" +
                 "  constexpr __MATX_INLINE__  __MATX_DEVICE__ index_t Size(int dim) const\n" +
                 "  {\n" +
                 "    return out_dims_[dim];\n " +
                 "  }\n" +    
                 "};\n")
          );
        }     
#endif

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<CapType>(indices...);
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
#if defined(MATX_EN_MATHDX) && defined (__CUDACC__)
          // Branch with cuFFTDx support
          if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            auto result = combine_capabilities<Cap>(dx_fft_helper_.GetShmRequired(), detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("DYN_SHM_SIZE: {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
            bool supported = true;
            if (!dx_fft_helper_.template CheckJITSizeAndTypeRequirements<OpA>()) {
              supported = false;
            } 
            else {
              supported = dx_fft_helper_.IsSupported();
            }

            auto result = combine_capabilities<Cap>(supported, detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("SUPPORTS_JIT: {}", result);
            return result;      
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
            MATX_LOG_DEBUG("JIT_CLASS_QUERY: true");
            return true;
          }
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            if (in.jit) {
              // Currently MatX only attempts to use the "best" EPT as returned by cuFFTDx. In the future we may
              // try other EPT values that yield different SHM values.
              if (dx_fft_helper_.IsSupported()) {
                auto epts = dx_fft_helper_.GetEPTs();
                // epts[0] = ElementsPerThread::EIGHT;
                // epts[1] = ElementsPerThread::EIGHT;
                auto result = combine_capabilities<Cap>(epts, detail::get_operator_capability<Cap>(a_, in));
                MATX_LOG_DEBUG("ELEMENTS_PER_THREAD (JIT supported): [{},{}]", static_cast<int>(result[0]), static_cast<int>(result[1]));
                return result;
              }
              else {
                // If we're asking for JIT and the parameters aren't supported, return invalid EPT
                const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::INVALID, ElementsPerThread::INVALID};
                auto result = combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(a_, in));
                MATX_LOG_DEBUG("ELEMENTS_PER_THREAD (JIT unsupported): [{},{}]", static_cast<int>(result[0]), static_cast<int>(result[1]));
                return result;                
              }
            }
            else {
              auto result = combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
              MATX_LOG_DEBUG("ELEMENTS_PER_THREAD (non-JIT): [{},{}]", static_cast<int>(result[0]), static_cast<int>(result[1]));
              return result;
            }
          }
          else if constexpr (Cap == OperatorCapability::GROUPS_PER_BLOCK) {
            int ffts_per_block_candidate;

            if constexpr (Rank() > 1) {
              const int ffts_per_block = dx_fft_helper_.GetFFTsPerBlock();
              const auto last_dim = a_.Size(a_.Rank() - 2);
              ffts_per_block_candidate = ffts_per_block;
              // Try to find an ffts_per_block that evenly divides into last dimension size
              // Decrease ffts_per_block until it divides evenly or until 1
              while (ffts_per_block_candidate > 1 && (last_dim % ffts_per_block_candidate != 0)) {
                --ffts_per_block_candidate;
              }
              MATX_LOG_DEBUG("GROUPS_PER_BLOCK from cuFFTDx: [{},{}]", ffts_per_block, ffts_per_block_candidate);
            }
            else {
              ffts_per_block_candidate = 1;
            }
            
            cuda::std::array<int, 2> groups_per_block = {ffts_per_block_candidate, ffts_per_block_candidate};
            auto result = combine_capabilities<Cap>(groups_per_block, detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("GROUPS_PER_BLOCK: [{},{}]", result[0], result[1]);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::SET_ELEMENTS_PER_THREAD) {
            dx_fft_helper_.set_current_elements_per_thread(in.ept);
            auto result = combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("SET_ELEMENTS_PER_THREAD: {}", static_cast<int>(in.ept));
            return result;
          }
          else if constexpr (Cap == OperatorCapability::GLOBAL_KERNEL) {
            // If MathDx is enabled we always return false. Other checks on size and type may prevent JIT compilation.
            MATX_LOG_DEBUG("GLOBAL_KERNEL: false");            
            return false;
          }
          else if constexpr (Cap == OperatorCapability::SET_GROUPS_PER_BLOCK) {
            dx_fft_helper_.set_ffts_per_block(in.groups_per_block);
            auto result = combine_capabilities<Cap>(capability_attributes<Cap>::default_value, detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("SET_GROUPS_PER_BLOCK: {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::BLOCK_DIM) {
            auto result = dx_fft_helper_.GetBlockDim();
            MATX_LOG_DEBUG("cuFFTDx block dim: {}", result);
            const auto my_block = cuda::std::array<int, 2>{result, result};
            return combine_capabilities<Cap>(my_block, detail::get_operator_capability<Cap>(a_, in));
          }
          else if constexpr (Cap == OperatorCapability::GENERATE_LTOIR) {
            auto result = combine_capabilities<Cap>(
                dx_fft_helper_.GenerateLTOIR(in.ltoir_symbols), 
                detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("GENERATE_LTOIR: {}", result);
            return result;
          }    
          else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
            // No need to use combine_capabilities here since we're just returning a string.
            const auto inner_op_jit_name = detail::get_operator_capability<Cap>(a_, in);
            auto result = get_jit_class_name() + "<" + inner_op_jit_name + ">";
            MATX_LOG_DEBUG("JIT_TYPE_QUERY: {}", result);
            return result;
          }
          else if constexpr (Cap == OperatorCapability::ASYNC_LOADS_REQUESTED) {
            // If this is a contiguous tensor input we want to do an async load so that we decrease register pressure. 
            // and increase bandwidth on newer architectures
            bool async_loads_requested = false;
            if constexpr (is_tensor_view_v<OpA>) {
              if (a_.IsContiguous()) {
                async_loads_requested = true;
              }
            }
            auto result = combine_capabilities<Cap>(async_loads_requested, detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("ASYNC_LOADS_REQUESTED: {}", result);
            return result;
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            auto result = combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
            return result;
          }
#else
          // Branch without cuFFTDx support
          if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
            bool supported = false;
            auto result = combine_capabilities<Cap>(supported, detail::get_operator_capability<Cap>(a_, in));
            MATX_LOG_DEBUG("SUPPORTS_JIT (no cuFFTDx): {}", result);
            return result;      
          } 
          else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
            MATX_LOG_DEBUG("JIT_TYPE_QUERY (no cuFFTDx): \"\"");
            return "";
          }
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            MATX_LOG_DEBUG("ALIASED_MEMORY (no cuFFTDx): false");
            // FFTs with cuFFT cannot have aliased memory errors since they allow the same input and output. Do not 
            // pass this property on to the child operators.
            return false;
          }
          else {
            // 1. Determine if the binary operation ITSELF intrinsically has this capability.
            auto self_has_cap = capability_attributes<Cap>::default_value;
            auto result = combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
            return result;
          }
#endif
        }

        __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

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
          MATX_LOG_TRACE("{} constructor: norm={}", str(), static_cast<int>(norm));
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

        

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
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


        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

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

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          // 1. Determine if the binary operation ITSELF intrinsically has this capability.
          auto self_has_cap = capability_attributes<Cap>::default_value;
          auto result = combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_, in));
          return result;
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
