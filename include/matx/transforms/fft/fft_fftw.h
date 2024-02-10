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

#include "matx/core/cache.h"
#include "matx/core/error.h"
#include "matx/core/make_tensor.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/executors/host.h"
#include "matx/transforms/fft/fft_common.h"
#include "matx/transforms/copy.h"
#include "matx/executors/support.h"
#ifdef MATX_EN_NVPL
#include <nvpl_fftw.h>
#endif

#include <cstdio>
#include <functional>
#include <optional>

namespace matx {

namespace detail {

/**
 * Parameters needed to execute an FFT/IFFT in cuFFT
 */
struct FftFFTWparams_t {
  int irank, orank;
  int n[MAX_FFT_RANK] = {0};
  int batch;
  int       batch_dims;
  int inembed[MAX_FFT_RANK] = {0};
  int onembed[MAX_FFT_RANK] = {0};
  int istride, ostride;
  int idist, odist;
  FFTType transform_type; // Known from input/output type, but still useful
  int fft_rank;
};


  template <typename OutTensorType, typename InTensorType>
  static FftFFTWparams_t GetFFTParams(OutTensorType &o,
                          const InTensorType &i, int fft_rank)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    FftFFTWparams_t params;
    constexpr auto RANK = OutTensorType::Rank();   
    using T1    = typename OutTensorType::scalar_type;
    using T2    = typename InTensorType::scalar_type;    

    params.irank = i.Rank();
    params.orank = o.Rank();

    params.transform_type = DeduceFFTTransformType<OutTensorType, InTensorType>();
    params.fft_rank =  fft_rank;

    if (fft_rank == 1) {
      params.batch_dims = 0;
      params.n[0] = (params.transform_type == FFTType::C2R ||
                      params.transform_type == FFTType::Z2D)
                        ? static_cast<int>(o.Size(RANK - 1))
                        : static_cast<int>(i.Size(RANK - 1));

      if (i.IsContiguous() && o.IsContiguous()) {
        params.batch = 1;
        for (int dim = i.Rank() - 2; dim >= 0; dim--) {
          params.batch_dims++;
          params.batch *= static_cast<int>(i.Size(dim));
        }
      }
      else {
        if (RANK == 1) {
          params.batch = 1;
          params.batch_dims = 0;
        }
        else {
          params.batch = static_cast<int>(TotalSize(i) / i.Size(RANK - 1));
          params.batch_dims = 1; 
        }
      }
      
      params.inembed[0] = static_cast<int>(i.Size(RANK - 1)); // Unused
      params.onembed[0] = static_cast<int>(o.Size(RANK - 1)); // Unused
      params.istride = static_cast<int>(i.Stride(RANK - 1));
      params.ostride = static_cast<int>(o.Stride(RANK - 1));
      params.idist = (RANK == 1) ? 1 : static_cast<int>(i.Stride(RANK - 2));  
      params.odist = (RANK == 1) ? 1 : static_cast<int>(o.Stride(RANK - 2));
    }
    else if (fft_rank == 2) {
      if (params.transform_type == FFTType::C2R ||
          params.transform_type == FFTType::Z2D) {
        params.n[1] = static_cast<int>(o.Size(RANK-1));
        params.n[0] = static_cast<int>(o.Size(RANK-2));
      }
      else {
        params.n[1] = static_cast<int>(i.Size(RANK-1));
        params.n[0] = static_cast<int>(i.Size(RANK-2));
      }

      params.batch = (RANK == 2) ? 1 : static_cast<int>(TotalSize(i) / ((i.Size(RANK - 1) * i.Size(RANK - 2))));
      params.inembed[1] = static_cast<int>(i.Size(RANK-1));
      params.onembed[1] = static_cast<int>(o.Size(RANK-1));
      params.istride = static_cast<int>(i.Stride(RANK-1));
      params.ostride = static_cast<int>(o.Stride(RANK-1));
      params.idist = (RANK<=2) ? 1 : (int) static_cast<int>(i.Stride(RANK-3));
      params.odist = (RANK<=2) ? 1 : (int) static_cast<int>(o.Stride(RANK-3));
    }

    if (params.fft_rank == 1) {
      if (params.transform_type == FFTType::C2R ||
          params.transform_type == FFTType::Z2D) {
        if (is_cuda_complex_v<T1> || !is_cuda_complex_v<T2>) {
          MATX_THROW(matxInvalidType, "FFT types inconsistent with C2R/Z2D transform");
        }
        if (params.n[0] != o.Size(OutTensorType::Rank()-1) ||
          (params.n[0]/2)+1 != i.Size(InTensorType::Rank()-1)) {
          MATX_THROW(matxInvalidSize, "Tensor sizes inconsistent with C2R/Z2D transform");
        }
      }
      else if (params.transform_type == FFTType::R2C ||
              params.transform_type == FFTType::D2Z) {
        if (is_cuda_complex_v<T2> || !is_cuda_complex_v<T1>) {
          MATX_THROW(matxInvalidType, "FFT types inconsistent with R2C/D2Z transform");
        }
        if (params.n[0] != i.Size(InTensorType::Rank()-1) ||
          (params.n[0]/2)+1 != o.Size(OutTensorType::Rank()-1)) {
          MATX_THROW(matxInvalidSize, "Tensor sizes inconsistent with R2C/D2Z transform");
        }
      }
      else {
        if (!is_complex_v<T2> || !is_complex_v<T1> || !std::is_same_v<T1, T2>) {
          MATX_THROW(matxInvalidType, "FFT types inconsistent with C2C transform");
        }
        if (params.n[0] != o.Size(OutTensorType::Rank()-1) ||
            params.n[0] != i.Size(OutTensorType::Rank()-1)) {
          MATX_THROW(matxInvalidSize, "Tensor sizes inconsistent with C2C transform");
        }
      }  
    }
    else {
      if (params.transform_type == FFTType::C2R ||
          params.transform_type == FFTType::Z2D) {
        MATX_ASSERT((o.Size(RANK-2) * (o.Size(RANK-1) / 2 + 1)) == i.Size(RANK-1) * i.Size(RANK-2),
                    matxInvalidSize);
        MATX_ASSERT(!is_cuda_complex_v<T1> && is_cuda_complex_v<T2>,
                    matxInvalidType);
      }
      else if (params.transform_type == FFTType::R2C ||
              params.transform_type == FFTType::D2Z) {
        MATX_ASSERT(o.Size(RANK-1) * o.Size(RANK-2) == (i.Size(RANK-2) * (i.Size(RANK-1) / 2 + 1)),
                    matxInvalidSize);
        MATX_ASSERT(!is_cuda_complex_v<T2> && is_cuda_complex_v<T1>,
                    matxInvalidType);
      }
      else {
        MATX_ASSERT((std::is_same_v<T1, T2>), matxInvalidType);
        MATX_ASSERT(is_complex_v<T2> && is_complex_v<T1>, matxInvalidType);
        MATX_ASSERT(o.Size(RANK-2) * o.Size(RANK-1) == i.Size(RANK-2) * i.Size(RANK-1),
                    matxInvalidSize);
      }

      for (int r = 0; r < RANK - 2; r++) {
        MATX_ASSERT(o.Size(r) == i.Size(r), matxInvalidSize);
      }         
    }

    MATX_ASSERT_STR(params.idist > 0, matxInvalidDim, "FFTs do not support batch strides of 0 (no cloned views)");

    return params;
  }


  template <typename TensorOp>
  __MATX_INLINE__ auto getFFTW1DSupportedTensor( const TensorOp &in) {

    constexpr int RANK=TensorOp::Rank();

    if constexpr ( !(is_tensor_view_v<TensorOp>)) {
      return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_HOST_MALLOC_MEMORY); 
    } else {

      bool supported = true;

      // If there are any unsupported layouts for fftw add them here
      if (supported) {
        return in;
      } else {
        return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_HOST_MALLOC_MEMORY); 
      }
    }
  }

  template <typename TensorOp>
  __MATX_INLINE__ auto getFFTW2DSupportedTensor( const TensorOp &in) {

    constexpr int RANK=TensorOp::Rank();

    if constexpr ( !is_tensor_view_v<TensorOp>) {
      return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_HOST_MALLOC_MEMORY); 
    } else {
      bool supported = true;

      // only a subset of strides are supported per fftw indexing scheme.
      if ( in.Stride(RANK-2) != in.Stride(RANK-1) * in.Size(RANK-1)) {
        supported = false;
      } else if constexpr ( RANK > 2) {
        if(in.Stride(RANK-3) != in.Size(RANK-2) * in.Stride(RANK-2)) {
          supported = false;
        }
      }
  
      if (supported) {
        return in;
      } else {
        return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_HOST_MALLOC_MEMORY); 
      }
    }
  }

  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void fft_exec([[maybe_unused]] OutputTensor &o, 
                                [[maybe_unused]] const InputTensor &i, 
                                [[maybe_unused]] const FftFFTWparams_t &params, 
                                [[maybe_unused]] detail::FFTDirection dir) {
    [[maybe_unused]] static constexpr bool fp32 = std::is_same_v<typename inner_op_type_t<typename OutputTensor::scalar_type>::type, float>;

#if MATX_EN_CPU_FFT
    auto fft_dir = (dir == detail::FFTDirection::FORWARD) ? FFTW_FORWARD : FFTW_BACKWARD;

    auto exec_plans = [&](typename OutputTensor::value_type *out_ptr, 
                          typename InputTensor::value_type *in_ptr) {
      fftwf_plan plan{};
      if constexpr (fp32) {
        if constexpr (DeduceFFTTransformType<OutputTensor, InputTensor>() == FFTType::C2C) {
          plan  = fftwf_plan_many_dft( params.fft_rank, 
                                      params.n, 
                                      params.batch, 
                                      reinterpret_cast<fftwf_complex*>(in_ptr), 
                                      params.inembed, 
                                      params.istride, 
                                      params.idist,
                                      reinterpret_cast<fftwf_complex*>(out_ptr), 
                                      params.onembed, 
                                      params.ostride, 
                                      params.odist,  
                                      fft_dir, 
                                      FFTW_ESTIMATE);
        }
        else if constexpr (DeduceFFTTransformType<OutputTensor, InputTensor>() == FFTType::C2R) {
          plan  = fftwf_plan_many_dft_c2r( params.fft_rank, 
                                      params.n, 
                                      params.batch, 
                                      reinterpret_cast<fftwf_complex*>(in_ptr), 
                                      params.inembed, 
                                      params.istride, 
                                      params.idist,
                                      out_ptr, 
                                      params.onembed, 
                                      params.ostride, 
                                      params.odist,  
                                      FFTW_ESTIMATE);
        }
        else if constexpr (DeduceFFTTransformType<OutputTensor, InputTensor>() == FFTType::R2C) {        
          plan  = fftwf_plan_many_dft_r2c( params.fft_rank, 
                                      params.n, 
                                      params.batch, 
                                      in_ptr, 
                                      params.inembed, 
                                      params.istride, 
                                      params.idist,
                                      reinterpret_cast<fftwf_complex*>(out_ptr), 
                                      params.onembed, 
                                      params.ostride, 
                                      params.odist,  
                                      FFTW_ESTIMATE);
        }

        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        fftwf_cleanup();         
      }
      else {
        if constexpr (DeduceFFTTransformType<OutputTensor, InputTensor>() == FFTType::Z2Z) {
          plan  = fftw_plan_many_dft( params.fft_rank, 
                                      params.n, 
                                      params.batch, 
                                      reinterpret_cast<fftw_complex*>(in_ptr), 
                                      params.inembed, 
                                      params.istride, 
                                      params.idist,
                                      reinterpret_cast<fftw_complex*>(out_ptr), 
                                      params.onembed, 
                                      params.ostride, 
                                      params.odist,  
                                      fft_dir, 
                                      FFTW_ESTIMATE);
        }
        else if constexpr (DeduceFFTTransformType<OutputTensor, InputTensor>() == FFTType::Z2D) {
          plan  = fftw_plan_many_dft_c2r( params.fft_rank, 
                                      params.n, 
                                      params.batch, 
                                      reinterpret_cast<fftw_complex*>(in_ptr), 
                                      params.inembed, 
                                      params.istride, 
                                      params.idist,
                                      out_ptr, 
                                      params.onembed, 
                                      params.ostride, 
                                      params.odist,  
                                      FFTW_ESTIMATE);
        }
        else if constexpr (DeduceFFTTransformType<OutputTensor, InputTensor>() == FFTType::D2Z) {
          plan  = fftw_plan_many_dft_r2c( params.fft_rank, 
                                      params.n, 
                                      params.batch, 
                                      in_ptr, 
                                      params.inembed, 
                                      params.istride, 
                                      params.idist,
                                      reinterpret_cast<fftw_complex*>(out_ptr), 
                                      params.onembed, 
                                      params.ostride, 
                                      params.odist,  
                                      FFTW_ESTIMATE);
        } 
        
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        fftw_cleanup();           
      }
    };

    exec_plans(o.Data(), i.Data());
#endif
  }

  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void fft1d_dispatch(OutputTensor o, const InputTensor i,
          uint64_t fft_size, detail::FFTDirection dir, FFTNorm norm, const HostExecutor &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(exec.GetNumThreads() == 1, matxInvalidParameter, "Only single-threaded host FFT supported");
    MATX_ASSERT_STR(TotalSize(i) < std::numeric_limits<int>::max(), matxInvalidSize, "Dimensions too large for host FFT currently");

    // converts operators to tensors
    auto out = getFFTW1DSupportedTensor(o);
    auto in_t = getFFTW1DSupportedTensor(i); 
    
    if(!in_t.isSameView(i)) {
      (in_t = i).run(exec);
    }
  
    auto in = detail::GetFFTInputView(out, in_t, fft_size, exec);

    // Get parameters required by these tensors
    auto params = GetFFTParams(out, in, 1);

    fft_exec(o, in, params, dir);

    if(!out.isSameView(o)) {
      (o = out).run(exec);
    }

    using s_type = typename detail::value_promote_t<typename inner_op_type_t<typename InputTensor::scalar_type>::type>;
    s_type factor;
    constexpr s_type s_one = static_cast<s_type>(1.0);

    if (dir == detail::FFTDirection::FORWARD) {
      factor = static_cast<s_type>(params.n[0]);

      if (norm == FFTNorm::ORTHO) {
        (o *= s_one / std::sqrt(factor)).run(exec);
      } else if (norm == FFTNorm::FORWARD) {
        (o *= s_one / factor).run(exec);
      }
    }
    else {
      factor = static_cast<s_type>(params.n[0]);

      if (norm == FFTNorm::ORTHO) {
        (o *= s_one / std::sqrt(factor)).run(exec);
      } else if (norm == FFTNorm::BACKWARD) {
        (o *= s_one / factor).run(exec);
      }    
    }  
  }


  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void fft2d_dispatch(OutputTensor o, const InputTensor i,
          detail::FFTDirection dir, FFTNorm norm, const HostExecutor &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_ASSERT_STR(TotalSize(i) < std::numeric_limits<int>::max(), matxInvalidSize, "Dimensions too large for host FFT currently");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(exec.GetNumThreads() == 1, matxInvalidParameter, "Only single-threaded host FFT supported");

    // converts operators to tensors
    auto out = getFFTW2DSupportedTensor(o);
    auto in = getFFTW2DSupportedTensor(i); 
    
    if(!in.isSameView(i)) {
      (in = i).run(exec);
    }

    // Get parameters required by these tensors
    auto params = GetFFTParams(out, in, 2);

    fft_exec(o, in, params, dir);

    if(!out.isSameView(o)) {
      (o = out).run(exec);
    }

    using s_type = typename detail::value_promote_t<typename inner_op_type_t<typename InputTensor::scalar_type>::type>;
    s_type factor;
    constexpr s_type s_one = static_cast<s_type>(1.0);

    if (dir == detail::FFTDirection::FORWARD) {
      factor = static_cast<s_type>(params.n[0] * params.n[1]);

      if (norm == FFTNorm::ORTHO) {
        (o *= s_one / std::sqrt(factor)).run(exec);
      } else if (norm == FFTNorm::FORWARD) {
        (o *= s_one / factor).run(exec);
      }
    }
    else {
      factor = static_cast<s_type>(params.n[0] * params.n[1]);

      if (norm == FFTNorm::ORTHO) {
        (o *= s_one / std::sqrt(factor)).run(exec);
      } else if (norm == FFTNorm::BACKWARD) {
        (o *= s_one / factor).run(exec);
      }    
    }  
  }  


  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void fft_impl(OutputTensor o, const InputTensor i,
          uint64_t fft_size, FFTNorm norm, const HostExecutor &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_STATIC_ASSERT_STR( (std::is_same_v<typename inner_op_type_t<typename OutputTensor::scalar_type>::type, float> ||
                            std::is_same_v<typename inner_op_type_t<typename  InputTensor::scalar_type>::type, double>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");    
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft1d_dispatch(o, i, fft_size, FFTDirection::FORWARD, norm, exec);
  }


  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void ifft_impl(OutputTensor o, const InputTensor i,
          uint64_t fft_size, FFTNorm norm, const HostExecutor &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_STATIC_ASSERT_STR( (std::is_same_v<typename inner_op_type_t<typename OutputTensor::scalar_type>::type, float> ||
                            std::is_same_v<typename inner_op_type_t<typename  InputTensor::scalar_type>::type, double>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");      
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft1d_dispatch(o, i, fft_size, FFTDirection::BACKWARD, norm, exec);
  }



  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void fft2_impl(OutputTensor o, const InputTensor i, FFTNorm norm, 
            const HostExecutor &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");
    MATX_STATIC_ASSERT_STR(InputTensor::Rank() >= 2, matxInvalidSize, "2D FFT must be rank 2 tensor or higher");      
    MATX_STATIC_ASSERT_STR( (std::is_same_v<typename inner_op_type_t<typename OutputTensor::scalar_type>::type, float> ||
                            std::is_same_v<typename inner_op_type_t<typename  InputTensor::scalar_type>::type, double>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");     
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft2d_dispatch(o, i, FFTDirection::FORWARD, norm, exec);
  }

  template <typename OutputTensor, typename InputTensor>
  __MATX_INLINE__ void ifft2_impl(OutputTensor o, const InputTensor i, FFTNorm norm, 
            const HostExecutor &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_STATIC_ASSERT_STR(InputTensor::Rank() >= 2, matxInvalidSize, "2D FFT must be rank 2 tensor or higher");      
    MATX_STATIC_ASSERT_STR( (std::is_same_v<typename inner_op_type_t<typename OutputTensor::scalar_type>::type, float> ||
                            std::is_same_v<typename inner_op_type_t<typename  InputTensor::scalar_type>::type, double>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");      
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft2d_dispatch(o, i, FFTDirection::BACKWARD, norm, exec);  
  }
    
} // end namespace detail



}; // end namespace matx
