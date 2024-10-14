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
#ifdef MATX_EN_X86_FFTW
#include <fftw3.h>
#endif
#ifdef MATX_EN_OMP
#include <omp.h>
#endif
#include <cstdio>
#include <functional>
#include <optional>
#include <cuda/atomic>

namespace matx {

namespace detail {

/**
 * Parameters needed to execute an FFT/IFFT in FFTW
 */
struct FftFFTWParams_t {
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
  bool is_fp32;
  bool in_place;
  detail::FFTDirection dir;
};

  template <typename OutTensorType, typename InTensorType>
  static FftFFTWParams_t GetFFTParams(OutTensorType &o,
                          const InTensorType &i, int fft_rank,
                          detail::FFTDirection dir)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    FftFFTWParams_t params;
    constexpr auto RANK = OutTensorType::Rank();   
    using T1    = typename OutTensorType::value_type;
    using T2    = typename InTensorType::value_type;    

    params.irank = i.Rank();
    params.orank = o.Rank();

    params.transform_type = DeduceFFTTransformType<OutTensorType, InTensorType>();
    params.fft_rank =  fft_rank;
    params.dir = dir;

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
      params.inembed[0] = static_cast<int>(i.Size(RANK-2));
      params.onembed[0] = static_cast<int>(o.Size(RANK-2));
      params.inembed[1] = static_cast<int>(i.Size(RANK-1));
      params.onembed[1] = static_cast<int>(o.Size(RANK-1));
      params.istride = static_cast<int>(i.Stride(RANK-1));
      params.ostride = static_cast<int>(o.Stride(RANK-1));
      params.idist = (RANK<=2) ? 1 : (int) static_cast<int>(i.Stride(RANK-3));
      params.odist = (RANK<=2) ? 1 : (int) static_cast<int>(o.Stride(RANK-3));
    }

    params.is_fp32 = is_fp32_inner_type_v<typename OutTensorType::value_type>;
    if constexpr (std::is_same_v<typename OutTensorType::value_type, 
                                typename InTensorType::value_type>) {
      params.in_place = o.Data() == i.Data();
    } else {
      params.in_place = false;
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

/**
 * Crude hash on FFT to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common FFT parameters change
 */
struct FftFFTWParamsKeyHash {
  std::size_t operator()(const FftFFTWParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n[0])) + (std::hash<uint64_t>()(k.n[1])) +
           (std::hash<uint64_t>()(k.fft_rank)) +
           (std::hash<uint64_t>()(k.batch)) + (std::hash<uint64_t>()(k.istride)) + 
           (std::hash<uint64_t>()(static_cast<uint64_t>(k.dir))) +
           (std::hash<uint64_t>()(static_cast<uint64_t>(k.is_fp32)));
  }
};

/**
 * Test FFT parameters for equality. Unlike the hash, all parameters must match.
 */
struct FftFFTWParamsKeyEq {
  bool operator()(const FftFFTWParams_t &l, const FftFFTWParams_t &t) const noexcept
  {
    return l.n[0] == t.n[0] && l.n[1] == t.n[1] && l.batch == t.batch &&
           l.dir == t.dir && l.fft_rank == t.fft_rank &&
           l.is_fp32 == t.is_fp32 && l.in_place == t.in_place &&
           l.inembed[0] == t.inembed[0] && l.inembed[1] == t.inembed[1] &&
           l.onembed[0] == t.onembed[0] && l.onembed[1] == t.onembed[1] &&
           l.istride == t.istride && l.ostride == t.ostride &&
           l.idist == t.idist && l.odist == t.odist &&
           l.transform_type == t.transform_type &&
           l.irank == t.irank && l.orank == t.orank ;
  }
};

using fft_fftw_cache_t = std::unordered_map<FftFFTWParams_t, std::any, FftFFTWParamsKeyHash, FftFFTWParamsKeyEq>;

#if MATX_EN_CPU_FFT
/**
 * Class to manage FFTW initialization and cleanup for single and
 * double precision transforms.
 * 
 */
class FFTWPlanManager {
public:
  static void inline InitFFTWF() {
    if (!init_fp32_) {
      int ret = fftwf_init_threads();
      MATX_ASSERT_STR(ret != 0, matxAssertError, "fftwf_init_threads() failed");
      init_fp32_ = true;
    }
  }

  static void inline InitFFTW() {
    if (!init_fp64_) {
      int ret = fftw_init_threads();
      MATX_ASSERT_STR(ret != 0, matxAssertError, "fftw_init_threads() failed");
      init_fp64_ = true;
    }
  }

  static void IncrementPlanCount() { active_plans_++; }

  static void DecrementPlanCount() {
    active_plans_--;
    if (active_plans_ == 0) {
      if (init_fp32_) {
          fftwf_cleanup_threads();
          fftwf_cleanup();
          init_fp32_ = false;
      }
      if (init_fp64_) {
          fftw_cleanup_threads();
          fftw_cleanup();
          init_fp64_ = false;
      }
    }
  }

private:
  static inline cuda::std::atomic<int> active_plans_ = 0;
  static inline cuda::std::atomic<bool> init_fp32_ = false;
  static inline cuda::std::atomic<bool> init_fp64_ = false;
};

/**
 * Class for FFTW plans
 * 
 * Once created, the FFT plan can be reused as long as the input and output
 * tensors' dimensions, strides, etc. all match. If the plan was originally
 * created to be in-place or out-of-place, the new tensors must match that to
 * reuse the plan.
 * 
 * Once the last plan is deallocated, all other persistent data is freed as well.
 * 
 */
template<typename OutTensorType, typename InTensorType> class matxFFTWPlan_t {
public:
  using out_value_type = typename OutTensorType::value_type;
  using plan_type = std::conditional_t<is_fp32_inner_type_v<out_value_type>, fftwf_plan, fftw_plan>;

  template <ThreadsMode MODE>
  matxFFTWPlan_t(OutTensorType &o, 
                const InTensorType &i, 
                const FftFFTWParams_t &params, 
                const HostExecutor<MODE> &exec) : params_(params) {
    auto fft_dir = (params_.dir == detail::FFTDirection::FORWARD) ? FFTW_FORWARD : FFTW_BACKWARD;
    auto in_ptr = i.Data();
    auto out_ptr = o.Data();

    if constexpr (is_fp32_) {
      FFTWPlanManager::InitFFTWF();
      fftwf_plan_with_nthreads(exec.GetNumThreads());
      if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::C2C) {
        plan_  = fftwf_plan_many_dft( params_.fft_rank, 
                                    params_.n, 
                                    params_.batch, 
                                    reinterpret_cast<fftwf_complex*>(in_ptr), 
                                    params_.inembed, 
                                    params_.istride, 
                                    params_.idist,
                                    reinterpret_cast<fftwf_complex*>(out_ptr), 
                                    params_.onembed, 
                                    params_.ostride, 
                                    params_.odist,  
                                    fft_dir, 
                                    FFTW_ESTIMATE);
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::C2R) {
        plan_  = fftwf_plan_many_dft_c2r( params_.fft_rank, 
                                    params_.n, 
                                    params_.batch, 
                                    reinterpret_cast<fftwf_complex*>(in_ptr), 
                                    params_.inembed, 
                                    params_.istride, 
                                    params_.idist,
                                    out_ptr, 
                                    params_.onembed, 
                                    params_.ostride, 
                                    params_.odist,  
                                    FFTW_ESTIMATE);
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::R2C) {        
        plan_  = fftwf_plan_many_dft_r2c( params_.fft_rank, 
                                    params_.n, 
                                    params_.batch, 
                                    in_ptr, 
                                    params_.inembed, 
                                    params_.istride, 
                                    params_.idist,
                                    reinterpret_cast<fftwf_complex*>(out_ptr), 
                                    params_.onembed, 
                                    params_.ostride, 
                                    params_.odist,  
                                    FFTW_ESTIMATE);
      }
    }
    else {
      FFTWPlanManager::InitFFTW();
      fftw_plan_with_nthreads(exec.GetNumThreads());
      if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::Z2Z) {
        plan_  = fftw_plan_many_dft( params_.fft_rank, 
                                    params_.n, 
                                    params_.batch, 
                                    reinterpret_cast<fftw_complex*>(in_ptr), 
                                    params_.inembed, 
                                    params_.istride, 
                                    params_.idist,
                                    reinterpret_cast<fftw_complex*>(out_ptr), 
                                    params_.onembed, 
                                    params_.ostride, 
                                    params_.odist,  
                                    fft_dir, 
                                    FFTW_ESTIMATE);
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::Z2D) {
        plan_  = fftw_plan_many_dft_c2r( params_.fft_rank, 
                                    params_.n, 
                                    params_.batch, 
                                    reinterpret_cast<fftw_complex*>(in_ptr), 
                                    params_.inembed, 
                                    params_.istride, 
                                    params_.idist,
                                    out_ptr, 
                                    params_.onembed, 
                                    params_.ostride, 
                                    params_.odist,  
                                    FFTW_ESTIMATE);
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::D2Z) {
        plan_  = fftw_plan_many_dft_r2c( params_.fft_rank, 
                                    params_.n, 
                                    params_.batch, 
                                    in_ptr, 
                                    params_.inembed, 
                                    params_.istride, 
                                    params_.idist,
                                    reinterpret_cast<fftw_complex*>(out_ptr), 
                                    params_.onembed, 
                                    params_.ostride, 
                                    params_.odist,  
                                    FFTW_ESTIMATE);
      } 
    }
    MATX_ASSERT_STR(plan_ != nullptr, matxAssertError, "fftw plan creation failed");

    FFTWPlanManager::IncrementPlanCount();
  }

  /**
   * @brief Destructor for matxFFTWPlan_t
   *
   * Deallocates the FFT plan and decrements the plan count
   */
  ~matxFFTWPlan_t() {
    if constexpr (is_fp32_) {
      fftwf_destroy_plan(plan_);
    } else {
      fftw_destroy_plan(plan_);
    }
    FFTWPlanManager::DecrementPlanCount();
  }

  /**
   * @brief Execute the cached FFTW plan
   * 
   * @param o 
   * @param i 
   */
  void inline Exec(OutTensorType &o, const InTensorType &i) {
    auto out = o.Data();
    auto in = i.Data();

    if constexpr (is_fp32_) {
      if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::C2C) {
        fftwf_execute_dft(plan_, 
                          reinterpret_cast<fftwf_complex*>(in), 
                          reinterpret_cast<fftwf_complex*>(out));
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::C2R) {
        fftwf_execute_dft_c2r(plan_, 
                          reinterpret_cast<fftwf_complex*>(in), 
                          out);
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::R2C) {     
        fftwf_execute_dft_r2c(plan_, 
                          in, 
                          reinterpret_cast<fftwf_complex*>(out));
      }
    } 
    else {
      if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::Z2Z) {
        fftw_execute_dft(plan_, 
                          reinterpret_cast<fftw_complex*>(in), 
                          reinterpret_cast<fftw_complex*>(out));
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::Z2D) {
        fftw_execute_dft_c2r(plan_, 
                          reinterpret_cast<fftw_complex*>(in), 
                          out);
      }
      else if constexpr (DeduceFFTTransformType<OutTensorType, InTensorType>() == FFTType::D2Z) {     
        fftw_execute_dft_r2c(plan_, 
                          in, 
                          reinterpret_cast<fftw_complex*>(out));
      }
    }
  }

private:
  static constexpr bool is_fp32_ = is_fp32_inner_type_v<out_value_type>;

  FftFFTWParams_t params_;
  plan_type plan_;
};
#endif

  template <typename Op>
  __MATX_INLINE__ auto getFFTW1DSupportedTensor(const Op &in) {
    // This would be better as a templated lambda, but we don't have those in C++17 yet
    const auto support_func = [&in]() {
      if constexpr (is_tensor_view_v<Op>) {
        if constexpr (Op::Rank() >= 2) {
          if (in.Stride(Op::Rank() - 2) != in.Stride(Op::Rank() - 1) * in.Size(Op::Rank() - 1)) {
            return false;
          }
        }
        if constexpr (Op::Rank() > 2) {
          if (in.Stride(Op::Rank() - 3) != in.Size(Op::Rank() - 2) * in.Stride(Op::Rank() - 2)) {
            return false;
          }
        }

        return true;
      }
      else {
        return true;
      }
    };
    
    return GetSupportedTensor(in, support_func, MATX_HOST_MALLOC_MEMORY);
  }


  template <typename Op>
  __MATX_INLINE__ auto getFFTW2DSupportedTensor( const Op &in) {
    // This would be better as a templated lambda, but we don't have those in C++17 yet
    const auto support_func = [&in]() {
      if constexpr (is_tensor_view_v<Op>) {
        if ( in.Stride(Op::Rank()-2) != in.Stride(Op::Rank()-1) * in.Size(Op::Rank()-1)) {
          return false;
        } else if constexpr ( Op::Rank() > 2) {
          if(in.Stride(Op::Rank()-3) != in.Size(Op::Rank()-2) * in.Stride(Op::Rank()-2)) {
            return false;
          }
        }

        return true;
      }
      else {
        return true;
      }
    };
    
    return GetSupportedTensor(in, support_func, MATX_HOST_MALLOC_MEMORY);
  }  


  template <typename OutputTensor, typename InputTensor, ThreadsMode MODE>
  __MATX_INLINE__ void fft_exec([[maybe_unused]] OutputTensor &o, 
                                [[maybe_unused]] const InputTensor &i, 
                                [[maybe_unused]] const FftFFTWParams_t &params, 
                                [[maybe_unused]] detail::FFTDirection dir,
                                [[maybe_unused]] const HostExecutor<MODE> &exec) {
#if MATX_EN_CPU_FFT                                  
    using cache_val_type = detail::matxFFTWPlan_t<OutputTensor, InputTensor>;
    detail::GetCache().LookupAndExec<detail::fft_fftw_cache_t>(
      detail::GetCacheIdFromType<detail::fft_fftw_cache_t>(),
      params,
      [&]() {
        return std::make_shared<cache_val_type>(o, i, params, exec);
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->Exec(o, i);
      }
    );
#endif
  }

  template <typename OutputTensor, typename InputTensor, ThreadsMode MODE>
  __MATX_INLINE__ void fft1d_dispatch(OutputTensor o, const InputTensor i,
          uint64_t fft_size, detail::FFTDirection dir, FFTNorm norm, const HostExecutor<MODE> &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(TotalSize(i) < std::numeric_limits<int>::max(), matxInvalidSize, "Dimensions too large for host FFT currently");

    // converts operators to tensors
    auto out = getFFTW1DSupportedTensor(o);
    auto in_t = getFFTW1DSupportedTensor(i); 

    if(!in_t.isSameView(i)) {
      (in_t = i).run(exec);
    }

    auto in = detail::GetFFTInputView(out, in_t, fft_size, exec);

    // Get parameters required by these tensors
    auto params = GetFFTParams(out, in, 1, dir);

    fft_exec(out, in, params, dir, exec);

    if(!out.isSameView(o)) {
      (o = out).run(exec);
    }

    using s_type = typename detail::value_promote_t<typename inner_op_type_t<typename InputTensor::value_type>::type>;
    s_type factor;
    constexpr s_type s_one = static_cast<s_type>(1.0);

    if (dir == detail::FFTDirection::FORWARD) {
      factor = static_cast<s_type>(params.n[0]);

      if (norm == FFTNorm::ORTHO) {
        (o *= static_cast<s_type>(1.0 / std::sqrt(factor))).run(exec);
      } else if (norm == FFTNorm::FORWARD) {
        (o *= static_cast<s_type>(1.0 / factor)).run(exec);
      }
    }
    else {
      factor = static_cast<s_type>(params.n[0]);

      if (norm == FFTNorm::ORTHO) {
        (o *= static_cast<s_type>(static_cast<s_type>(1) / std::sqrt(factor))).run(exec);
      } else if (norm == FFTNorm::BACKWARD) {
        (o *= static_cast<s_type>(static_cast<s_type>(1) / factor)).run(exec);
      }    
    }  
  }


  template <typename OutputTensor, typename InputTensor, ThreadsMode MODE>
  __MATX_INLINE__ void fft2d_dispatch(OutputTensor o, const InputTensor i,
          detail::FFTDirection dir, FFTNorm norm, const HostExecutor<MODE> &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_ASSERT_STR(TotalSize(i) < std::numeric_limits<int>::max(), matxInvalidSize, "Dimensions too large for host FFT currently");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // converts operators to tensors
    auto out = getFFTW2DSupportedTensor(o);
    auto in = getFFTW2DSupportedTensor(i); 
    
    if(!in.isSameView(i)) {
      (in = i).run(exec);
    }

    // Get parameters required by these tensors
    auto params = GetFFTParams(out, in, 2, dir);

    fft_exec(out, in, params, dir, exec);

    if(!out.isSameView(o)) {
      (o = out).run(exec);
    }

    using s_type = typename detail::value_promote_t<typename inner_op_type_t<typename InputTensor::value_type>::type>;
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


  template <typename OutputTensor, typename InputTensor, ThreadsMode MODE>
  __MATX_INLINE__ void fft_impl(OutputTensor o, const InputTensor i,
          uint64_t fft_size, FFTNorm norm, const HostExecutor<MODE> &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_STATIC_ASSERT_STR( (is_fp32_inner_type_v<typename OutputTensor::value_type> ||
                            is_fp64_inner_type_v<typename InputTensor::value_type>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");    
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft1d_dispatch(o, i, fft_size, FFTDirection::FORWARD, norm, exec);
  }


  template <typename OutputTensor, typename InputTensor, ThreadsMode MODE>
  __MATX_INLINE__ void ifft_impl(OutputTensor o, const InputTensor i,
          uint64_t fft_size, FFTNorm norm, const HostExecutor<MODE> &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_STATIC_ASSERT_STR( (is_fp32_inner_type_v<typename OutputTensor::value_type> ||
                            is_fp64_inner_type_v<typename InputTensor::value_type>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");      
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft1d_dispatch(o, i, fft_size, FFTDirection::BACKWARD, norm, exec);
  }



  template <typename OutputTensor, typename InputTensor, ThreadsMode MODE>
  __MATX_INLINE__ void fft2_impl(OutputTensor o, const InputTensor i, FFTNorm norm, 
            const HostExecutor<MODE> &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");
    MATX_STATIC_ASSERT_STR(InputTensor::Rank() >= 2, matxInvalidSize, "2D FFT must be rank 2 tensor or higher");      
    MATX_STATIC_ASSERT_STR( (is_fp32_inner_type_v<typename OutputTensor::value_type> ||
                            is_fp64_inner_type_v<typename InputTensor::value_type>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");     
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft2d_dispatch(o, i, FFTDirection::FORWARD, norm, exec);
  }

  template <typename OutputTensor, typename InputTensor, ThreadsMode MODE>
  __MATX_INLINE__ void ifft2_impl(OutputTensor o, const InputTensor i, FFTNorm norm, 
            const HostExecutor<MODE> &exec)
  {
    MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
      "Input and output tensor ranks must match");  
    MATX_STATIC_ASSERT_STR(InputTensor::Rank() >= 2, matxInvalidSize, "2D FFT must be rank 2 tensor or higher");      
    MATX_STATIC_ASSERT_STR( (is_fp32_inner_type_v<typename OutputTensor::value_type> ||
                            is_fp64_inner_type_v<typename InputTensor::value_type>), matxInvalidType,
                            "Host FFTs only support single or double precision floats");      
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    MATX_ASSERT_STR(MATX_EN_CPU_FFT, matxInvalidExecutor, "Trying to run a host FFT executor but host FFT support is not configured");

    fft2d_dispatch(o, i, FFTDirection::BACKWARD, norm, exec);  
  }
    
} // end namespace detail



}; // end namespace matx