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

#include <cufft.h>
#include <cufftXt.h>

#include "matx/core/cache.h"
#include "matx/core/error.h"
#include "matx/core/make_tensor.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/transforms/copy.h"

#include <cstdio>
#include <functional>
#include <optional>

namespace matx {

enum class FFTNorm {
  BACKWARD, /// fft is unscaled, ifft is 1/N
  FORWARD, /// fft is scaled 1/N, ifft is not scaled
  ORTHO /// fft is scaled 1/sqrt(N), ifft is scaled 1/sqrt(N)
};

namespace detail {

static constexpr int MAX_FFT_RANK = 2;


/**
 * Parameters needed to execute an FFT/IFFT in cuFFT
 */
struct FftParams_t {
  long long irank, orank;
  long long n[MAX_FFT_RANK] = {0};
  long long batch;
  int       batch_dims;
  long long inembed[MAX_FFT_RANK] = {0};
  long long onembed[MAX_FFT_RANK] = {0};
  long long istride, ostride;
  long long idist, odist;
  cufftType transform_type; // Known from input/output type, but still useful
  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType exec_type;
  int fft_rank;
  cudaStream_t stream = 0;
};

/* Base class for FFTs. This should not be used directly. */
template <typename OutTensorType, typename InTensorType> class matxFFTPlan_t {
public:
  using index_type = typename OutTensorType::shape_type;
  using T1    = typename OutTensorType::scalar_type;
  using T2    = typename InTensorType::scalar_type;
  static constexpr auto RANK = OutTensorType::Rank();    
  static_assert(OutTensorType::Rank() == InTensorType::Rank(), "Input and output FFT ranks must match");

  /**
   * Execute an FFT in a stream
   *
   * Runs the FFT on the device with the active plan. The input and output views
   * don't have to be the same as were used for plan creation, but the rank and
   * dimensions must match.
   *
   * @param o
   *   Output view
   * @param i
   *   Input view
   * @param stream
   *   CUDA stream
   **/
  void inline Forward(OutTensorType &o,
                      const InTensorType &i, cudaStream_t stream, FFTNorm norm = FFTNorm::BACKWARD)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    cufftSetStream(this->plan_, stream);

    // Normalize input if necessary
    using s_type = typename detail::value_promote_t<typename InTensorType::scalar_type>;
    s_type factor;
    if (params_.fft_rank == 1) {
      factor = static_cast<s_type>(params_.n[0]);
    } else {
      factor = static_cast<s_type>(params_.n[0] * params_.n[1]);
    }

    Exec(o, i, CUFFT_FORWARD);

    if (norm == FFTNorm::ORTHO) {
      (o *= 1.0 / std::sqrt(factor)).run(stream);
    } else if (norm == FFTNorm::FORWARD) {
      (o *= 1.0 / factor).run(stream);
    }

  }

  /**
   * Execute an IFFT in a stream
   *
   * Runs the inverse FFT on the device with the active plan. The input and
   *output views don't have to be the same as were used for plan creation, but
   *the rank and dimensions must match.
   *
   * @param o
   *   Output view
   * @param i
   *   Input view
   * @param stream
   *   CUDA stream
   **/
  void inline Inverse(OutTensorType &o,
                      const InTensorType &i, cudaStream_t stream, FFTNorm norm = FFTNorm::BACKWARD)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    cufftSetStream(this->plan_, stream);
    Exec(o, i, CUFFT_INVERSE);

    // cuFFT doesn't scale IFFT the same as MATLAB/Python. Scale it here to
    // match
    using s_type = typename detail::value_promote_t<typename OutTensorType::scalar_type>;
    s_type factor;
    if (params_.fft_rank == 1) {
      factor = static_cast<s_type>(params_.n[0]);
    } else {
      factor = static_cast<s_type>(params_.n[0] * params_.n[1]);
    }

    if (norm == FFTNorm::ORTHO) {
      (o *= 1.0 / std::sqrt(factor)).run(stream);
    } else if (norm == FFTNorm::BACKWARD) {
      (o *= 1.0 / factor).run(stream);
    }

  }

  static FftParams_t GetFFTParams(OutTensorType &o,
                          const InTensorType &i, int fft_rank)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    FftParams_t params;

    // Default to default stream, but caller will generally overwrite this
    params.stream = 0;

    params.irank = i.Rank();
    params.orank = o.Rank();

    params.transform_type = DeduceFFTTransformType();
    params.input_type = matxFFTPlan_t<OutTensorType, InTensorType>::GetInputType();
    params.output_type = matxFFTPlan_t<OutTensorType, InTensorType>::GetOutputType();
    params.exec_type = matxFFTPlan_t<OutTensorType, InTensorType>::GetExecType();
    params.fft_rank =  fft_rank;

    if (fft_rank == 1) {
      params.batch_dims = 0;
      params.n[0] = (params.transform_type == CUFFT_C2R ||
                      params.transform_type == CUFFT_Z2D)
                        ? o.Size(RANK - 1)
                        : i.Size(RANK - 1);

      if (i.IsContiguous() && o.IsContiguous()) {
        size_t freeMem, totalMem;
        auto err = cudaMemGetInfo(&freeMem, &totalMem);
        // Use up to 30% of free memory to batch, assuming memory use matches batch size
        double max_for_fft_workspace = static_cast<double>(freeMem) * 0.3;

        params.batch = 1;
        for (int dim = i.Rank() - 2; dim >= 0; dim--) {
          if (static_cast<double>(params.batch * i.Size(dim) * sizeof(typename InTensorType::scalar_type)) > max_for_fft_workspace) {
            break;
          }

          params.batch_dims++;
          params.batch *= i.Size(dim);
        }
      }
      else {
        if (RANK == 1) {
          params.batch = 1;
          params.batch_dims = 0;
        }
        else {
          params.batch = i.Size(RANK-2);
          params.batch_dims = 1; 
        }
      }
      
      params.inembed[0] = i.Size(RANK - 1); // Unused
      params.onembed[0] = o.Size(RANK - 1); // Unused
      params.istride = i.Stride(RANK - 1);
      params.ostride = o.Stride(RANK - 1);
      params.idist = (RANK == 1) ? 1 : i.Stride(RANK - 2);  
      params.odist = (RANK == 1) ? 1 : o.Stride(RANK - 2);

      if constexpr (is_complex_half_v<T1> || is_complex_half_v<T1>) {
        if ((params.n[0] & (params.n[0] - 1)) != 0) {
          MATX_THROW(matxInvalidDim,
                     "Half precision only supports power of two transforms");
        }
      }
    }
    else if (fft_rank == 2) {
      if (params.transform_type == CUFFT_C2R ||
          params.transform_type == CUFFT_Z2D) {
        params.n[0] = o.Size(RANK-1);
        params.n[1] = o.Size(RANK-2);
      }
      else {
        params.n[1] = i.Size(RANK-1);
        params.n[0] = i.Size(RANK-2);
      }

      params.batch = (RANK == 2) ? 1 : i.Size(RANK - 3);
      params.inembed[1] = o.Size(RANK-1);
      params.onembed[1] = i.Size(RANK-1);
      params.istride = i.Stride(RANK-1);
      params.ostride = o.Stride(RANK-1);
      params.idist = (RANK<=2) ? 1 : (int) i.Stride(RANK-3);
      params.odist = (RANK<=2) ? 1 : (int) o.Stride(RANK-3);

      if constexpr (is_complex_half_v<T1> || is_complex_half_v<T1>) {
        if ((params.n[0] & (params.n[0] - 1)) != 0 ||
            (params.n[1] & (params.n[1] - 1)) != 0) {
          MATX_THROW(matxInvalidDim,
                     "Half precision only supports power of two transforms");
        }
      }
    }

    MATX_ASSERT_STR(params.idist > 0, matxInvalidDim, "FFTs do not support batch strides of 0 (no cloned views)");

    return params;
  }

protected:
  matxFFTPlan_t(){};

  virtual void Exec(OutTensorType &o, const InTensorType &i, int dir) = 0;

  inline void InternalExec(const void *idata, void *odata, int dir)
  {
    [[maybe_unused]] cufftResult res;

    res = cufftXtExec(this->plan_, (void *)idata, (void *)odata, dir);
    MATX_ASSERT(res == CUFFT_SUCCESS, matxCufftError);
  }

  static inline constexpr cudaDataType GetInputType()
  {
    return GetIOType<T2>();
  }

  static inline constexpr cudaDataType GetOutputType()
  {
    return GetIOType<T1>();
  }

  static inline constexpr cudaDataType GetExecType()
  {
    constexpr auto it = GetInputType();
    constexpr auto ot = GetOutputType();

    if constexpr (it == CUDA_C_16F || ot == CUDA_C_16F) {
      return CUDA_C_16F;
    }
    else if constexpr (it == CUDA_C_16BF || ot == CUDA_C_16BF) {
      return CUDA_C_16BF;
    }
    else if constexpr (it == CUDA_C_32F || ot == CUDA_C_32F) {
      return CUDA_C_32F;
    }

    return CUDA_C_64F;
  }

  template <typename T> static inline constexpr cudaDataType GetIOType()
  {
    if constexpr (std::is_same_v<T, matxFp16Complex>) {
      return CUDA_C_16F;
    }
    else if constexpr (std::is_same_v<T, matxBf16Complex>) {
      return CUDA_C_16BF;
    }
    if constexpr (std::is_same_v<T, matxFp16>) {
      return CUDA_R_16F;
    }
    else if constexpr (std::is_same_v<T, matxBf16>) {
      return CUDA_R_16BF;
    }
    if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
      return CUDA_C_32F;
    }
    else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
      return CUDA_C_64F;
    }
    if constexpr (std::is_same_v<T, float>) {
      return CUDA_R_32F;
    }
    else if constexpr (std::is_same_v<T, double>) {
      return CUDA_R_64F;
    }

    return CUDA_C_32F;
  }

  static cufftType DeduceFFTTransformType()
  {
    // Deduce plan type from view types
    if constexpr (std::is_same_v<T1, cuda::std::complex<float>>) {
      if constexpr (std::is_same_v<T2, cuda::std::complex<float>>) {
        return CUFFT_C2C;
      }
      else if constexpr (std::is_same_v<T2, float>) {
        return CUFFT_R2C;
      }
    }
    else if constexpr (std::is_same_v<T1, float> &&
                       std::is_same_v<T2, cuda::std::complex<float>>) {
      return CUFFT_C2R;
    }
    else if constexpr (std::is_same_v<T1, cuda::std::complex<double>>) {
      if constexpr (std::is_same_v<T2, cuda::std::complex<double>>) {
        return CUFFT_Z2Z;
      }
      else if constexpr (std::is_same_v<T2, double>) {
        return CUFFT_D2Z;
      }
    }
    else if constexpr (std::is_same_v<T1, double> &&
                       std::is_same_v<T2, cuda::std::complex<double>>) {
      return CUFFT_Z2D;
    }
    else if constexpr (is_complex_half_v<T1>) {
      if constexpr (is_complex_half_v<T2>) {
        return CUFFT_C2C;
      }
      else if constexpr (is_half_v<T1>) {
        return CUFFT_R2C;
      }
    }
    else if constexpr (is_half_v<T1> && is_complex_half_v<T2>) {
      return CUFFT_C2R;
    }
    MATX_THROW(matxNotSupported,
               "Could not deduce FFT types from input and output view types!");
  }

  virtual ~matxFFTPlan_t() {
    if (this->workspace_ != nullptr) {

      matxFree(workspace_);
      this->workspace_ = nullptr;
    }

    cufftDestroy(this->plan_); 
  }

  cufftHandle plan_;
  FftParams_t params_;
  void *workspace_;
  int fftrank_ = 0;
};

/**
 * Create a 1D FFT plan
 *
 * An FFT plan is used to set up all parameters and memory needed to execute an
 * FFT. All parameters of the FFT normally needed when using cuFFT directly are
 * deduced by matx using the View classes passed in. Because MatX uses cuFFT
 * directly, all limitations and properties of cuFFT must be adhered to. Please
 * see the cuFFT documentation to see these limitations. Once the plan has been
 * created, FFTs can be executed as many times as needed using the Exec()
 * functions. It is not necessary to pass in the same views as were used to
 * create the plans as long as the rank and dimensions are idential.
 *
 * If a tensor larger than rank 1 is passed, all other dimensions are batch
 * dimensions
 *
 * @tparam T1
 *   Output view data type
 * @tparam T2
 *   Input view data type
 */
template <typename OutTensorType, typename InTensorType = OutTensorType>
class matxFFTPlan1D_t : public matxFFTPlan_t<OutTensorType, InTensorType> {
public:
  using T1    = typename OutTensorType::scalar_type;
  using T2    = typename InTensorType::scalar_type;
  static constexpr int RANK = OutTensorType::Rank();
  
/**
 * Construct a 1D FFT plan
 *
 * @param o
 *   Output view
 * @param i
 *   Input view
 * @param stream
 *   CUDA stream in which device memory allocations may be made
 *
 * */
matxFFTPlan1D_t(OutTensorType &o, const InTensorType &i, cudaStream_t stream = 0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  int dev;
  cudaGetDevice(&dev);

  this->workspace_ = nullptr;
  this->params_ = this->GetFFTParams(o, i, 1);

  if (this->params_.transform_type == CUFFT_C2R ||
      this->params_.transform_type == CUFFT_Z2D) {
    if (is_cuda_complex_v<T1> || !is_cuda_complex_v<T2>) {
      MATX_THROW(matxInvalidType, "FFT types inconsistent with C2R/Z2D transform");
    }
    if (this->params_.n[0] != o.Size(OutTensorType::Rank()-1) ||
       (this->params_.n[0]/2)+1 != i.Size(InTensorType::Rank()-1)) {
      MATX_THROW(matxInvalidSize, "Tensor sizes inconsistent with C2R/Z2D transform");
    }
  }
  else if (this->params_.transform_type == CUFFT_R2C ||
           this->params_.transform_type == CUFFT_D2Z) {
    if (is_cuda_complex_v<T2> || !is_cuda_complex_v<T1>) {
      MATX_THROW(matxInvalidType, "FFT types inconsistent with R2C/D2Z transform");
    }
    if (this->params_.n[0] != i.Size(InTensorType::Rank()-1) ||
       (this->params_.n[0]/2)+1 != o.Size(OutTensorType::Rank()-1)) {
      MATX_THROW(matxInvalidSize, "Tensor sizes inconsistent with R2C/D2Z transform");
    }
  }
  else {
    if (!is_complex_v<T2> || !is_complex_v<T1> || !std::is_same_v<T1, T2>) {
      MATX_THROW(matxInvalidType, "FFT types inconsistent with C2C transform");
    }
    if (this->params_.n[0] != o.Size(InTensorType::Rank()-1) ||
        this->params_.n[0] != i.Size(InTensorType::Rank()-1)) {
      MATX_THROW(matxInvalidSize, "Tensor sizes inconsistent with C2C transform");
    }
  }

  size_t workspaceSize;
  cufftCreate(&this->plan_);
  [[maybe_unused]] cufftResult error;

  error = cufftXtGetSizeMany(this->plan_, 1, this->params_.n, this->params_.inembed,
                      this->params_.istride, this->params_.idist,
                      this->params_.input_type, this->params_.onembed,
                      this->params_.ostride, this->params_.odist,
                      this->params_.output_type, this->params_.batch,
                      &workspaceSize, this->params_.exec_type);
  MATX_ASSERT(error == CUFFT_SUCCESS, matxCufftError);

  matxAlloc((void **)&this->workspace_, workspaceSize, MATX_ASYNC_DEVICE_MEMORY, stream);

  cufftSetWorkArea(this->plan_, this->workspace_);

  error = cufftXtMakePlanMany(
      this->plan_, 1, this->params_.n, this->params_.inembed,
      this->params_.istride, this->params_.idist, this->params_.input_type,
      this->params_.onembed, this->params_.ostride, this->params_.odist,
      this->params_.output_type, this->params_.batch, &workspaceSize,
      this->params_.exec_type);

  MATX_ASSERT(error == CUFFT_SUCCESS, matxCufftError);
}

private:
virtual void inline Exec(OutTensorType &o, const InTensorType &i,
                         int dir) override
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  if (OutTensorType::Rank() == this->params_.batch_dims + 1) {
    this->InternalExec(static_cast<const void *>(i.Data()),
                      static_cast<void *>(o.Data()), dir);
  }
  else {
    using shape_type = typename InTensorType::desc_type::shape_type;
    std::array<shape_type, InTensorType::Rank()> idx{0};
    auto i_shape = i.Shape();
    // Get total number of batches
    size_t total_iter = std::accumulate(i_shape.begin(), i_shape.begin() + InTensorType::Rank() - (this->params_.batch_dims + 1), 1, std::multiplies<shape_type>());
    for (size_t iter = 0; iter < total_iter; iter++) {
      auto ip = std::apply([&i](auto... param) { return i.GetPointer(param...); }, idx);
      auto op = std::apply([&o](auto... param) { return o.GetPointer(param...); }, idx);
      this->InternalExec(static_cast<const void *>(ip), static_cast<void *>(op), dir);

      // Update all but the last 2 indices
      UpdateIndices<InTensorType, shape_type, InTensorType::Rank()>(i, idx, this->params_.batch_dims + 1);
    }
  }
   
}

}; 

/**
 * Create a 2D FFT plan
 *
 * An FFT plan is used to set up all parameters and memory needed to execute an
 * FFT. All parameters of the FFT normally needed when using cuFFT directly are
 * deduced by matx using the View classes passed in. Because MatX uses cuFFT
 * directly, all limitations and properties of cuFFT must be adhered to. Please
 * see the cuFFT documentation to see these limitations. Once the plan has been
 * created, FFTs can be executed as many times as needed using the Exec()
 * functions. It is not necessary to pass in the same views as were used to
 * create the plans as long as the rank and dimensions are idential.
 *
 * If a tensor larger than rank 2 is passed, all other dimensions are batch
 * dimensions
 *
 * @tparam T1
 *   Output view data type
 * @tparam T2
 *   Input view data type
 * @param stream
 *   CUDA stream in which device memory allocations may be made
 */
template <typename OutTensorType, typename InTensorType = OutTensorType>
class matxFFTPlan2D_t : public matxFFTPlan_t<OutTensorType, InTensorType> {
public:
  static constexpr int RANK = OutTensorType::Rank();
  using T1    = typename OutTensorType::scalar_type;
  using T2    = typename InTensorType::scalar_type;

  /**
   * Construct a 2D FFT plan
   *
   * @param o
   *   Output view
   * @param i
   *   Input view
   *
   * */
  matxFFTPlan2D_t(OutTensorType &o, const InTensorType &i, cudaStream_t stream = 0)
  {
    static_assert(RANK >= 2, "2D FFTs require a rank-2 tensor or higher");
    
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    int dev;
    cudaGetDevice(&dev);

    this->workspace_ = nullptr;
    this->params_ = this->GetFFTParams(o, i, 2);

    if (this->params_.transform_type == CUFFT_C2R ||
        this->params_.transform_type == CUFFT_Z2D) {
      MATX_ASSERT((o.Size(RANK-2) * (o.Size(RANK-1) / 2 + 1)) == i.Size(RANK-1) * i.Size(RANK-2),
                  matxInvalidSize);
      MATX_ASSERT(!is_cuda_complex_v<T1> && is_cuda_complex_v<T2>,
                  matxInvalidType);
    }
    else if (this->params_.transform_type == CUFFT_R2C ||
            this->params_.transform_type == CUFFT_D2Z) {
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

    size_t workspaceSize;
    cufftCreate(&this->plan_);
    [[maybe_unused]] cufftResult error;
    cufftXtGetSizeMany(this->plan_, 2, this->params_.n, this->params_.inembed,
                       this->params_.istride, this->params_.idist,
                       this->params_.input_type, this->params_.onembed,
                       this->params_.ostride, this->params_.odist,
                       this->params_.output_type, this->params_.batch,
                       &workspaceSize, this->params_.exec_type);

    matxAlloc((void **)&this->workspace_, workspaceSize, MATX_ASYNC_DEVICE_MEMORY, stream);
    cufftSetWorkArea(this->plan_, this->workspace_);

    error = cufftXtMakePlanMany(
        this->plan_, 2, this->params_.n, this->params_.inembed,
        this->params_.istride, this->params_.idist, this->params_.input_type,
        this->params_.onembed, this->params_.ostride, this->params_.odist,
        this->params_.output_type, this->params_.batch, &workspaceSize,
        this->params_.exec_type);

    MATX_ASSERT(error == CUFFT_SUCCESS, matxCufftError);
  }

private:
  /**
   * Execute an FFT plan
   *
   * Runs the FFT on the device with the active plan. The input and output views
   * don't have to be the same as were used for plan creation, but the rank and
   * dimensions must match.
   *
   * @param o
   *   Output view
   * @param i
   *   Input view
   * @param dir
   *   Direction of FFT
   **/
  virtual void inline Exec(OutTensorType &o, const InTensorType &i,
                           int dir) override
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    
    static_assert(RANK >= 2);

    if constexpr (RANK <= 3) {
      this->InternalExec(static_cast<const void *>(i.Data()),
                         static_cast<void *>(o.Data()), dir);
    }
    else  {
      using shape_type = typename InTensorType::desc_type::shape_type;
      int batch_offset = 3;
      std::array<shape_type, InTensorType::Rank()> idx{0};
      auto i_shape = i.Shape();
      // Get total number of batches
      size_t total_iter = std::accumulate(i_shape.begin(), i_shape.begin() + InTensorType::Rank() - batch_offset, 1, std::multiplies<shape_type>());
      for (size_t iter = 0; iter < total_iter; iter++) {
        auto ip = std::apply([&i](auto... param) { return i.GetPointer(param...); }, idx);
        auto op = std::apply([&o](auto... param) { return o.GetPointer(param...); }, idx);

        this->InternalExec(static_cast<const void *>(ip), static_cast<void *>(op), dir);

        // Update all but the last 2 indices
        UpdateIndices<InTensorType, shape_type, InTensorType::Rank()>(i, idx, batch_offset);
      }    
    }

}

};

/**
 *  Crude hash on FFT to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common FFT parameters change
 */
struct FftParamsKeyHash {
  std::size_t operator()(const FftParams_t &k) const noexcept
  {
    return (std::hash<uint64_t>()(k.n[0])) + (std::hash<uint64_t>()(k.n[1])) +
           (std::hash<uint64_t>()(k.fft_rank)) +
           (std::hash<uint64_t>()(k.exec_type)) +
           (std::hash<uint64_t>()(k.batch)) + (std::hash<uint64_t>()(k.istride)) +
           (std::hash<uint64_t>()((uint64_t)k.stream));
  }
};

/**
 * Test FFT parameters for equality. Unlike the hash, all parameters must match.
 */
struct FftParamsKeyEq {
  bool operator()(const FftParams_t &l, const FftParams_t &t) const noexcept
  {
    return l.n[0] == t.n[0] && l.n[1] == t.n[1] && l.batch == t.batch &&
           l.fft_rank == t.fft_rank && l.stream == t.stream &&
           l.inembed[0] == t.inembed[0] && l.inembed[1] == t.inembed[1] &&
           l.onembed[0] == t.onembed[0] && l.onembed[1] == t.onembed[1] &&
           l.istride == t.istride && l.ostride == t.ostride &&
           l.idist == t.idist && l.odist == t.odist &&
           l.transform_type == t.transform_type &&
           l.input_type == t.input_type && l.output_type == t.output_type &&
           l.exec_type == t.exec_type && l.irank == t.irank && l.orank == t.orank ;
  }
};

using fft_cache_t = std::unordered_map<FftParams_t, std::any, FftParamsKeyHash, FftParamsKeyEq>;


template <typename OutputTensor, typename InputTensor>
auto  GetFFTInputView([[maybe_unused]] OutputTensor &o,
                    const InputTensor &i, uint64_t fft_size,
                    [[maybe_unused]] cudaStream_t stream)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  using index_type = typename OutputTensor::shape_type;
  using T1    = typename OutputTensor::scalar_type;
  using T2    = typename InputTensor::scalar_type;
  constexpr int RANK = OutputTensor::Rank();

  index_type starts[RANK] = {0};
  index_type ends[RANK];
  index_type in_size = i.Lsize();
  index_type nom_fft_size = in_size;
  index_type act_fft_size;

  // Auto-detect FFT size
  if (fft_size == 0) {
    act_fft_size = o.Lsize();

    // If R2C transform, set length of FFT appropriately
    if constexpr ((std::is_same_v<T2, float> &&
                  std::is_same_v<T1, cuda::std::complex<float>>) ||
                  (std::is_same_v<T2, double> &&
                  std::is_same_v<T1, cuda::std::complex<double>>) ||
                  (std::is_same_v<T2, matxBf16> &&
                  std::is_same_v<T1, matxBf16Complex>) ||
                  (std::is_same_v<T2, matxFp16> &&
                  std::is_same_v<T1, matxFp16Complex>)) { // R2C
      nom_fft_size = in_size;
      act_fft_size = (o.Lsize() - 1) * 2;
    }
    else if constexpr ((std::is_same_v<T1, float> &&
                        std::is_same_v<T2, cuda::std::complex<float>>) ||
                      (std::is_same_v<T1, double> &&
                        std::is_same_v<T2, cuda::std::complex<double>>) ||
                      (std::is_same_v<T1, matxBf16> &&
                        std::is_same_v<T2, matxBf16Complex>) ||
                      (std::is_same_v<T1, matxFp16> &&
                        std::is_same_v<T2, matxFp16Complex>)) { // C2R
      nom_fft_size = (in_size - 1) * 2;
      act_fft_size = (o.Lsize() / 2) + 1;
    }
  }
  else {
    // Force FFT size
    act_fft_size = static_cast<index_type>(fft_size);
  }

  // Set up new shape if transform size doesn't match tensor
  if (nom_fft_size != act_fft_size) {
    std::fill_n(ends, RANK, matxEnd);

    // FFT shorter than the size of the input signal. Create a new view of this
    // slice.
    if (act_fft_size < nom_fft_size) {
      ends[RANK - 1] = act_fft_size;
      return i.Slice(starts, ends);
    }
    else { // FFT length is longer than the input. Pad input

      // If the input needs to be padded we have to temporarily allocate a new
      // buffer, zero the output, then copy our input buffer. This is not very
      // efficient, but if cufft adds a zero-padding feature later we can take
      // advantage of that without changing the API.

      // Create a new shape where n is the size of the last dimension
      auto shape = i.Shape();
      *(shape.end() - 1) = act_fft_size;
      auto tot = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<typename decltype(shape)::value_type>());

      // Make a new buffer large enough for our input
      auto i_new = make_tensor<T2>(shape, MATX_ASYNC_DEVICE_MEMORY, stream);
      ends[RANK - 1] = i.Lsize();
      auto i_pad_part_v = i_new.Slice(starts, ends);

      (i_new = static_cast<promote_half_t<T2>>(0)).run(stream);
      // example-begin copy-test-1
      matx::copy(i_pad_part_v, i, stream);
      // example-end copy-test-1
      return i_new;
    }
  }

  return i;
}

template <typename TensorOp>
__MATX_INLINE__ auto getCufft1DSupportedTensor( const TensorOp &in, cudaStream_t stream) {

  constexpr int RANK=TensorOp::Rank();

  if constexpr ( !(is_tensor_view_v<TensorOp>)) {
    return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream); 
  } else {

    bool supported = true;

    // If there are any unsupported layouts for cufft add them here
    if (supported) {
      return in;
    } else {
      return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream); 
    }
  }
}

template <typename TensorOp>
__MATX_INLINE__ auto getCufft2DSupportedTensor( const TensorOp &in, cudaStream_t stream) {

  constexpr int RANK=TensorOp::Rank();

  if constexpr ( !is_tensor_view_v<TensorOp>) {
    return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream); 
  } else {
    bool supported = true;

    // only a subset of strides are supported per cufft indexing scheme.
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
      return make_tensor<typename TensorOp::scalar_type>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream); 
    }
  }
}

} // end namespace detail



/**
 * Run a 1D FFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 1D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam OutputTensor
 *   Output tensor or operator type
 * @tparam InputTensor
 *   Input tensor or operator type
 * @param o
 *   Output tensor or operator. The length of the fastest-changing dimension dictates the
 * size of FFT. If this size is longer than the length of the input tensor, the
 * tensor will potentially be copied and zero-padded to a new block of memory.
 * Future releases may remove this restriction to where there is no copy.
 * 
 * Note: fft_size must be unsigned so that the axis overload does not match both 
 * prototypes with index_t. 
 * @param i
 *   input tensor or operator
 * @param fft_size
 *   Size of FFT. Setting to 0 uses the output size to figure out the FFT size.
 * @param norm
 *   Normalization to apply to IFFT
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputTensor>
__MATX_INLINE__ void fft_impl(OutputTensor o, const InputTensor i,
         uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD, cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");  
  
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  // converts operators to tensors
  auto out = getCufft1DSupportedTensor(o, stream);
  auto in_t = getCufft1DSupportedTensor(i, stream); 
  
  if(!in_t.isSameView(i)) {
    (in_t = i).run(stream);
  }
 
  // TODO should combine this function with above...
  // currently will result in an extra allocation/transfer when using fft_size to grow
  // adjusts size of tensor based on fft_size
  auto in = detail::GetFFTInputView(out, in_t, fft_size, stream);

  // Get parameters required by these tensors
  auto params = detail::matxFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 1);
  params.stream = stream;

  using cache_val_type = detail::matxFFTPlan1D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cache_t>(
    detail::CacheName::FFT_1D,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(out, in, stream);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Forward(out, in, stream, norm);
    }
  );

  if(!out.isSameView(o)) {
    (o = out).run(stream);
  }
}


template <typename OutputTensor, typename InputTensor>
__MATX_INLINE__ void ifft_impl(OutputTensor o, const InputTensor i,
          uint64_t fft_size = 0, FFTNorm norm = FFTNorm::BACKWARD, cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");
  
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  // converts operators to tensors
  auto out = getCufft1DSupportedTensor(o, stream);
  auto in_t = getCufft1DSupportedTensor(i, stream);
  
  if(!in_t.isSameView(i)) {
   (in_t = i).run(stream);
  }
  
  // TODO should combine into function above
  // adjusts size of tensor based on fft_size
  auto in = detail::GetFFTInputView(out, in_t, fft_size, stream);

  // Get parameters required by these tensors
  auto params = detail::matxFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 1);
  params.stream = stream;

  using cache_val_type = detail::matxFFTPlan1D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cache_t>(
    detail::CacheName::FFT_1D,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(out, in, stream);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Inverse(out, in, stream, norm);
    }
  );

  if(!out.isSameView(o)) {
    (o = out).run(stream);
  }
}


/**
 * Run a 2D FFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 2D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam OutputTensor
 *   Output operator or tensor
 * @tparam InputTensor
 *   Input operator or tensor
 * @param o
 *   Output operator or tensor
 * @param i
 *   input operator or tensor
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputTensor>
__MATX_INLINE__ void fft2_impl(OutputTensor o, const InputTensor i,
           cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");
  
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
 
  auto out = detail::getCufft2DSupportedTensor(o, stream);
  auto in = detail::getCufft2DSupportedTensor(i, stream);

  if(!in.isSameView(i)) {
    (in = i).run(stream);
  }

  // Get parameters required by these tensors
  auto params = detail::matxFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 2);
  params.stream = stream;

  using cache_val_type = detail::matxFFTPlan2D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cache_t>(
    detail::CacheName::FFT_2D,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(out, in, stream);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Forward(out, in, stream);
    }
  );  

  if(!out.isSameView(o)) {
    (o = out).run(stream);
  }
}


/**
 * Run a 2D IFFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 2D IFFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam OutputTensor
 *   Output operator or tensor type
 * @tparam InputTensor
 *   Input operator or tensor type
 * @param o
 *   Output tensor
 * @param i
 *   input tensor
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputTensor>
__MATX_INLINE__ void ifft2_impl(OutputTensor o, const InputTensor i,
           cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output operator ranks must match");
  
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  
  auto out = detail::getCufft2DSupportedTensor(o, stream);
  auto in = detail::getCufft2DSupportedTensor(i, stream);

  if(!in.isSameView(i)) {
    (in = i).run(stream);
  }

    // Get parameters required by these tensors
  auto params = detail::matxFFTPlan_t<decltype(in), decltype(out)>::GetFFTParams(out, in, 2);
  params.stream = stream;
  
  // Get cache or new FFT plan if it doesn't exist
  using cache_val_type = detail::matxFFTPlan2D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cache_t>(
    detail::CacheName::FFT_2D,
    params,
    [&]() {
      return std::make_shared<cache_val_type>(out, in, stream);
    },
    [&](std::shared_ptr<cache_val_type> ctype) {
      ctype->Inverse(out, in, stream);
    }
  );
  
  if(!out.isSameView(o)) {
    (o = out).run(stream);
  }  
}


}; // end namespace matx
