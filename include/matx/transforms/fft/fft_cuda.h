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
#include "matx/transforms/fft/fft_common.h"
#include "matx/transforms/copy.h"

#include <cstdio>
#include <functional>
#include <optional>

namespace matx {

namespace detail {

/**
 * Parameters needed to execute an FFT/IFFT in cuFFT
 */
struct FftCUDAParams_t {
  long long irank, orank;
  long long n[MAX_FFT_RANK] = {0};
  long long batch;
  int       batch_dims;
  long long inembed[MAX_FFT_RANK] = {0};
  long long onembed[MAX_FFT_RANK] = {0};
  long long istride, ostride;
  long long idist, odist;
  FFTType transform_type; // Known from input/output type, but still useful
  cudaDataType input_type;
  cudaDataType output_type;
  cudaDataType exec_type;
  int fft_rank;
  cudaStream_t stream = 0;
};

/* Base class for FFTs. This should not be used directly. */
template <typename OutTensorType, typename InTensorType> class matxCUDAFFTPlan_t {
public:
  using index_type = typename OutTensorType::shape_type;
  using T1    = typename OutTensorType::value_type;
  using T2    = typename InTensorType::value_type;
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
    using s_type = typename detail::value_promote_t<typename InTensorType::value_type>;
    s_type factor;
    if (params_.fft_rank == 1) {
      factor = static_cast<s_type>(params_.n[0]);
    } else {
      factor = static_cast<s_type>(params_.n[0] * params_.n[1]);
    }
printf("running fft\n");
    Exec(o, i, CUFFT_FORWARD);
printf("scaling fft\n");
    if (norm == FFTNorm::ORTHO) {
      (o *= static_cast<s_type>(1.0 / std::sqrt(factor))).run(stream);
    } else if (norm == FFTNorm::FORWARD) {
      (o *= static_cast<s_type>(1.0 / factor)).run(stream);
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
    using s_type = typename detail::value_promote_t<typename OutTensorType::value_type>;
    s_type factor;
    if (params_.fft_rank == 1) {
      factor = static_cast<s_type>(params_.n[0]);
    } else {
      factor = static_cast<s_type>(params_.n[0] * params_.n[1]);
    }

    if (norm == FFTNorm::ORTHO) {
      (o *= static_cast<s_type>(static_cast<s_type>(1) / std::sqrt(factor))).run(stream);
    } else if (norm == FFTNorm::BACKWARD) {
      (o *= static_cast<s_type>(static_cast<s_type>(1) / factor)).run(stream);
    }

  }

  static FftCUDAParams_t GetFFTParams(OutTensorType &o,
                          const InTensorType &i, int fft_rank)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    FftCUDAParams_t params;

    // Default to default stream, but caller will generally overwrite this
    params.stream = 0;

    params.irank = i.Rank();
    params.orank = o.Rank();

    params.transform_type = DeduceFFTTransformType<OutTensorType, InTensorType>();

    params.input_type = matxCUDAFFTPlan_t<OutTensorType, InTensorType>::GetInputType();
    params.output_type = matxCUDAFFTPlan_t<OutTensorType, InTensorType>::GetOutputType();
    params.exec_type = matxCUDAFFTPlan_t<OutTensorType, InTensorType>::GetExecType();
    params.fft_rank =  fft_rank;

    if (fft_rank == 1) {
      params.batch_dims = 0;
      params.n[0] = (params.transform_type == FFTType::C2R ||
                      params.transform_type == FFTType::Z2D)
                        ? o.Size(RANK - 1)
                        : i.Size(RANK - 1);

      if (i.IsContiguous() && o.IsContiguous()) {
        size_t freeMem, totalMem;
        [[maybe_unused]] auto err = cudaMemGetInfo(&freeMem, &totalMem);
        MATX_ASSERT_STR(err == cudaSuccess, matxCudaError, "Failed to get memInfo from device");
        // Use up to 30% of free memory to batch, assuming memory use matches batch size
        double max_for_fft_workspace = static_cast<double>(freeMem) * 0.3;

        params.batch = 1;
        for (int dim = i.Rank() - 2; dim >= 0; dim--) {
          if (static_cast<double>(params.batch * i.Size(dim) * sizeof(typename InTensorType::value_type)) > max_for_fft_workspace) {
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
      if (params.transform_type == FFTType::C2R ||
          params.transform_type == FFTType::Z2D) {
        params.n[1] = o.Size(RANK-1);
        params.n[0] = o.Size(RANK-2);
      }
      else {
        params.n[1] = i.Size(RANK-1);
        params.n[0] = i.Size(RANK-2);
      }

      params.batch = (RANK == 2) ? 1 : i.Size(RANK - 3);
      params.inembed[1] = i.Size(RANK-1);
      params.onembed[1] = o.Size(RANK-1);
      params.istride = i.Stride(RANK-1);
      params.ostride = o.Stride(RANK-1);
      params.idist = (RANK<=2) ? 1 : (int) i.Stride(RANK-3);
      params.odist = (RANK<=2) ? 1 : (int) o.Stride(RANK-3);

      if constexpr (is_complex_half_v<T1> || is_half_v<T1>) {
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
  matxCUDAFFTPlan_t(){};

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

  virtual ~matxCUDAFFTPlan_t() {
    if (this->workspace_ != nullptr) {
      // Pass the default stream until we allow user-deletable caches
      matxFree(workspace_, cudaStreamDefault);
      this->workspace_ = nullptr;
    }

    cufftDestroy(this->plan_);
  }

  cufftHandle plan_;
  FftCUDAParams_t params_;
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
class matxCUDAFFTPlan1D_t : public matxCUDAFFTPlan_t<OutTensorType, InTensorType> {
public:
  using T1    = typename OutTensorType::value_type;
  using T2    = typename InTensorType::value_type;
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
matxCUDAFFTPlan1D_t(OutTensorType &o, const InTensorType &i, cudaStream_t stream = 0)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  int dev;
  cudaGetDevice(&dev);

  this->workspace_ = nullptr;
  this->params_ = this->GetFFTParams(o, i, 1);

  if (this->params_.transform_type == FFTType::C2R ||
      this->params_.transform_type == FFTType::Z2D) {
    if (is_cuda_complex_v<T1> || !is_cuda_complex_v<T2>) {
      MATX_THROW(matxInvalidType, "FFT types inconsistent with C2R/Z2D transform");
    }
    if (this->params_.n[0] != o.Size(OutTensorType::Rank()-1) ||
       (this->params_.n[0]/2)+1 != i.Size(InTensorType::Rank()-1)) {
      MATX_THROW(matxInvalidSize, "Tensor sizes inconsistent with C2R/Z2D transform");
    }
  }
  else if (this->params_.transform_type == FFTType::R2C ||
           this->params_.transform_type == FFTType::D2Z) {
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
    cuda::std::array<shape_type, InTensorType::Rank()> idx{0};
    auto i_shape = i.Shape();
    // Get total number of batches
    size_t total_iter = std::accumulate(i_shape.begin(), i_shape.begin() + InTensorType::Rank() - (this->params_.batch_dims + 1), 1, std::multiplies<shape_type>());
    for (size_t iter = 0; iter < total_iter; iter++) {
      auto ip = cuda::std::apply([&i](auto... param) { return i.GetPointer(param...); }, idx);
      auto op = cuda::std::apply([&o](auto... param) { return o.GetPointer(param...); }, idx);
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
class matxCUDAFFTPlan2D_t : public matxCUDAFFTPlan_t<OutTensorType, InTensorType> {
public:
  static constexpr int RANK = OutTensorType::Rank();
  using T1    = typename OutTensorType::value_type;
  using T2    = typename InTensorType::value_type;

  /**
   * Construct a 2D FFT plan
   *
   * @param o
   *   Output view
   * @param i
   *   Input view
   *
   * */
  matxCUDAFFTPlan2D_t(OutTensorType &o, const InTensorType &i, cudaStream_t stream = 0)
  {
    static_assert(RANK >= 2, "2D FFTs require a rank-2 tensor or higher");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    int dev;
    cudaGetDevice(&dev);

    this->workspace_ = nullptr;
    this->params_ = this->GetFFTParams(o, i, 2);

    if (this->params_.transform_type == FFTType::C2R ||
        this->params_.transform_type == FFTType::Z2D) {
      MATX_ASSERT((o.Size(RANK-2) * (o.Size(RANK-1) / 2 + 1)) == i.Size(RANK-1) * i.Size(RANK-2),
                  matxInvalidSize);
      MATX_ASSERT(!is_cuda_complex_v<T1> && is_cuda_complex_v<T2>,
                  matxInvalidType);
    }
    else if (this->params_.transform_type == FFTType::R2C ||
            this->params_.transform_type == FFTType::D2Z) {
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
      cuda::std::array<shape_type, InTensorType::Rank()> idx{0};
      auto i_shape = i.Shape();
      // Get total number of batches
      size_t total_iter = std::accumulate(i_shape.begin(), i_shape.begin() + InTensorType::Rank() - batch_offset, 1, std::multiplies<shape_type>());
      for (size_t iter = 0; iter < total_iter; iter++) {
        auto ip = cuda::std::apply([&i](auto... param) { return i.GetPointer(param...); }, idx);
        auto op = cuda::std::apply([&o](auto... param) { return o.GetPointer(param...); }, idx);

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
struct FftCUDAParamsKeyHash {
  std::size_t operator()(const FftCUDAParams_t &k) const noexcept
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
struct FftCUDAParamsKeyEq {
  bool operator()(const FftCUDAParams_t &l, const FftCUDAParams_t &t) const noexcept
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

using fft_cuda_cache_t = std::unordered_map<FftCUDAParams_t, std::any, FftCUDAParamsKeyHash, FftCUDAParamsKeyEq>;

template <typename Op>
__MATX_INLINE__ auto getCufft1DSupportedTensor( const Op &in, cudaStream_t stream) {
  // This would be better as a templated lambda, but we don't have those in C++17 yet
  const auto support_func = []() {
    return true;
  };
  
  return GetSupportedTensor(in, support_func, MATX_ASYNC_DEVICE_MEMORY, stream);
}

template <typename Op>
__MATX_INLINE__ auto getCufft2DSupportedTensor( const Op &in, cudaStream_t stream) {
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
  
  return GetSupportedTensor(in, support_func, MATX_ASYNC_DEVICE_MEMORY, stream);
}


template <typename OutputTensor, typename InputTensor>
__MATX_INLINE__ void fft_impl(OutputTensor o, const InputTensor i,
         uint64_t fft_size, FFTNorm norm, const cudaExecutor &exec)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");

  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
  const auto stream = exec.getStream();

  // converts operators to tensors
  auto out = getCufft1DSupportedTensor(o, stream);
  auto in_t = getCufft1DSupportedTensor(i, stream);

  if(!in_t.isSameView(i)) {
    (in_t = i).run(stream);
  }


  // TODO should combine this function with above...
  // currently will result in an extra allocation/transfer when using fft_size to grow
  // adjusts size of tensor based on fft_size
  auto in = detail::GetFFTInputView(out, in_t, fft_size, exec);

  // Get parameters required by these tensors
  auto params = detail::matxCUDAFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 1);
  params.stream = stream;

  using cache_val_type = detail::matxCUDAFFTPlan1D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::fft_cuda_cache_t>(),
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
          uint64_t fft_size, FFTNorm norm, const cudaExecutor &exec)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");

  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const auto stream = exec.getStream();

  // converts operators to tensors
  auto out = getCufft1DSupportedTensor(o, stream);
  auto in_t = getCufft1DSupportedTensor(i, stream);

  if(!in_t.isSameView(i)) {
   (in_t = i).run(stream);
  }

  // TODO should combine into function above
  // adjusts size of tensor based on fft_size
  auto in = detail::GetFFTInputView(out, in_t, fft_size, exec);

  // Get parameters required by these tensors
  auto params = detail::matxCUDAFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 1);
  params.stream = stream;

  using cache_val_type = detail::matxCUDAFFTPlan1D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::fft_cuda_cache_t>(),
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


template <typename OutputTensor, typename InputTensor>
__MATX_INLINE__ void fft2_impl(OutputTensor o, const InputTensor i, FFTNorm norm,
           const cudaExecutor &exec)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");

  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const auto stream = exec.getStream();

  auto out = detail::getCufft2DSupportedTensor(o, stream);
  auto in = detail::getCufft2DSupportedTensor(i, stream);

  if(!in.isSameView(i)) {
    printf("fft copy\n");
    (in = i).run(stream);
  }

  // Get parameters required by these tensors
  auto params = detail::matxCUDAFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 2);
  params.stream = stream;

  using cache_val_type = detail::matxCUDAFFTPlan2D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::fft_cuda_cache_t>(),
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
__MATX_INLINE__ void ifft2_impl(OutputTensor o, const InputTensor i, FFTNorm norm,
           const cudaExecutor &exec)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output operator ranks must match");

  MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

  const auto stream = exec.getStream();

  auto out = detail::getCufft2DSupportedTensor(o, stream);
  auto in = detail::getCufft2DSupportedTensor(i, stream);

  if(!in.isSameView(i)) {
    (in = i).run(stream);
  }

    // Get parameters required by these tensors
  auto params = detail::matxCUDAFFTPlan_t<decltype(out), decltype(in)>::GetFFTParams(out, in, 2);
  params.stream = stream;

  // Get cache or new FFT plan if it doesn't exist
  using cache_val_type = detail::matxCUDAFFTPlan2D_t<decltype(out), decltype(in)>;
  detail::GetCache().LookupAndExec<detail::fft_cuda_cache_t>(
    detail::GetCacheIdFromType<detail::fft_cuda_cache_t>(),
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

} // end namespace detail

}; // end namespace matx
