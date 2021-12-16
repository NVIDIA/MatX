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

#include "matx_cache.h"
#include "matx_dim.h"
#include "matx_error.h"
#include "matx_make.h"
#include "matx_tensor.h"

#include <cstdio>
#include <functional>
#include <optional>

namespace matx {

static constexpr int MAX_FFT_RANK = 2;

/**
 * Parameters needed to execute an FFT/IFFT in cuFFT
 */
struct FftParams_t {
  long long n[MAX_FFT_RANK] = {0};
  long long batch;
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
  static constexpr int RANK = OutTensorType::Rank();    

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
                      const InTensorType &i, cudaStream_t stream)
  {
    cufftSetStream(this->plan_, stream);
    Exec(o, i, CUFFT_FORWARD);
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
                      const InTensorType &i, cudaStream_t stream)
  {
    cufftSetStream(this->plan_, stream);
    Exec(o, i, CUFFT_INVERSE);

    // cuFFT doesn't scale IFFT the same as MATLAB/Python. Scale it here to
    // match
    if (params_.fft_rank == 1) {
      (o = o * 1.0 / static_cast<double>(params_.n[0])).run(stream);
    }
    else {
      (o = o * 1.0 / static_cast<double>(params_.n[0] * params_.n[1]))
          .run(stream);
    }
  }

  static FftParams_t GetFFTParams(OutTensorType &o,
                          const InTensorType &i, int fft_rank)
  {
    FftParams_t params;

    params.transform_type = DeduceFFTTransformType();
    params.input_type = matxFFTPlan_t<OutTensorType, InTensorType>::GetInputType();
    params.output_type = matxFFTPlan_t<OutTensorType, InTensorType>::GetOutputType();
    params.exec_type = matxFFTPlan_t<OutTensorType, InTensorType>::GetExecType();

    if (fft_rank == 1) {
      if constexpr (o.Rank() == 1 && i.Rank() == 1) {
        params.n[0] = (params.transform_type == CUFFT_C2R ||
                       params.transform_type == CUFFT_Z2D)
                          ? o.Size(0)
                          : i.Size(0);
        params.fft_rank = 1;
        params.batch = 1;
        params.inembed[0] = i.Size(0); // Unused
        params.onembed[0] = o.Size(0); // Unused
        params.istride = i.Stride(0);
        params.ostride = o.Stride(0);
        params.idist = i.Size(0);
        params.odist = o.Size(0);
      }
      else if constexpr (o.Rank() == 2 && i.Rank() == 2) {
        params.n[0] = (params.transform_type == CUFFT_C2R ||
                       params.transform_type == CUFFT_Z2D)
                          ? o.Size(1)
                          : i.Size(1);
        params.fft_rank = 1;
        params.batch = i.Size(0);
        params.inembed[0] = i.Size(1); // Unused
        params.onembed[0] = o.Size(1); // Unused
        params.istride = i.Stride(1);
        params.ostride = o.Stride(1);
        params.idist = i.Stride(0);
        params.odist = o.Stride(0);
      }
      else if constexpr (o.Rank() == 3 && i.Rank() == 3) {
        params.n[0] = (params.transform_type == CUFFT_C2R ||
                       params.transform_type == CUFFT_Z2D)
                          ? o.Size(2)
                          : i.Size(2);
        params.fft_rank = 1;
        params.batch = i.Size(1);
        params.inembed[0] = i.Size(2); // Unused
        params.onembed[0] = o.Size(2); // Unused
        params.istride = i.Stride(2);
        params.ostride = o.Stride(2);
        params.idist = i.Stride(1);
        params.odist = o.Stride(1);
      }
      else if constexpr (o.Rank() == 4 && i.Rank() == 4) {
        params.n[0] = (params.transform_type == CUFFT_C2R ||
                       params.transform_type == CUFFT_Z2D)
                          ? o.Size(3)
                          : i.Size(3);
        params.fft_rank = 1;
        params.batch = i.Size(2);
        params.inembed[0] = i.Size(3); // Unused
        params.onembed[0] = o.Size(3); // Unused
        params.istride = i.Stride(3);
        params.ostride = o.Stride(3);
        params.idist = i.Stride(2);
        params.odist = o.Stride(2);
      }

      if constexpr (is_complex_half_v<T1> || is_complex_half_v<T1>) {
        if ((params.n[0] & (params.n[0] - 1)) != 0) {
          MATX_THROW(matxInvalidDim,
                     "Half precision only supports power of two transforms");
        }
      }
    }
    else if (fft_rank == 2) {
      if constexpr (o.Rank() == 2 && i.Rank() == 2) {
        if (params.transform_type == CUFFT_C2R ||
            params.transform_type == CUFFT_Z2D) {
          params.n[0] = o.Size(1);
          params.n[1] = o.Size(0);
        }
        else {
          params.n[0] = i.Size(1);
          params.n[1] = i.Size(0);
        }
        params.batch = 1;
        params.fft_rank = 2;
        params.inembed[1] = o.Size(1);
        params.onembed[1] = i.Size(1); //??
        params.istride = i.Stride(1);
        params.ostride = o.Stride(1);
        params.idist = i.Size(0) * i.Size(1);
        params.odist = o.Size(0) * o.Size(1);
      }
      else if constexpr (o.Rank() == 3 && i.Rank() == 3) {
        if (params.transform_type == CUFFT_C2R ||
            params.transform_type == CUFFT_Z2D) {
          params.n[0] = o.Size(2);
          params.n[1] = o.Size(1);
        }
        else {
          params.n[0] = i.Size(2);
          params.n[1] = i.Size(1);
        }

        params.batch = i.Size(0);
        params.fft_rank = 2;
        params.inembed[1] = o.Size(2);
        params.onembed[1] = i.Size(2);
        params.istride = i.Stride(2);
        params.ostride = o.Stride(2);
        params.idist = i.Size(1) * i.Size(2);
        params.odist = o.Size(1) * o.Size(2);
      }
      else if constexpr (o.Rank() == 4 && i.Rank() == 4) {
        if (params.transform_type == CUFFT_C2R ||
            params.transform_type == CUFFT_Z2D) {
          params.n[0] = o.Size(3);
          params.n[1] = o.Size(2);
        }
        else {
          params.n[0] = i.Size(3);
          params.n[1] = i.Size(2);
        }

        params.batch = i.Size(0) * i.Size(1);
        params.fft_rank = 2;
        params.inembed[1] = o.Size(3);
        params.onembed[1] = i.Size(3);
        params.istride = i.Stride(3);
        params.ostride = o.Stride(3);
        params.idist = i.Size(2) * i.Size(3);
        params.odist = o.Size(2) * o.Size(3);
      }

      if constexpr (is_complex_half_v<T1> || is_complex_half_v<T1>) {
        if ((params.n[0] & (params.n[0] - 1)) != 0 ||
            (params.n[1] & (params.n[1] - 1)) != 0) {
          MATX_THROW(matxInvalidDim,
                     "Half precision only supports power of two transforms");
        }
      }
    }

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

  /**
   * Destructs an FFT plan
   *
   * Frees all memory associated with the plan.
   **/
  virtual ~matxFFTPlan_t()
  {
    if (workspace_ != nullptr) {
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
 *
 * */
matxFFTPlan1D_t(OutTensorType &o, const InTensorType &i)
{
  int dev;
  cudaGetDevice(&dev);

  this->workspace_ = nullptr;
  this->params_ = this->GetFFTParams(o, i, 1);

  if (this->params_.transform_type == CUFFT_C2R ||
      this->params_.transform_type == CUFFT_Z2D) {
    MATX_ASSERT(!is_cuda_complex_v<T1> && is_cuda_complex_v<T2>,
                matxInvalidType);
  }
  else if (this->params_.transform_type == CUFFT_R2C ||
           this->params_.transform_type == CUFFT_D2Z) {
    MATX_ASSERT(!is_cuda_complex_v<T2> && is_cuda_complex_v<T1>,
                matxInvalidType);
  }
  else {
    MATX_ASSERT(is_complex_v<T2> && is_complex_v<T1>, matxInvalidType);
    MATX_ASSERT((std::is_same_v<T1, T2>), matxInvalidType);
  }

  if constexpr (RANK >= 2) {
    for (int r = 0; r < RANK - 1; r++) {
      MATX_ASSERT(o.Size(r) == i.Size(r), matxInvalidSize);
    }
  }

  size_t workspaceSize;
  cufftCreate(&this->plan_);
  cufftResult error;
  cufftXtGetSizeMany(this->plan_, 1, this->params_.n, this->params_.inembed,
                      this->params_.istride, this->params_.idist,
                      this->params_.input_type, this->params_.onembed,
                      this->params_.ostride, this->params_.odist,
                      this->params_.output_type, this->params_.batch,
                      &workspaceSize, this->params_.exec_type);

  matxAlloc((void **)&this->workspace_, workspaceSize);
  cudaMemPrefetchAsync(this->workspace_, workspaceSize, dev, 0);
  cufftSetWorkArea(this->plan_, this->workspace_);

  error = cufftXtMakePlanMany(
      this->plan_, 1, this->params_.n, this->params_.inembed,
      this->params_.ostride, this->params_.idist, this->params_.input_type,
      this->params_.onembed, this->params_.ostride, this->params_.odist,
      this->params_.output_type, this->params_.batch, &workspaceSize,
      this->params_.exec_type);

  MATX_ASSERT(error == CUFFT_SUCCESS, matxCufftError);
}

private:
virtual void inline Exec(OutTensorType &o, const InTensorType &i,
                         int dir) override
{
  if constexpr (OutTensorType::Rank() <= 2) {
    this->InternalExec(static_cast<const void *>(i.Data()),
                      static_cast<void *>(o.Data()), dir);
  }
  else if constexpr (OutTensorType::Rank() == 3) {
    for (index_t z = 0; z < o.Size(0); z++) {
      this->InternalExec(static_cast<const void *>(&i(z, 0, 0)),
                        static_cast<void *>(&o(z, 0, 0)), dir);
    }
  }
  else if constexpr (OutTensorType::Rank() == 4) {
    for (index_t z = 0; z < o.Size(0); z++) {
      for (index_t y = 0; y < o.Size(1); y++) {
        this->InternalExec(static_cast<const void *>(&i(z, y, 0, 0)),
                          static_cast<void *>(&o(z, y, 0, 0)), dir);
      }
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
  matxFFTPlan2D_t(OutTensorType &o, const InTensorType &i)
  {
    static_assert(RANK >= 2, "2D FFTs require a rank-2 tensor or higher");
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
    cufftResult error;
    cufftXtGetSizeMany(this->plan_, 2, this->params_.n, this->params_.inembed,
                       this->params_.istride, this->params_.idist,
                       this->params_.input_type, this->params_.onembed,
                       this->params_.ostride, this->params_.odist,
                       this->params_.output_type, this->params_.batch,
                       &workspaceSize, this->params_.exec_type);

    matxAlloc((void **)&this->workspace_, workspaceSize);
    cudaMemPrefetchAsync(this->workspace_, workspaceSize, dev, 0);
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
    static_assert(RANK >= 2);

    if constexpr (RANK <= 3) {
      this->InternalExec(static_cast<const void *>(i.Data()),
                         static_cast<void *>(o.Data()), dir);
    }
    else if constexpr (RANK == 4) {
      for (index_t z = 0; z < o.Size(0); z++) {
        this->InternalExec(static_cast<const void *>(&i(z, 0, 0, 0)),
                          static_cast<void *>(&o(z, 0, 0, 0)), dir);
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
           l.exec_type == t.exec_type;
  }
};

// Static caches of 1D and 2D FFTs
static matxCache_t<FftParams_t, FftParamsKeyHash, FftParamsKeyEq> cache_1d;
static matxCache_t<FftParams_t, FftParamsKeyHash, FftParamsKeyEq> cache_2d;

template <typename TensorType1, typename TensorType2>
auto  GetFFTInputView([[maybe_unused]] TensorType1 &o,
                    const TensorType2 &i,
                    [[maybe_unused]] cudaStream_t stream)
{
  using index_type = typename TensorType1::shape_type;
  using T1    = typename TensorType1::scalar_type;
  using T2    = typename TensorType2::scalar_type;
  constexpr int RANK = TensorType1::Rank();

  index_type starts[RANK] = {0};
  index_type ends[RANK];
  index_type in_size = i.Lsize();
  index_type nom_fft_size = in_size;
  index_type act_fft_size = o.Lsize();

  // If R2C transform, set length of FFT appropriately
  if constexpr ((std::is_same_v<T2, float> &&
                 std::is_same_v<T1, cuda::std::complex<float>>) ||
                (std::is_same_v<T2, double> &&
                 std::is_same_v<T1, cuda::std::complex<double>>) ||
                (std::is_same_v<T2, matxBf16> &&
                 std::is_same_v<T1, matxBf16Complex>) ||
                (std::is_same_v<T2, matxFp16> &&
                 std::is_same_v<T1, matxFp16Complex>)) { // R2C
    nom_fft_size = in_size / 2 + 1;
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

  // Set up new shape if transform size doesn't match tensor
  if (nom_fft_size != act_fft_size) {
    std::fill_n(ends, RANK, matxEnd);

    // FFT shorter than the size of the input signal. Create a new view of this
    // slice.
    if (act_fft_size < nom_fft_size) {
      ends[RANK - 1] = nom_fft_size;
      return i.Slice(starts, ends);
    }
    else { // FFT length is longer than the input. Pad input
      T2 *i_pad;

      // If the input needs to be padded we have to temporarily allocate a new
      // buffer, zero the output, then copy our input buffer. This is not very
      // efficient, but if cufft adds a zero-padding feature later we can take
      // advantage of that without changing the API.

      // Create a new shape where n is the size of the last dimension
      auto desc = i.Descriptor();
      desc.SetSize(RANK - 1, act_fft_size);

      // Make a new buffer large enough for our input
      matxAlloc(reinterpret_cast<void **>(&i_pad),
                sizeof(T1) * desc.TotalSize(), MATX_ASYNC_DEVICE_MEMORY,
                stream);

      auto i_new = make_tensor(i_pad, desc);
      ends[RANK - 1] = i.Lsize();
      auto i_pad_part_v = i_new.Slice(starts, ends);

      (i_new = static_cast<promote_half_t<T2>>(0)).run(stream);
      matx::copy(i_pad_part_v, i, stream);
      return i_new;
    }
  }

  return i;
}

/**
 * Run a 1D FFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 1D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam T1
 *   Output view data type
 * @tparam T2
 *   Input view data type
 * @tparam RANK
 *   Rank of input and output tensors
 * @param o
 *   Output tensor. The length of the fastest-changing dimension dictates the
 * size of FFT. If this size is longer than the length of the input tensor, the
 * tensor will potentially be copied and zero-padded to a new block of memory.
 * Future releases may remove this restriction to where there is no copy.
 * @param i
 *   input tensor
 * @param stream
 *   CUDA stream
 */
template <typename TensorType1, typename TensorType2>
void fft(TensorType1 &o, const TensorType2 &i,
         cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(TensorType1::Rank() == TensorType2::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");  
  auto i_new = GetFFTInputView(o, i, stream);

  // Get parameters required by these tensors
  auto params = matxFFTPlan_t<TensorType1, TensorType2>::GetFFTParams(o, i_new, 1);
  params.stream = stream;

  // Get cache or new FFT plan if it doesn't exist
  auto ret = cache_1d.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new matxFFTPlan1D_t<TensorType1, TensorType2>{o, i_new};
    cache_1d.Insert(params, static_cast<void *>(tmp));
    tmp->Forward(o, i_new, stream);
  }
  else {
    auto fft_type = static_cast<matxFFTPlan1D_t<TensorType1, TensorType2> *>(ret.value());
    fft_type->Forward(o, i_new, stream);
  }
  
  // If we async-allocated memory for zero-padding, free it here
  // if (i_new.Data() != i.Data()) {
  //   matxFree(i_new.Data());
  // }
}

/**
 * Run a 1D IFFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 1D IFFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam T1
 *   Output view data type
 * @tparam T2
 *   Input view data type
 * @tparam RANK
 *   Rank of input and output tensors
 * @param o
 *   Output tensor. The length of the fastest-changing dimension dictates the
 * size of FFT. If this size is longer than the length of the input tensor, the
 * tensor will potentially be copied and zero-padded to a new block of memory.
 * Future releases may remove this restriction to where there is no copy.
 * @param i
 *   input tensor
 * @param stream
 *   CUDA stream
 */
template <typename TensorType1, typename TensorType2>
void ifft(TensorType1 &o, const TensorType2 &i,
          cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(TensorType1::Rank() == TensorType2::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");

  auto i_new = GetFFTInputView(o, i, stream);

  // Get parameters required by these tensors
  auto params = matxFFTPlan_t<TensorType1, TensorType2>::GetFFTParams(o, i_new, 1);
  params.stream = stream;

  // Get cache or new FFT plan if it doesn't exist
  auto ret = cache_1d.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new matxFFTPlan1D_t<TensorType1, TensorType2>{o, i_new};
    cache_1d.Insert(params, static_cast<void *>(tmp));
    tmp->Inverse(o, i_new, stream);
  }
  else {
    auto fft_type = static_cast<matxFFTPlan1D_t<TensorType1, TensorType2> *>(ret.value());
    fft_type->Inverse(o, i_new, stream);
  }

  // If we async-allocated memory for zero-padding, free it here
  // if (i_new.Data() != i.Data()) {
  //   matxFree(i_new.Data());
  // }
}

/**
 * Run a 2D FFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 2D FFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam T1
 *   Output view data type
 * @tparam T2
 *   Input view data type
 * @tparam RANK
 *   Rank of input and output tensors
 * @param o
 *   Output tensor
 * @param i
 *   input tensor
 * @param stream
 *   CUDA stream
 */
template <typename OutputTensor, typename InputTensor>
void fft2(OutputTensor &o, const InputTensor &i,
           cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(OutputTensor::Rank() == InputTensor::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");

  // Get parameters required by these tensors
  auto params = matxFFTPlan_t<OutputTensor, InputTensor>::GetFFTParams(o, i, 2);
  params.stream = stream;

  // Get cache or new FFT plan if it doesn't exist
  auto ret = cache_2d.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new matxFFTPlan2D_t<OutputTensor, InputTensor>{o, i};
    cache_2d.Insert(params, static_cast<void *>(tmp));
    tmp->Forward(o, i, stream);
  }
  else {
    auto fft_type = static_cast<matxFFTPlan2D_t<OutputTensor, InputTensor> *>(ret.value());
    fft_type->Forward(o, i, stream);
  }
}

/**
 * Run a 2D IFFT with a cached plan
 *
 * Creates a new FFT plan in the cache if none exists, and uses that to execute
 * the 2D IFFT. Note that FFTs and IFFTs share the same plans if all dimensions
 * match
 *
 * @tparam T1
 *   Output view data type
 * @tparam T2
 *   Input view data type
 * @tparam RANK
 *   Rank of input and output tensors
 * @param o
 *   Output tensor
 * @param i
 *   input tensor
 * @param stream
 *   CUDA stream
 */
template <typename TensorType1, typename TensorType2>
void ifft2(TensorType1 &o, const TensorType2 &i,
           cudaStream_t stream = 0)
{
  MATX_STATIC_ASSERT_STR(TensorType1::Rank() == TensorType2::Rank(), matxInvalidDim,
    "Input and output tensor ranks must match");

  // Get parameters required by these tensors
  auto params = matxFFTPlan_t<TensorType1, TensorType2>::GetFFTParams(o, i, 2);
  params.stream = stream;

  // Get cache or new FFT plan if it doesn't exist
  auto ret = cache_2d.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new matxFFTPlan2D_t<TensorType1, TensorType2>{o, i};
    cache_2d.Insert(params, static_cast<void *>(tmp));
    tmp->Inverse(o, i, stream);
  }
  else {
    auto fft_type = static_cast<matxFFTPlan2D_t<TensorType1, TensorType2> *>(ret.value());
    fft_type->Inverse(o, i, stream);
  }
}
}; // end namespace matx
