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

#ifdef MATX_EN_CUTENSOR
#include <cstdio>
#include <numeric>
#include "error.h"

#include <cutensornet.h>
#include <cutensor.h>

#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"



namespace matx {
namespace detail {
namespace cutensor {

/**
 * Parameters needed for einsum.
 */
template <typename... InT>
struct EinsumParams_t {
  using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<InT...>>;
  using first_value_type = typename first_type::value_type;

  cuda::std::array<std::vector<int32_t>, sizeof...(InT) + 1> modes_;
  cuda::std::array<std::vector<int64_t>, sizeof...(InT) + 1> extents_;
  cuda::std::array<std::vector<int64_t>, sizeof...(InT) + 1> strides_;
  cuda::std::array<int32_t, sizeof...(InT)> nmodes_;
  uint32_t alignment_out_;
  uint32_t alignments_in_[sizeof...(InT)];
  int64_t num_slices_;

  std::string subs;
  MatXDataType_t dtype;
  cudaStream_t stream;
};

template <typename OutputTensor, typename... InT>
class matxEinsumHandle_t {
public:
  using first_type = cuda::std::tuple_element_t<0, cuda::std::tuple<InT...>>;
  using first_value_type = typename first_type::value_type;

  //static_assert(TensorTypesMatch<InT...>(), "All tensor data types in contraction must match");

  matxEinsumHandle_t(OutputTensor &out, const std::string &subscripts, cudaStream_t stream, const InT&... tensors)
  {
    [[maybe_unused]] cutensornetStatus_t status;
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    size_t i;
    params_ = GetEinsumParams(out, subscripts, tensors...);

    // cutensornetLoggerSetLevel(5); // Turn on to see debug from cuTENSOR/cuTensorNet

    // Convert all parameter structures
    int32_t *modes[sizeof...(InT) + 1];
    int64_t *extents[sizeof...(InT) + 1];
    int64_t *strides[sizeof...(InT) + 1];
    void *data_in[sizeof...(InT)];
    for (i = 0; i < sizeof...(InT) + 1; i++) {
      modes[i] = params_.modes_[i].data();
      extents[i] = params_.extents_[i].data();
      strides[i] = params_.strides_[i].data();
    }

    i = 0;
    ((data_in[i++] = tensors.Data()), ...);

    status = cutensornetCreate(&handle_);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to create cuTensorNet handle");

    // setup tensor network
    status = cutensornetCreateNetworkDescriptor(handle_,
                                                sizeof...(InT),
                                                params_.nmodes_.data(),
                                                extents,
                                                strides,
                                                modes,
                                                nullptr,
                                                out.Rank(),
                                                extents[sizeof...(InT)],
                                                strides[sizeof...(InT)],
                                                modes[sizeof...(InT)],
                                                MatXTypeToCudaType<typename OutputTensor::value_type>(),
                                                CUTENSORNET_COMPUTE_32F,
                                                &descNet_);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "Failed to create cuTensorNet network descriptor");

    cutensornetContractionOptimizerConfig_t optimizerConfig;
    status = cutensornetCreateContractionOptimizerConfig(handle_, &optimizerConfig);

    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to create cuTensorNet optimizer config");

    cutensornetContractionOptimizerInfo_t optimizerInfo;
    status = cutensornetCreateContractionOptimizerInfo(handle_, descNet_, &optimizerInfo);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to create cuTensorNet contraction optimizer info");

    int imbalance_factor = 30;
    status = cutensornetContractionOptimizerConfigSetAttribute(
                                                               handle_,
                                                               optimizerConfig,
                                                               CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR,
                                                               &imbalance_factor,
                                                               sizeof(imbalance_factor));
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError, "Failed to run contraction optimizer");

    size_t freeMem, totalMem;
    auto err = cudaMemGetInfo(&freeMem, &totalMem);
    MATX_ASSERT(err == cudaSuccess, matxCudaError);

    // cuTensorNet recommends a large amount of memory for optimizing, but we limit to 2GB
    workSize_ = static_cast<decltype(workSize_)>(std::min(static_cast<double>(freeMem) * 0.8, 2e9));

    status = cutensornetContractionOptimize(handle_,
                                              descNet_,
                                              optimizerConfig,
                                              workSize_,
                                              optimizerInfo);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "Failed to create run cuTensorNet optimizer");

    status = cutensornetContractionOptimizerInfoGetAttribute(
                handle_,
                optimizerInfo,
                CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                &params_.num_slices_,
                sizeof(params_.num_slices_));
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
        "Failed to get number of slices from cuTensorNet optimizer");

    MATX_ASSERT(params_.num_slices_ > 0, matxcuTensorError);

    status = cutensornetCreateWorkspaceDescriptor(handle_, &workDesc_);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError, "Failed to create cuTENSOR workspace descriptor");

    int64_t requiredWorkspaceSize = 0;
    status = cutensornetWorkspaceComputeContractionSizes(handle_,
                                                         descNet_,
                                                         optimizerInfo,
                                                         workDesc_);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
        "Failed to compute cuTENSOR workspace size");

    status = cutensornetWorkspaceGetMemorySize(handle_,
                                                   workDesc_,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &requiredWorkspaceSize);

    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to get cuTENSOR memory size");

    MATX_ASSERT_STR(workSize_ > requiredWorkspaceSize, matxOutOfMemory, "Not enough workspace memory is available.");

    matxAlloc(&workspace_, requiredWorkspaceSize, MATX_ASYNC_DEVICE_MEMORY, stream);

    status = cutensornetWorkspaceSetMemory(handle_,
                                               workDesc_,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               workspace_,
                                               requiredWorkspaceSize);

    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to set cuTENSOR memory");

    /*******************************
     * Initialize all pair-wise contraction plans (for cuTENSOR)
     *******************************/
    status = cutensornetCreateContractionPlan(handle_,
                                                descNet_,
                                                optimizerInfo,
                                                workDesc_,
                                                &plan_);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetCreateContractionPlan failed");


    /*******************************
    * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
    *******************************/
    cutensornetContractionAutotunePreference_t autotunePref;
    status = cutensornetCreateContractionAutotunePreference(handle_,
                            &autotunePref);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetCreateContractionAutotunePreference failed");

    const int numAutotuningIterations = 5; // may be 0
    status = cutensornetContractionAutotunePreferenceSetAttribute(
                            handle_,
                            autotunePref,
                            CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                            &numAutotuningIterations,
                            sizeof(numAutotuningIterations));
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetContractionAutotunePreferenceSetAttribute failed");

    // modify the plan again to find the best pair-wise contractions
    status = cutensornetContractionAutotune(handle_,
                            plan_,
                            data_in,
                            out.Data(),
                            workDesc_,
                            autotunePref,
                            stream);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetContractionAutotune failed");

    status = cutensornetDestroyContractionAutotunePreference(autotunePref);
    MATX_ASSERT_STR(status == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetDestroyContractionAutotunePreference failed");

  }

  /**
   * @brief Tokenizes an einsum string into a vector
   *
   * @param str einsum string
   * @param out tokenized vector
   * @return true if tokenized successfully, or false otherwise
   */
  static bool ParseEinsum(const std::string &str, std::vector<std::string> &out) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    // Find output separator
    auto iout = str.find("->");
    if (iout == std::string::npos) {
      return false;
    }

    auto istr = str.substr(0, iout);
    size_t start = 0;
    while (start < istr.size()) {
      auto sep = istr.find(",", start);
      if (sep == std::string::npos) {
        out.push_back(istr.substr(start));
        break;
      }

      out.push_back(istr.substr(start, sep - start));
      start += sep - start + 1;
    }

    // Nothing after the output separator -> indicates this is a 0D output
    out.push_back(str.substr(iout + 2));

    return true;
  }

  static EinsumParams_t<InT...> GetEinsumParams(OutputTensor &out, const std::string &subscripts, const InT&... tensors)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    EinsumParams_t<InT...> params;
    std::vector<std::string> tokens;
    size_t i;

    ParseEinsum(subscripts, tokens);

    MATX_ASSERT_STR(tokens.size() - 1 == sizeof...(InT), matxInvalidDim, "Number of subscript groups in Einstein notation must match the number of operators (input and output)");

    // Set all the token characters
    for (i = 0; i < tokens.size(); i++) {
      for (const char &c: tokens[i]) {
        params.modes_[i].push_back(static_cast<int32_t>(c));
      }
    }

    i = 0;
    ((params.nmodes_[i++] = tensors.Rank()), ...);

    i = 0;
    MATX_ASSERT_STR(((tokens[i++].length() == static_cast<size_t>(tensors.Rank())), ...), matxInvalidDim,
        "Tensor rank must match number of einsum subscripts");

    auto set_sizes = [](auto &t, std::vector<int64_t> &sizes) {
      for (int32_t s = 0; s < t.Rank(); s++) {
        if constexpr (t.Rank() > 0) {
          sizes.push_back(t.Size(s));
        }
        else {
          sizes.push_back(1);
        }
      }
    };

    auto set_strides = [](auto &t, std::vector<int64_t> &strides) {
      if constexpr (t.Rank() > 0) {
        for (int32_t s = 0; s < t.Rank(); s++) {
          strides.push_back(t.Stride(s));
        }
      }
      else {
        strides.push_back(1);
      }
    };

    i = 0;
    ((set_sizes(tensors, params.extents_[i++])), ...);
    set_sizes(out, params.extents_[i]); // output tensor

    i = 0;
    ((set_strides(tensors, params.strides_[i++])), ...);
    set_strides(out, params.strides_[i]); // output tensor

    // Align pointers
    // Notice that pointers are allocated via cudaMalloc are aligned to 256 byte
    // boundaries by default; however here we're checking the pointer alignment explicitly
    // to demonstrate how one would check the alignment for arbitrary pointers.
    auto getMaximalPointerAlignment = [](const void* ptr) {
      const uint64_t ptrAddr  = reinterpret_cast<uint64_t>(ptr);
      uint32_t alignment = 1;
      while(ptrAddr % alignment == 0 &&
            alignment < 256) // at the latest we terminate once the alignment reached 256 bytes (we could be going, but any alignment larger or equal to 256 is equally fine)
      {
          alignment *= 2;
      }
      return alignment;
    };

    i = 0;
    ((params.alignments_in_[i++] = getMaximalPointerAlignment(tensors.Data())), ...);

    params.alignment_out_ = getMaximalPointerAlignment(out.Data());
    params.dtype = TypeToInt<typename EinsumParams_t<InT...>::first_value_type>();

    params.subs = subscripts;
    return params;
  }

  ~matxEinsumHandle_t()
  {
    matxFree(workspace_, cudaStreamDefault);
  }

  inline void Exec(OutputTensor &out, cudaStream_t stream, const InT... tensors)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    [[maybe_unused]] cutensornetStatus_t status;

    cutensornetSliceGroup_t sliceGroup{};
    status = cutensornetCreateSliceGroupFromIDRange(handle_, 0, params_.num_slices_, 1, &sliceGroup);
    MATX_ASSERT_STR_EXP(status, CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetCreateSliceGroupFromIDRange failed");

    void *data_in[sizeof...(InT)];
    size_t i = 0;
    ((data_in[i++] = tensors.Data()), ...);

    int32_t accumulateOutput = 0;
    status = cutensornetContractSlices(handle_,
                              plan_,
                              data_in,
                              out.Data(),
                              accumulateOutput,
                              workDesc_,
                              sliceGroup,
                              stream);

    MATX_ASSERT_STR_EXP(status, CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetContraction failed");

  }

  private:
    static void StringToIntArray(const std::string_view &str, int32_t modes[]) {
      size_t mode = 0;
      for (const char &c: str) {
        modes[mode++] = static_cast<int32_t>(c);
      }
    }

    cutensornetContractionPlan_t plan_;
    int64_t workSize_;
    void *workspace_;
    cutensornetWorkspaceDescriptor_t workDesc_;
    cutensornetHandle_t handle_;
    cutensornetNetworkDescriptor_t descNet_;
    EinsumParams_t<InT...> params_;
};

template <typename... InT>
bool operator==(const EinsumParams_t<InT...> &l, const EinsumParams_t<InT...> &r) {
  return l.modes_ == r.modes_ &&
          l.extents_ == r.extents_ &&
          l.strides_ == r.strides_ &&
          l.nmodes_ == r.nmodes_ &&
          l.alignment_out_ == r.alignment_out_ &&
          std::equal(std::begin(l.alignments_in_), std::end(l.alignments_in_), std::begin(r.alignments_in_)) &&
          l.num_slices_ == r.num_slices_ &&
          l.subs == r.subs &&
          l.dtype == r.dtype &&
          l.stream == r.stream;
}

template <typename... InT>
struct EinsumParamsKeyHash {
  std::size_t operator()(const EinsumParams_t<InT...> &k) const noexcept
  {
    return std::hash<std::string>()(k.subs) +
           std::hash<uint64_t>()((size_t)k.stream);
  }
};


template <typename... InT>
struct EinsumParamsKeyEq {
  bool operator()(const EinsumParams_t<InT...> &l, const EinsumParams_t<InT...> &t) const noexcept
  {
    return  l == t;
  }
};

} // end namespace cutensor
} // end namespace detail
} // end namespace matx
#endif

namespace matx {
namespace cutensor {

  /**
   * @brief Evaluates the Einstein summation on the operands
   *
   * einsum() is a multi-purpose tool capable of performing various operations on tensors in a compact
   * syntax. A non-exhaustive list of operations are: tensor contractions, GEMMs, dot products, and tranposes.
   * Because einsum is extremely powerful, not all features are supported or tested in MatX yet. Currently only
   * tensor contractions are tested. Other operations may work, but they're not tested yet.
   *
   * MatX uses a syntax very similar to NumPy's einsum syntax:
   * https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
   *
   * Ellipses are not supported yet, but a variadic list of tensors for contraction is supported. The output
   * operator '->' is required in MatX currently, and serves to provide error checking on the output tensor size.
   *
   *
   * @tparam OutputType Output tensor type
   * @tparam InT Types of input tensors
   * @param out Output tensor
   * @param subscripts String containing Einstein notation of operation to perform
   * @param stream CUDA stream
   * @param tensors List of input tensors
   */
  template <typename OutputType, typename... InT>
  void einsum_impl([[maybe_unused]] OutputType &out, [[maybe_unused]] const std::string &subscripts, [[maybe_unused]] cudaStream_t stream, [[maybe_unused]] InT... tensors)
  {
#ifdef MATX_EN_CUTENSOR
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    // Get parameters required by these tensors
    auto params = matx::detail::cutensor::matxEinsumHandle_t<OutputType, InT...>::GetEinsumParams(out, subscripts, tensors...);
    params.stream = stream;

    using einsum_cache_t = std::unordered_map<detail::cutensor::EinsumParams_t<InT...>, std::any, detail::cutensor::EinsumParamsKeyHash<InT...>, detail::cutensor::EinsumParamsKeyEq<InT...>>;
    using cache_val_type = matx::detail::cutensor::matxEinsumHandle_t<OutputType, InT...>;
    detail::GetCache().LookupAndExec<einsum_cache_t>(
      detail::GetCacheIdFromType<einsum_cache_t>(),
      params,
      [&]() {
        auto tmp = std::make_shared<cache_val_type>(out, subscripts, stream, tensors...);
        return tmp;
      },
      [&](std::shared_ptr<cache_val_type> ctype) {
        ctype->Exec(out, stream, tensors...);
      }
    );
#else
    MATX_THROW(matxNotSupported, "einsum() currently requires MATX_EN_CUTENSOR=ON but MATX_EN_CUTENSOR=OFF");
#endif
  }
}

} // end namespace matx
