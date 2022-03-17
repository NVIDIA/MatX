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

#if MATX_ENABLE_CUTENSOR
#include <cstdio>
#include <numeric>
#include "matx_error.h"
#include "matx_tensor.h"


#include <cutensornet.h>
#include <cutensor.h>




namespace matx {
namespace detail {
namespace cutensor {

/**
 * Parameters needed for einsum.
 */
template <typename... InT>
struct EinsumParams_t {
  using first_type = std::tuple_element_t<0, std::tuple<InT...>>;
  using first_value_type = typename first_type::value_type;

  std::array<std::vector<int32_t>, sizeof...(InT) + 1> modes_;
  std::array<std::vector<int64_t>, sizeof...(InT) + 1> extents_;
  std::array<std::vector<int64_t>, sizeof...(InT) + 1> strides_;
  std::array<int32_t, sizeof...(InT)> nmodes_;
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
  using first_type = std::tuple_element_t<0, std::tuple<InT...>>;
  using first_value_type = typename first_type::value_type;

  //static_assert(TensorTypesMatch<InT...>(), "All tensor data types in contraction must match");

  matxEinsumHandle_t(OutputTensor &out, const std::string &subscripts, cudaStream_t stream, const InT&... tensors)
  {
    size_t i;
    params_ = GetEinsumParams(out, subscripts, tensors...);
    //cutensornetLoggerSetLevel(5);
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

    MATX_ASSERT_STR(cutensornetCreate(&handle_) == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to create cuTensorNet handle");    

    // setup tensor network
    MATX_ASSERT_STR_EXP(cutensornetCreateNetworkDescriptor(handle_,
                                                sizeof...(InT), 
                                                params_.nmodes_.data(), 
                                                extents, 
                                                strides, 
                                                modes, 
                                                params_.alignments_in_,
                                                out.Rank(), 
                                                extents[sizeof...(InT)], 
                                                strides[sizeof...(InT)], 
                                                modes[sizeof...(InT)], 
                                                params_.alignment_out_,
                                                MatXTypeToCudaType<typename OutputTensor::scalar_type>(), 
                                                CUTENSORNET_COMPUTE_32F,
                                                &descNet_), CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "Failed to create cuTensorNet network descriptor");

    cutensornetContractionOptimizerConfig_t optimizerConfig;
    MATX_ASSERT_STR(cutensornetCreateContractionOptimizerConfig(handle_, &optimizerConfig) == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to create cuTensorNet optimizer config");

    cutensornetContractionOptimizerInfo_t optimizerInfo;
    MATX_ASSERT_STR(cutensornetCreateContractionOptimizerInfo(handle_, descNet_, &optimizerInfo) == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
      "Failed to create cuTensorNet contraction optimizer info");

    size_t freeMem, totalMem;
    MATX_ASSERT(cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess, matxCudaError);

    // cuTensorNet recommends a large amount of memory for optimizing, but we limit to 2GB
    workSize_ = static_cast<decltype(workSize_)>(std::min(static_cast<double>(freeMem) * 0.8, 2e9));

    MATX_ASSERT_STR(cutensornetContractionOptimize(handle_,
                                              descNet_,
                                              optimizerConfig,
                                              workSize_,
                                              optimizerInfo) == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "Failed to create run cuTensorNet optimizer");

    MATX_ASSERT_STR(cutensornetContractionOptimizerInfoGetAttribute(
                handle_,
                optimizerInfo,
                CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                &params_.num_slices_,
                sizeof(params_.num_slices_)) == CUTENSORNET_STATUS_SUCCESS, matxcuTensorError,
        "Failed to get number of slices from cuTensorNet optimizer");

    MATX_ASSERT(params_.num_slices_ > 0, matxcuTensorError);

    /*******************************
     * Initialize all pair-wise contraction plans (for cuTENSOR)
     *******************************/
    MATX_ASSERT_STR(cutensornetCreateContractionPlan(handle_,
                                                descNet_,
                                                optimizerInfo,
                                                workSize_,
                                                &plan_) == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetCreateContractionPlan failed");


    /*******************************
    * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
    *******************************/
    cutensornetContractionAutotunePreference_t autotunePref;
    MATX_ASSERT_STR(cutensornetCreateContractionAutotunePreference(handle_,
                            &autotunePref) == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetCreateContractionAutotunePreference failed");

    // Allocate the real amount needed and free the old amount
    MATX_ASSERT_STR(cutensornetContractionGetWorkspaceSize(handle_, descNet_, optimizerInfo, &workSize_) == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetContractionGetWorkspaceSize failed");

    matxAlloc(&workspace_, workSize_, MATX_ASYNC_DEVICE_MEMORY, stream);

    const int numAutotuningIterations = 5; // may be 0
    MATX_ASSERT_STR(cutensornetContractionAutotunePreferenceSetAttribute(
                            handle_,
                            autotunePref,
                            CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                            &numAutotuningIterations,
                            sizeof(numAutotuningIterations)) == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetContractionAutotunePreferenceSetAttribute failed");

    // modify the plan again to find the best pair-wise contractions
    MATX_ASSERT_STR(cutensornetContractionAutotune(handle_,
                            plan_,
                            data_in,
                            out.Data(),
                            workspace_, 
                            workSize_,
                            autotunePref,
                            stream) == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetContractionAutotune failed");

    MATX_ASSERT_STR(cutensornetDestroyContractionAutotunePreference(autotunePref) == CUTENSORNET_STATUS_SUCCESS,
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
    EinsumParams_t<InT...> params; 
    std::vector<std::string> tokens;
    size_t i;

    ParseEinsum(subscripts, tokens);

    MATX_ASSERT_STR(tokens.size() - 1 == sizeof...(InT), matxInvalidDim, "Number of subscript groups in Einstein notation must match the number of tensors (input and output)");

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

    return params;
  }

  ~matxEinsumHandle_t()
  {
    matxFree(workspace_);
  }

  inline void Exec(OutputTensor &out, cudaStream_t stream, const InT... tensors)
  {
    void *data_in[sizeof...(InT)];
    size_t i = 0;
    ((data_in[i++] = tensors.Data()), ...);

    for (int64_t slice = 0; slice < params_.num_slices_; slice++)
    {

        MATX_ASSERT_STR(cutensornetContraction(handle_,
                                plan_,
                                data_in,
                                out.Data(),
                                workspace_, 
                                workSize_, 
                                slice, 
                                stream) == CUTENSORNET_STATUS_SUCCESS,
      matxcuTensorError, "cutensornetContraction failed");
    }
  }    

  private:
    static void StringToIntArray(const std::string_view &str, int32_t modes[]) {
      size_t mode = 0;
      for (const char &c: str) {
        modes[mode++] = static_cast<int32_t>(c);
      }
    }

    cutensornetContractionPlan_t plan_;
    uint64_t workSize_;
    void *workspace_;
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

template <typename... InT>
// Static caches of covariance matrices
static matxCache_t<EinsumParams_t<InT...>, EinsumParamsKeyHash<InT...>, EinsumParamsKeyEq<InT...>> einsum_cache;
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
  void einsum(OutputType &out, const std::string &subscripts, cudaStream_t stream, InT... tensors)
  {
#if MATX_ENABLE_CUTENSOR    
    // Get parameters required by these tensors
    auto params = matx::detail::cutensor::matxEinsumHandle_t<OutputType, InT...>::GetEinsumParams(out, subscripts, tensors...);
    params.stream = stream;

    auto ret = matx::detail::cutensor::einsum_cache<InT...>.Lookup(params);
    if (ret == std::nullopt) {
      auto tmp = new matx::detail::cutensor::matxEinsumHandle_t<OutputType, InT...>{out, subscripts, stream, tensors...};
      matx::detail::cutensor::einsum_cache<InT...>.Insert(params, static_cast<void *>(tmp));

      tmp->Exec(out, stream, tensors...);
    }
    else {
      auto einsum_type = static_cast<matx::detail::cutensor::matxEinsumHandle_t<OutputType, InT...> *>(ret.value());
      einsum_type->Exec(out, stream, tensors...);
    }
#endif    
  }
}

} // end namespace matx

