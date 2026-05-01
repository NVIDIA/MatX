////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include <vector>
#include <string>
#include <numeric>
#include "matx/core/defines.h"
#include "matx/core/error.h"
#include "matx/core/storage.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/core/utils.h"
#include "matx/core/capabilities.h"
#include "matx/core/vector.h"
#include "matx/core/nvtx.h"
#include "matx/operators/set.h"

namespace matx {

namespace detail {

/**
 * @brief Generate a JIT class name for a tensor with runtime rank/shape/strides.
 *
 * Delegates to the common make_jit_tensor_class_name_str in utils.h.
 */
template <typename T>
__MATX_INLINE__ std::string make_jit_tensor_class_name(int rank, const index_t *shape, const index_t *strides) {
  return make_jit_tensor_class_name_str(type_to_string_c_name<T>(), rank, shape, strides);
}

/**
 * @brief Generate JIT operator struct code for a tensor with runtime rank/shape/strides.
 *
 * Delegates to the common make_jit_tensor_struct_str in utils.h.
 */
template <typename T, typename StrideType>
__MATX_INLINE__ auto make_jit_tensor_op_str(
    const std::string &class_name,
    int rank,
    const std::string &sizes_str,
    const std::string &strides_str)
{
  const std::string rank_str = std::to_string(rank);
  return cuda::std::make_tuple(
    class_name,
    make_jit_tensor_struct_str(
      class_name,
      type_to_string<T>(),
      type_to_string<StrideType>(),
      rank_str,
      sizes_str,
      strides_str)
  );
}

template <typename ShapeType>
  requires is_tuple_c<remove_cvref_t<ShapeType>>
__MATX_INLINE__ std::vector<index_t> shape_to_vector(ShapeType &&shape)
{
  std::vector<index_t> out;
  constexpr int rank = static_cast<int>(cuda::std::tuple_size_v<remove_cvref_t<ShapeType>>);
  out.reserve(rank);
  cuda::std::apply([&out](auto... dims) { (out.push_back(static_cast<index_t>(dims)), ...); },
                   std::forward<ShapeType>(shape));
  return out;
}

__MATX_INLINE__ std::vector<index_t> shape_to_vector(const std::vector<index_t> &shape)
{
  return shape;
}

__MATX_INLINE__ std::vector<index_t> shape_to_vector(std::initializer_list<index_t> shape)
{
  return std::vector<index_t>(shape);
}

__MATX_INLINE__ index_t total_size_from_shape(const std::vector<index_t> &shape)
{
  index_t total = 1;
  for (const auto dim : shape) {
    MATX_ASSERT_STR(dim >= 0, matxInvalidSize, "Dynamic tensor shape dimensions must be non-negative");
    total *= dim;
  }
  return total;
}

/**
 * @brief A tensor type with rank determined at runtime.
 *
 * Unlike tensor_t<T, RANK>, dynamic_tensor_t<T> does not encode the rank as a
 * template parameter. The rank, shape, and strides are all stored at runtime.
 *
 * Dynamic tensors can ONLY be executed via CUDAJITExecutor. At JIT compilation
 * time the runtime rank is resolved to a concrete static rank, and the
 * generated CUDA kernel code is identical to what a static tensor_t produces.
 *
 * @tparam T Element type
 */
template <typename T>
class dynamic_tensor_t {
public:
  // Type traits consumed by the operator expression system
  using matxop = bool;
  using matxoplvalue = bool;
  using tensor_view = bool;
  using dynamic_tensor = bool;
  using type = T;
  using value_type = T;
  using stride_type = index_t;
  using shape_container = std::vector<index_t>;

  dynamic_tensor_t() = default;

  explicit dynamic_tensor_t(matxMemorySpace_t space, cudaStream_t stream = 0)
      : alloc_space_(space), alloc_stream_(stream) {}

  /**
   * Construct a dynamic tensor from a raw pointer, shape, and strides.
   * The caller is responsible for the lifetime of the data pointer.
   */
  dynamic_tensor_t(T *data, const std::vector<index_t> &shape, const std::vector<index_t> &strides)
      : ldata_(data),
        rank_(static_cast<int>(shape.size()))
  {
    MATX_ASSERT_STR(rank_ <= MATX_MAX_DYNAMIC_RANK, matxInvalidParameter,
                    "Dynamic tensor rank exceeds MATX_MAX_DYNAMIC_RANK");
    MATX_ASSERT_STR(shape.size() == strides.size(), matxInvalidParameter,
                    "Shape and strides must have the same length");
    for (int i = 0; i < rank_; i++) { shape_[i] = shape[i]; strides_[i] = strides[i]; }
  }

  /**
   * Construct a dynamic tensor from a raw pointer and shape (row-major strides).
   */
  dynamic_tensor_t(T *data, const std::vector<index_t> &shape)
      : ldata_(data),
        rank_(static_cast<int>(shape.size()))
  {
    MATX_ASSERT_STR(rank_ <= MATX_MAX_DYNAMIC_RANK, matxInvalidParameter,
                    "Dynamic tensor rank exceeds MATX_MAX_DYNAMIC_RANK");
    for (int i = 0; i < rank_; i++) shape_[i] = shape[i];
    compute_row_major_strides();
  }

  /**
   * Construct a dynamic tensor with owning storage, shape, and strides.
   */
  dynamic_tensor_t(Storage<T> storage, const std::vector<index_t> &shape, const std::vector<index_t> &strides)
      : ldata_(storage.data()),
        rank_(static_cast<int>(shape.size())),
        storage_(std::move(storage))
  {
    MATX_ASSERT_STR(rank_ <= MATX_MAX_DYNAMIC_RANK, matxInvalidParameter,
                    "Dynamic tensor rank exceeds MATX_MAX_DYNAMIC_RANK");
    MATX_ASSERT_STR(shape.size() == strides.size(), matxInvalidParameter,
                    "Shape and strides must have the same length");
    for (int i = 0; i < rank_; i++) { shape_[i] = shape[i]; strides_[i] = strides[i]; }
  }

  /**
   * Construct a dynamic tensor with owning storage and shape (row-major strides).
   */
  dynamic_tensor_t(Storage<T> storage, const std::vector<index_t> &shape)
      : ldata_(storage.data()),
        rank_(static_cast<int>(shape.size())),
        storage_(std::move(storage))
  {
    MATX_ASSERT_STR(rank_ <= MATX_MAX_DYNAMIC_RANK, matxInvalidParameter,
                    "Dynamic tensor rank exceeds MATX_MAX_DYNAMIC_RANK");
    for (int i = 0; i < rank_; i++) shape_[i] = shape[i];
    compute_row_major_strides();
  }

  // Returns MATX_MAX_DYNAMIC_RANK so that all existing operator code compiles
  // without modification (arrays are oversized but valid, if constexpr guards
  // enter the correct branch).  Use DynRank() for the actual runtime rank.
  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return MATX_MAX_DYNAMIC_RANK; }

  // Runtime rank -- use this to get the actual rank value.
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ int32_t DynRank() const { return rank_; }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const { return shape_[dim]; }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Stride(int dim) const { return strides_[dim]; }

  __MATX_INLINE__ T *Data() const { return ldata_; }

  __MATX_INLINE__ index_t TotalSize() const {
    index_t total = 1;
    for (int i = 0; i < rank_; ++i) {
      total *= shape_[i];
    }
    return total;
  }

  __MATX_INLINE__ const std::string str() const {
    return std::string("dynamic_tensor_") + std::to_string(rank_) + "_" + detail::to_short_str<T>();
  }

  __MATX_INLINE__ matxMemorySpace_t AllocationSpace() const { return alloc_space_; }
  __MATX_INLINE__ cudaStream_t AllocationStream() const { return alloc_stream_; }
  __MATX_INLINE__ void SetAllocationParams(matxMemorySpace_t space, cudaStream_t stream) {
    alloc_space_ = space;
    alloc_stream_ = stream;
  }

  // ---------- Lazy assignment operators ----------

  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const T2 &op) {
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, op_base);
  }

  // Compound assignment operators (+=, -=, *=, /=) are not yet supported
  // for dynamic tensors. Use explicit expressions like (a = a + b) instead.

  // ---------- JIT support ----------
#ifdef MATX_EN_JIT
  struct JIT_Storage {
    T *ldata_;
  };

  __MATX_INLINE__ JIT_Storage ToJITStorage() const {
    return JIT_Storage{ldata_};
  }

  __MATX_INLINE__ std::string get_jit_class_name() const {
    return detail::make_jit_tensor_class_name<T>(rank_, shape_, strides_);
  }

  __MATX_INLINE__ auto get_jit_op_str() const {
    const std::string class_name = get_jit_class_name();
    return detail::make_jit_tensor_op_str<T, stride_type>(
      class_name, rank_,
      dims_to_string(shape_, rank_),
      dims_to_string(strides_, rank_));
  }
#endif

  template <detail::OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
    if constexpr (Cap == detail::OperatorCapability::ELEMENTS_PER_THREAD) {
      // For dynamic tensors, conservatively report EPT ONE..MAX and let JIT dispatch refine.
      // We don't have compile-time rank to check stride contiguity, but we can check at runtime.
      if (rank_ == 0) {
        return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
      }
      if (strides_[rank_ - 1] != 1) {
        return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
      }
      // Check contiguity
      bool contiguous = true;
      index_t expected = 1;
      for (int i = rank_ - 1; i >= 0; --i) {
        if (strides_[i] != expected) { contiguous = false; break; }
        expected *= shape_[i];
      }
      if (!contiguous) {
        return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
      }
      if constexpr (sizeof(T) != detail::alignment_by_type<T>()) {
        return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
      } else {
        const int type_width_bytes = static_cast<int>(sizeof(T));
        int width = in.jit ? 32 : MAX_VEC_WIDTH_BYTES / type_width_bytes;
        index_t lsize = shape_[rank_ - 1];
        while (width > 1) {
          if (((lsize % width) == 0) &&
              ((reinterpret_cast<uintptr_t>(ldata_) % (sizeof(T) * width)) == 0)) {
            break;
          }
          width /= 2;
        }
        return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, static_cast<detail::ElementsPerThread>(width)};
      }
    }
    else if constexpr (Cap == detail::OperatorCapability::MAX_EPT_VEC_LOAD) {
      const int type_width_bytes = static_cast<int>(sizeof(T));
      int vec = MAX_VEC_WIDTH_BYTES / type_width_bytes;
      int power = 1;
      while (power * 2 <= vec) { power *= 2; }
      return power;
    }
    else if constexpr (Cap == detail::OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
      return get_jit_class_name();
#else
      return std::string("");
#endif
    }
    else if constexpr (Cap == detail::OperatorCapability::SUPPORTS_JIT) {
#ifdef MATX_EN_JIT
      return true;
#else
      return false;
#endif
    }
    else if constexpr (Cap == detail::OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
      const auto [key, value] = get_jit_op_str();
      if (in.find(key) == in.end()) {
        in[key] = value;
      }
      return true;
#else
      return false;
#endif
    }
    else if constexpr (Cap == detail::OperatorCapability::GLOBAL_KERNEL) {
      return true;
    }
    else if constexpr (Cap == detail::OperatorCapability::PASS_THROUGH_THREADS) {
      return false;
    }
    else if constexpr (Cap == detail::OperatorCapability::DYN_SHM_SIZE) {
      return 0;
    }
    else if constexpr (Cap == detail::OperatorCapability::ELEMENT_WISE) {
      return true;
    }
    else if constexpr (Cap == detail::OperatorCapability::ALIASED_MEMORY) {
      return false;
    }
    else {
      return detail::capability_attributes<Cap>::default_value;
    }
  }

private:
  std::string dims_to_string(const index_t *arr, int n) const {
    std::string s;
    for (int i = 0; i < n; ++i) {
      if (i != 0) s += ", ";
      s += std::to_string(arr[i]);
    }
    return s;
  }

  void compute_row_major_strides() {
    if (rank_ > 0) {
      strides_[rank_ - 1] = 1;
      for (int i = rank_ - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
      }
    }
  }

  T *ldata_ = nullptr;
  int rank_ = 0;
  index_t shape_[MATX_MAX_DYNAMIC_RANK] = {};
  index_t strides_[MATX_MAX_DYNAMIC_RANK] = {};
  Storage<T> storage_; // Optional owning storage (empty if non-owning)
  matxMemorySpace_t alloc_space_ = MATX_MANAGED_MEMORY;
  cudaStream_t alloc_stream_ = 0;
};

} // namespace detail


/**
 * @brief Create a dynamic-rank tensor with runtime rank to be assigned later.
 *
 * The tensor is intentionally rankless until make_tensor(tensor, shape, ...) is
 * called. This keeps rank selection entirely runtime-driven.
 */
template <typename T>
auto make_tensor(matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                 cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_LOG_DEBUG("make_tensor<dynamic_tensor_t<T>>(space, stream): space={}, stream={}",
                 static_cast<int>(space), reinterpret_cast<void *>(stream));
  return detail::dynamic_tensor_t<T>(space, stream);
}

/**
 * @brief Initialize a dynamic-rank tensor with owning storage and runtime shape.
 *
 * This is the dynamic-rank equivalent of "placement" make_tensor.
 */
template <typename TensorType>
  requires is_dynamic_tensor_v<TensorType>
void make_tensor(TensorType &tensor, const std::vector<index_t> &shape,
                 matxMemorySpace_t space, cudaStream_t stream)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(shape.size() <= MATX_MAX_DYNAMIC_RANK, matxInvalidParameter,
                  "Dynamic tensor rank exceeds MATX_MAX_DYNAMIC_RANK");
  auto storage =
      make_owning_storage<typename TensorType::value_type>(detail::total_size_from_shape(shape), space, stream);
  tensor = detail::dynamic_tensor_t<typename TensorType::value_type>(std::move(storage), shape);
  tensor.SetAllocationParams(space, stream);
}

template <typename TensorType>
  requires is_dynamic_tensor_v<TensorType>
void make_tensor(TensorType &tensor, const std::vector<index_t> &shape)
{
  make_tensor(tensor, shape, tensor.AllocationSpace(), tensor.AllocationStream());
}

template <typename TensorType>
  requires is_dynamic_tensor_v<TensorType>
void make_tensor(TensorType &tensor, std::initializer_list<index_t> shape,
                 matxMemorySpace_t space, cudaStream_t stream)
{
  make_tensor(tensor, std::vector<index_t>(shape), space, stream);
}

template <typename TensorType>
  requires is_dynamic_tensor_v<TensorType>
void make_tensor(TensorType &tensor, std::initializer_list<index_t> shape)
{
  make_tensor(tensor, std::vector<index_t>(shape));
}

template <typename TensorType, typename ShapeType>
  requires (is_dynamic_tensor_v<TensorType> && is_tuple_c<remove_cvref_t<ShapeType>>)
void make_tensor(TensorType &tensor, ShapeType &&shape,
                 matxMemorySpace_t space, cudaStream_t stream)
{
  auto shape_vec = detail::shape_to_vector(std::forward<ShapeType>(shape));
  make_tensor(tensor, shape_vec, space, stream);
}

template <typename TensorType, typename ShapeType>
  requires (is_dynamic_tensor_v<TensorType> && is_tuple_c<remove_cvref_t<ShapeType>>)
void make_tensor(TensorType &tensor, ShapeType &&shape)
{
  auto shape_vec = detail::shape_to_vector(std::forward<ShapeType>(shape));
  make_tensor(tensor, shape_vec);
}

/**
 * @brief Initialize a dynamic-rank tensor as a non-owning view from pointer+shape.
 */
template <typename TensorType>
  requires is_dynamic_tensor_v<TensorType>
void make_tensor(TensorType &tensor, typename TensorType::value_type *data,
                 const std::vector<index_t> &shape)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_ASSERT_STR(shape.size() <= MATX_MAX_DYNAMIC_RANK, matxInvalidParameter,
                  "Dynamic tensor rank exceeds MATX_MAX_DYNAMIC_RANK");
  tensor = detail::dynamic_tensor_t<typename TensorType::value_type>(data, shape);
}

template <typename TensorType>
  requires is_dynamic_tensor_v<TensorType>
void make_tensor(TensorType &tensor, typename TensorType::value_type *data,
                 std::initializer_list<index_t> shape)
{
  make_tensor(tensor, data, std::vector<index_t>(shape));
}

template <typename TensorType, typename ShapeType>
  requires (is_dynamic_tensor_v<TensorType> && is_tuple_c<remove_cvref_t<ShapeType>>)
void make_tensor(TensorType &tensor, typename TensorType::value_type *data,
                 ShapeType &&shape)
{
  auto shape_vec = detail::shape_to_vector(std::forward<ShapeType>(shape));
  make_tensor(tensor, data, shape_vec);
}

} // namespace matx
