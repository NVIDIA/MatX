////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "matx/core/nvtx.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_desc.h"
#include "matx/core/dlpack.h"
#include "matx/core/log.h"
namespace matx {

/**
 * Create a tensor with a C array for the shape using implicitly-allocated memory
 *
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is manged memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 **/
template <typename T, int RANK>
auto make_tensor( const index_t (&shape)[RANK],
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  for (int i = 0; i < RANK; i++) {
    if (i > 0) shape_str += ",";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  MATX_LOG_DEBUG("make_tensor<T,RANK>(shape, space, stream): shape={}, space={}, stream={}", 
                 shape_str, static_cast<int>(space), reinterpret_cast<void*>(stream));

  DefaultDescriptor<RANK> desc{shape};
  auto storage = make_owning_storage<T>(desc.TotalSize(), space, stream);
  return tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor from existing storage and a shape specification
 *
 * @param storage Storage object containing the data
 * @param shape Shape specification for the tensor
 * @returns New tensor
 **/
template <typename T, typename ShapeType>
  requires (!is_matx_descriptor<ShapeType> && !std::is_array_v<remove_cvref_t<ShapeType>>)
auto make_tensor(Storage<T> storage, ShapeType &&shape) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor<T,ShapeType>(storage, shape): ptr={}", reinterpret_cast<const void*>(storage.data()));

  constexpr int RANK = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK> desc{std::forward<ShapeType>(shape)};
  return tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor with a C array for the shape using implicitly-allocated memory
 *
 * @param tensor Tensor object to store newly-created tensor into
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is manged memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
void make_tensor( TensorType &tensor,
                  const index_t (&shape)[TensorType::Rank()],
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  for (int i = 0; i < TensorType::Rank(); i++) {
    if (i > 0) shape_str += ",";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  MATX_LOG_DEBUG("make_tensor(tensor&, shape, space, stream): shape={}, space={}, stream={}", 
                 shape_str, static_cast<int>(space), reinterpret_cast<void*>(stream));

  auto tmp = make_tensor<typename TensorType::value_type, TensorType::Rank()>(shape, space, stream);
  tensor.Shallow(tmp);
}

/**
 * Create a tensor with a C array for the shape using implicitly-allocated memory.
 * Caller is responsible for deleting the tensor.
 *
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns Pointer to new tensor
 **/
template <typename T, int RANK>
auto make_tensor_p( const index_t (&shape)[RANK],
                    matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                    cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  for (int i = 0; i < RANK; i++) {
    if (i > 0) shape_str += ",";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  MATX_LOG_DEBUG("make_tensor_p<T,RANK>(shape, space, stream): shape={}, space={}, stream={}", 
                 shape_str, static_cast<int>(space), reinterpret_cast<void*>(stream));

  DefaultDescriptor<RANK> desc{shape};
  auto storage = make_owning_storage<T>(desc.TotalSize(), space, stream);
  return new tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor from a conforming container type
 *
 * Conforming containers have sequential iterators defined (both const and non-const). cuda::std::array
 * and std::vector meet this criteria.
 *
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename T, typename ShapeType>
  requires (!is_matx_shape<ShapeType> &&
            !is_matx_descriptor<ShapeType> &&
            !std::is_array_v<remove_cvref_t<ShapeType>>)
auto make_tensor( ShapeType &&shape,
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor<T,ShapeType>(shape, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));

  constexpr int rank = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<rank> desc{std::move(shape)};

  auto storage = make_owning_storage<T>(desc.TotalSize(), space, stream);

  return tensor_t<T,
    cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value,
    decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor from a conforming container type
 *
 * Conforming containers have sequential iterators defined (both const and non-const). cuda::std::array
 * and std::vector meet this criteria.
 *
 * @param tensor Tensor object to store newly-created tensor into
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename TensorType, typename ShapeType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType> && !std::is_array_v<remove_cvref_t<ShapeType>>)
auto make_tensor( TensorType &tensor,
                  ShapeType &&shape,
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor(tensor&, shape, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));

  auto tmp = make_tensor<typename TensorType::value_type, ShapeType>(std::forward<ShapeType>(shape), space, stream);
  tensor.Shallow(tmp);
}

/**
 * Create a tensor from a conforming container type
 *
 * Conforming containers have sequential iterators defined (both const and non-const). cuda::std::array
 * and std::vector meet this criteria.  Caller is responsible for deleting tensor.
 *
 * @param shape  Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns Pointer to new tensor
 *
 **/
template <typename T, typename ShapeType>
  requires (!is_matx_shape<ShapeType> &&
            !std::is_array_v<remove_cvref_t<ShapeType>>)
auto make_tensor_p( ShapeType &&shape,
                    matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                    cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor_p<T,ShapeType>(shape, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));

  DefaultDescriptor<static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value)> desc{std::move(shape)};

  auto storage = make_owning_storage<T>(desc.TotalSize(), space, stream);
  return new tensor_t<T,
  cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value,
  decltype(desc)>{std::move(storage), std::move(desc)};
}


/**
 * Create a 0D tensor with implicitly-allocated memory.
 *
 * @param t Unused empty {} initializer
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename T>
auto make_tensor( [[maybe_unused]] const std::initializer_list<detail::no_size_t> t,
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_LOG_DEBUG("make_tensor<T>(0D, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));
  using shape_t = cuda::std::array<index_t, 0>;
  return make_tensor<T, shape_t>(shape_t{}, space, stream);
}

/**
 * Create a 0D tensor with implicitly-allocated memory.
 *
 * @param tensor Tensor object to store newly-created tensor into
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_LOG_DEBUG("make_tensor(tensor&, 0D, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));
  auto tmp = make_tensor<typename TensorType::value_type>({}, space, stream);
  tensor.Shallow(tmp);
}

/**
 * Create a 0D tensor with user-defined memory.
 *
 * @param t Unused empty {} initializer
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename T>
auto make_tensor_p( [[maybe_unused]] const std::initializer_list<detail::no_size_t> t,
                    matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                    cudaStream_t stream = 0) {
  MATX_LOG_DEBUG("make_tensor_p<T>(0D, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));

  cuda::std::array<index_t, 0> shape;
  return make_tensor_p<T, decltype(shape)>(std::move(shape), space, stream);
}

/**
 * Create a tensor with user-defined memory and a C array
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param owning
 *   If this class owns memory of data
 * @returns New tensor
 **/
template <typename T, int RANK>
auto make_tensor( T *data,
                  const index_t (&shape)[RANK],
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  for (int i = 0; i < RANK; i++) {
    if (i > 0) shape_str += ",";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  MATX_LOG_DEBUG("make_tensor<T,RANK>(data, shape, owning): ptr={}, shape={}, owning={}", 
                 reinterpret_cast<void*>(data), shape_str, owning);

  DefaultDescriptor<RANK> desc{shape};
  auto storage = owning ? make_owning_storage<T>(desc.TotalSize()) : make_non_owning_storage<T>(data, desc.TotalSize());
  return tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor with user-defined memory and a C array
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @returns New tensor
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  typename TensorType::value_type *data,
                  const index_t (&shape)[TensorType::Rank()]) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  for (int i = 0; i < TensorType::Rank(); i++) {
    if (i > 0) shape_str += ",";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  MATX_LOG_DEBUG("make_tensor(tensor&, data, shape): ptr={}, shape={}", 
                 reinterpret_cast<void*>(data), shape_str);

  auto tmp = make_tensor<typename TensorType::value_type, TensorType::Rank()>(data, shape, false);
  tensor.Shallow(tmp);
}

/**
 * Create a tensor with user-defined memory and conforming shape type
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename T, typename ShapeType>
  requires (!is_matx_descriptor<ShapeType> &&
            !std::is_array_v<remove_cvref_t<ShapeType>> &&
            is_tuple_c<remove_cvref_t<ShapeType>>)
auto make_tensor( T *data,
                  ShapeType &&shape,
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor<T,ShapeType>(data, shape, owning): ptr={}, owning={}", 
                 reinterpret_cast<void*>(data), owning);

  constexpr int RANK = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK>
    desc{std::forward<ShapeType>(shape)};
  auto storage = owning ? make_owning_storage<T>(desc.TotalSize()) : make_non_owning_storage<T>(data, desc.TotalSize());
  return tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor with user-defined memory and conforming shape type
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @returns New tensor
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  typename TensorType::value_type *data,
                  typename TensorType::shape_container &&shape) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor(tensor&, data, shape): ptr={}", reinterpret_cast<void*>(data));
  
  auto tmp = make_tensor<typename TensorType::value_type, typename TensorType::shape_container>(data, std::forward<typename TensorType::shape_container>(shape), false);
  tensor.Shallow(tmp);
}

/**
 * Create a 0D tensor with user-defined memory
 *
 * @param ptr
 *  Pointer to data
 * @param t Unused empty {} initializer
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename T>
auto make_tensor( T *ptr,
                  [[maybe_unused]] const std::initializer_list<detail::no_size_t> t,
                  bool owning = false) {
  MATX_LOG_DEBUG("make_tensor<T>(ptr, 0D, owning): ptr={}, owning={}", 
                 reinterpret_cast<void*>(ptr), owning);
  cuda::std::array<index_t, 0> shape{};
  return make_tensor<T, decltype(shape)>(ptr, std::move(shape), owning);
}

/**
 * Create a 0D tensor with user-defined memory
 *
 * @param tensor
 *  Tensor object to store newly-created tensor into
 * @param ptr
 *  Pointer to data
 * @returns New tensor
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  typename TensorType::value_type *ptr) {
  MATX_LOG_DEBUG("make_tensor(tensor&, ptr, 0D): ptr={}", reinterpret_cast<void*>(ptr));
  auto tmp = make_tensor<typename TensorType::value_type>(ptr, {}, false);
  tensor.Shallow(tmp);
}



/**
 * Create a tensor with user-defined memory and conforming shape type
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
  * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename T, typename ShapeType>
  requires (!is_matx_descriptor<ShapeType> &&
            !std::is_array_v<remove_cvref_t<ShapeType>> &&
            is_tuple_c<remove_cvref_t<ShapeType>>)
auto make_tensor_p( T *const data,
                    ShapeType &&shape,
                    bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor_p<T,ShapeType>(data, shape, owning): ptr={}, owning={}", 
                 reinterpret_cast<const void*>(data), owning);

  constexpr int RANK = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK>
    desc{std::forward<ShapeType>(shape)};
  auto storage = owning ? make_owning_storage<T>(desc.TotalSize()) : make_non_owning_storage<T>(data, desc.TotalSize());
  return new tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor with custom allocator using C-array shape
 *
 * @param shape
 *   Shape of tensor as C-array
 * @param alloc
 *   Custom allocator (PMR allocator, custom allocator pointer, etc.)
 * @returns New tensor
 **/
template <typename T, int RANK, typename Allocator>
auto make_tensor( const index_t (&shape)[RANK],
                  Allocator&& alloc) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  for (int i = 0; i < RANK; i++) {
    if (i > 0) shape_str += ",";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  MATX_LOG_DEBUG("make_tensor<T,RANK,Allocator>(shape, alloc): shape={}", shape_str);

  DefaultDescriptor<RANK> desc{shape};
  auto storage = make_owning_storage<T>(desc.TotalSize(), std::forward<Allocator>(alloc));
  return tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor with custom allocator using conforming shape type
 *
 * @param shape
 *   Shape of tensor (tuple, array, etc.)
 * @param alloc
 *   Custom allocator (PMR allocator, custom allocator pointer, etc.)
 * @returns New tensor
 **/
template <typename T, typename ShapeType, typename Allocator>
  requires (!is_matx_shape<ShapeType> && !is_matx_descriptor<ShapeType> &&
            !std::is_array_v<remove_cvref_t<ShapeType>>)
auto make_tensor( ShapeType &&shape,
                  Allocator&& alloc) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor<T,ShapeType,Allocator>(shape, alloc)");

  constexpr int RANK = static_cast<int>(cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK> desc{std::forward<ShapeType>(shape)};
  auto storage = make_owning_storage<T>(desc.TotalSize(), std::forward<Allocator>(alloc));
  return tensor_t<T, RANK, decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Create a tensor with custom allocator using existing tensor reference
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param shape
 *   Shape of tensor as C-array
 * @param alloc
 *   Custom allocator (PMR allocator, custom allocator pointer, etc.)
 **/
template <typename TensorType, typename Allocator>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
void make_tensor( TensorType &tensor,
                  const index_t (&shape)[TensorType::Rank()],
                  Allocator&& alloc) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  for (int i = 0; i < TensorType::Rank(); i++) {
    if (i > 0) shape_str += ",";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  MATX_LOG_DEBUG("make_tensor(tensor&, shape, alloc): shape={}", shape_str);

  auto tmp = make_tensor<typename TensorType::value_type, TensorType::Rank()>(shape, std::forward<Allocator>(alloc));
  tensor.Shallow(tmp);
}

/**
 * Create a tensor with custom allocator using existing tensor reference and conforming shape
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param shape
 *   Shape of tensor (tuple, array, etc.)
 * @param alloc
 *   Custom allocator (PMR allocator, custom allocator pointer, etc.)
 **/
template <typename TensorType, typename ShapeType, typename Allocator>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType> &&
            !std::is_array_v<remove_cvref_t<ShapeType>>)
void make_tensor( TensorType &tensor,
                  ShapeType &&shape,
                  Allocator&& alloc) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor(tensor&, shape, alloc)");

  auto tmp = make_tensor<typename TensorType::value_type>(std::forward<ShapeType>(shape), std::forward<Allocator>(alloc));
  tensor.Shallow(tmp);
}


/**
 * Create a tensor with user-defined memory and an existing descriptor
 *
 * @param data
 *   Pointer to device data
 * @param desc
 *   Tensor descriptor (tensor_desc_t)
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename T, typename D>
  requires is_matx_descriptor<remove_cvref_t<D>>
auto make_tensor( T* const data,
                  D &&desc,
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor<T,D>(data, desc, owning): ptr={}, owning={}", 
                 reinterpret_cast<const void*>(data), owning);

  using Dstrip = typename remove_cvref<D>::type;
  auto storage = owning ? make_owning_storage<T>(desc.TotalSize()) : make_non_owning_storage<T>(data, desc.TotalSize());
  return tensor_t<T, Dstrip::Rank(), Dstrip>{std::move(storage), std::forward<D>(desc)};
}

/**
 * Create a tensor with user-defined memory and an existing descriptor
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param data
 *   Pointer to device data
 * @param desc
 *   Tensor descriptor (tensor_desc_t)
 * @returns New tensor
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  typename TensorType::value_type* const data,
                  typename TensorType::desc_type &&desc) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor(tensor&, data, desc): ptr={}", reinterpret_cast<const void*>(data));

  // This tensor should be non-owning regardless of the original ownership since it will go out of scope at the end of the function
  auto tmp = make_tensor<typename TensorType::value_type, typename TensorType::desc_type>(data, std::forward<typename TensorType::desc_type>(desc), false);
  tensor.Shallow(tmp);
}

/**
 * Create a tensor with implicitly-allocated memory and an existing descriptor
 *
 * @param desc Tensor descriptor (tensor_desc_t)
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 **/
template <typename T, typename D>
  requires is_matx_descriptor<remove_cvref_t<D>>
auto make_tensor( D &&desc,
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor<T,D>(desc, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));

  using Dstrip = typename remove_cvref<D>::type;

  auto storage = make_owning_storage<T>(desc.TotalSize(), space, stream);
  return tensor_t<T, Dstrip::Rank(), Dstrip>{std::move(storage), std::forward<D>(desc)};
}

/**
 * Create a tensor with implicitly-allocated memory and an existing descriptor
 *
 * @param tensor Tensor object to store newly-created tensor into
 * @param desc Tensor descriptor (tensor_desc_t)
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType> && is_matx_descriptor<typename TensorType::desc_type>)
auto make_tensor( TensorType &&tensor,
                  typename TensorType::desc_type &&desc,
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY,
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor(tensor&&, desc, space, stream): space={}, stream={}", 
                 static_cast<int>(space), reinterpret_cast<void*>(stream));

  auto tmp = make_tensor<typename TensorType::value_type, typename TensorType::desc_type>(std::forward<typename TensorType::desc_type>(desc), space, stream);
  tensor.Shallow(tmp);
}

/**
 * Create a tensor with user-defined memory and C-array shapes and strides
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param strides
 *   Strides of tensor
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename T, int RANK>
auto make_tensor( T *const data,
                  const index_t (&shape)[RANK],
                  const index_t (&strides)[RANK],
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  std::string strides_str = "[";
  for (int i = 0; i < RANK; i++) {
    if (i > 0) { shape_str += ","; strides_str += ","; }
    shape_str += std::to_string(shape[i]);
    strides_str += std::to_string(strides[i]);
  }
  shape_str += "]";
  strides_str += "]";
  MATX_LOG_DEBUG("make_tensor<T,RANK>(data, shape, strides, owning): ptr={}, shape={}, strides={}, owning={}", 
                 reinterpret_cast<const void*>(data), shape_str, strides_str, owning);

  DefaultDescriptor<RANK>  desc{shape, strides};
  auto storage = owning ? make_owning_storage<T>(desc.TotalSize()) : make_non_owning_storage<T>(data, desc.TotalSize());
  return tensor_t<T,RANK, decltype(desc)>{std::move(storage), std::move(desc), data};
}

/**
 * Create a tensor with user-defined memory and C-array shapes and strides
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param strides
 *   Strides of tensor
 * @returns New tensor
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  typename TensorType::value_type *const data,
                  const index_t (&shape)[TensorType::Rank()],
                  const index_t (&strides)[TensorType::Rank()]) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  std::string shape_str = "[";
  std::string strides_str = "[";
  for (int i = 0; i < TensorType::Rank(); i++) {
    if (i > 0) { shape_str += ","; strides_str += ","; }
    shape_str += std::to_string(shape[i]);
    strides_str += std::to_string(strides[i]);
  }
  shape_str += "]";
  strides_str += "]";
  MATX_LOG_DEBUG("make_tensor(tensor&, data, shape, strides): ptr={}, shape={}, strides={}", 
                 reinterpret_cast<const void*>(data), shape_str, strides_str);

  auto tmp = make_tensor<typename TensorType::value_type, TensorType::Rank()>(data, shape, strides, false);
  tensor.Shallow(tmp);
}


/**
 * Create a static-sized tensor with implicit memory using compile-time shape
 * template parameters.
 *
 * Example: make_tensor<float, 10, 20>()
 *
 * @returns New tensor
 **/
template <typename T, index_t I, index_t ...Is>
auto make_tensor() {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor<T,I,Is...>()");

  static_tensor_desc_t<I, Is...> desc{};
  auto storage = make_owning_storage<T>(desc.TotalSize());
  return tensor_t<T, desc.Rank(), decltype(desc)>{std::move(storage), std::move(desc)};
}

/**
 * Backward-compatible wrapper for static-sized tensors.
 *
 * Prefer make_tensor<T, I, Is...>() for new code.
 */
template <typename T, index_t I, index_t ...Is>
[[deprecated("make_static_tensor is deprecated; use make_tensor<T, I, Is...>() instead.")]]
auto make_static_tensor() {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  MATX_LOG_DEBUG("make_static_tensor<T,I,Is...>() -> make_tensor<T,I,Is...>()");
  return make_tensor<T, I, Is...>();
}

namespace detail {
inline void validate_dlpack_condition(bool condition, matxError_t error, const char *message)
{
  if (!condition) {
    MATX_THROW(error, message);
  }
}

template <bool Same>
void validate_dlpack_dtype(const char *message)
{
  if constexpr (!Same) {
    MATX_THROW(matxInvalidType, message);
  }
}

template <typename T, index_t Rank>
void validate_dlpack_tensor_type(const DLTensor &dt) {
  using BaseT = std::remove_cv_t<T>;
  using LaneInfo = detail::DLPackLaneInfo<BaseT>;
  using ScalarT = typename LaneInfo::scalar_type;
  constexpr uint16_t lanes = LaneInfo::lanes;

  // MatX doesn't track the memory type or device ID, so we don't need to copy it
  validate_dlpack_condition(dt.ndim == Rank, matxInvalidDim, "DLPack rank doesn't match MatX rank");

  validate_dlpack_condition(
      dt.dtype.lanes == lanes, matxInvalidType,
      "DLPack vector lane mismatch: dtype.lanes must match MatX value_type lane width");

  switch (dt.dtype.code) {
    case kDLComplex: {
      switch (dt.dtype.bits) {
        case 128: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, cuda::std::complex<double>>>(
              "DLPack dtype mismatch: code=kDLComplex bits=128 requires MatX base scalar type cuda::std::complex<double>");
          break;
        }
        case 64: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, cuda::std::complex<float>>>(
              "DLPack dtype mismatch: code=kDLComplex bits=64 requires MatX base scalar type cuda::std::complex<float>");
          break;
        }
        case 32: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, matxFp16Complex>>(
              "DLPack dtype mismatch: code=kDLComplex bits=32 requires MatX base scalar type matxFp16Complex");
          break;
        }
        default:
          MATX_THROW(matxInvalidSize, "Invalid complex float size from DLPack");
      }
      break;
    }

    case kDLFloat: {
      switch (dt.dtype.bits) {
        case 64: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, double>>(
              "DLPack dtype mismatch: code=kDLFloat bits=64 requires MatX base scalar type double");
          break;
        }
        case 32: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, float>>(
              "DLPack dtype mismatch: code=kDLFloat bits=32 requires MatX base scalar type float");
          break;
        }
        case 16: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, matxFp16>>(
              "DLPack dtype mismatch: code=kDLFloat bits=16 requires MatX base scalar type matxFp16");
          break;
        }
        default:
          MATX_THROW(matxInvalidSize, "Invalid float size from DLPack");
      }
      break;
    }
    case kDLBfloat: {
      switch (dt.dtype.bits) {
        case 16: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, matxBf16>>(
              "DLPack dtype mismatch: code=kDLBfloat bits=16 requires MatX base scalar type matxBf16");
          break;
        }
        default:
          MATX_THROW(matxInvalidSize, "Invalid bfloat size from DLPack");
      }
      break;
    }
    case kDLInt: {
      switch (dt.dtype.bits) {
        case 64: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, int64_t>>(
              "DLPack dtype mismatch: code=kDLInt bits=64 requires MatX base scalar type int64_t");
          break;
        }
        case 32: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, int32_t>>(
              "DLPack dtype mismatch: code=kDLInt bits=32 requires MatX base scalar type int32_t");
          break;
        }
        case 16: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, int16_t>>(
              "DLPack dtype mismatch: code=kDLInt bits=16 requires MatX base scalar type int16_t");
          break;
        }
        case 8: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, int8_t>>(
              "DLPack dtype mismatch: code=kDLInt bits=8 requires MatX base scalar type int8_t");
          break;
        }
        default:
          MATX_THROW(matxInvalidSize, "Invalid signed integer size from DLPack");
      }
      break;
    }
    case kDLUInt: {
      switch (dt.dtype.bits) {
        case 64: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, uint64_t>>(
              "DLPack dtype mismatch: code=kDLUInt bits=64 requires MatX base scalar type uint64_t");
          break;
        }
        case 32: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, uint32_t>>(
              "DLPack dtype mismatch: code=kDLUInt bits=32 requires MatX base scalar type uint32_t");
          break;
        }
        case 16: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, uint16_t>>(
              "DLPack dtype mismatch: code=kDLUInt bits=16 requires MatX base scalar type uint16_t");
          break;
        }
        case 8: {
          validate_dlpack_dtype<std::is_same_v<ScalarT, uint8_t>>(
              "DLPack dtype mismatch: code=kDLUInt bits=8 requires MatX base scalar type uint8_t");
          break;
        }
        default:
          MATX_THROW(matxInvalidSize, "Invalid unsigned integer size from DLPack");
      }
      break;
    }
    case kDLBool: {
      validate_dlpack_dtype<std::is_same_v<ScalarT, bool>>(
          "DLPack dtype mismatch: code=kDLBool requires MatX base scalar type bool");
      break;
    }
    default:
      MATX_THROW(matxInvalidType, "Unsupported DLPack data type code");
  }
}

template <typename T>
T *dlpack_data_pointer(const DLTensor &dt)
{
  using BaseT = std::remove_cv_t<T>;

  validate_dlpack_condition(dt.data != nullptr, matxInvalidParameter,
                            "DLPack data cannot be null for MatX tensors");
  validate_dlpack_condition(dt.byte_offset % sizeof(BaseT) == 0, matxInvalidType,
                            "DLPack byte_offset must align with element type size");
  validate_dlpack_condition(dt.byte_offset <= static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()),
                            matxInvalidSize, "DLPack byte_offset is too large");

  auto *base = reinterpret_cast<uint8_t *>(dt.data);
  return reinterpret_cast<T *>(base + static_cast<std::ptrdiff_t>(dt.byte_offset));
}

template <typename T>
constexpr uint64_t dlpack_max_addressable_elements()
{
  const auto index_max = static_cast<uint64_t>(std::numeric_limits<index_t>::max());
  const auto storage_max = static_cast<uint64_t>(
      std::numeric_limits<size_t>::max() / sizeof(std::remove_cv_t<T>));
  return index_max < storage_max ? index_max : storage_max;
}

template <typename TensorType>
  requires is_tensor<TensorType>
size_t dlpack_shape_and_strides(const DLTensor &dt,
                                index_t (&shape)[TensorType::Rank()],
                                index_t (&strides)[TensorType::Rank()]) {
  // DLPack does not expose the backing allocation length. These checks validate
  // producer metadata and MatX address arithmetic before the borrowed view is used.
  using T = typename TensorType::value_type;
  constexpr auto max_index = static_cast<uint64_t>(std::numeric_limits<index_t>::max());
  constexpr auto max_addressable = dlpack_max_addressable_elements<T>();
  uint64_t logical_elements = 1;

  if constexpr (TensorType::Rank() > 0) {
    validate_dlpack_condition(dt.shape != nullptr, matxInvalidParameter,
                              "DLPack shape cannot be null for non-scalar tensors");
  }

  for (int r = 0; r < TensorType::Rank(); r++) {
    const int64_t dim = dt.shape[r];
    validate_dlpack_condition(dim > 0, matxInvalidSize,
                              "DLPack shape dimensions must be positive for MatX tensors");
    validate_dlpack_condition(static_cast<uint64_t>(dim) <= max_index, matxInvalidSize,
                              "DLPack shape dimension exceeds MatX index range");
    validate_dlpack_condition(logical_elements <= max_addressable / static_cast<uint64_t>(dim), matxInvalidSize,
                              "DLPack tensor shape is too large");

    shape[r] = static_cast<index_t>(dim);
    logical_elements *= static_cast<uint64_t>(dim);
  }

  if (dt.strides != nullptr) {
    for (int r = 0; r < TensorType::Rank(); r++) {
      const int64_t stride = dt.strides[r];
      validate_dlpack_condition(stride >= 0, matxInvalidParameter,
                                "DLPack negative strides are not supported by MatX import");
      validate_dlpack_condition(static_cast<uint64_t>(stride) <= max_index, matxInvalidSize,
                                "DLPack stride exceeds MatX index range");
      strides[r] = static_cast<index_t>(stride);
    }
  }
  else {
    // Older DLPack producers may use null strides to indicate contiguous layout.
    if constexpr (TensorType::Rank() > 0) {
      strides[TensorType::Rank() - 1] = 1;
      for (int r = TensorType::Rank() - 2; r >= 0; r--) {
        const auto next_stride = static_cast<uint64_t>(strides[r + 1]);
        const auto next_shape = static_cast<uint64_t>(shape[r + 1]);
        validate_dlpack_condition(next_stride <= max_index / next_shape, matxInvalidSize,
                                  "DLPack contiguous stride calculation overflowed MatX index range");
        strides[r] = static_cast<index_t>(next_stride * next_shape);
      }
    }
  }

  uint64_t max_offset = 0;
  for (int r = 0; r < TensorType::Rank(); r++) {
    const auto dim_extent = static_cast<uint64_t>(shape[r] - 1);
    const auto stride = static_cast<uint64_t>(strides[r]);
    validate_dlpack_condition(dim_extent == 0 || stride <= max_addressable / dim_extent, matxInvalidSize,
                              "DLPack strided tensor span is too large");
    const auto dim_offset = dim_extent * stride;
    validate_dlpack_condition(max_offset <= max_addressable - dim_offset, matxInvalidSize,
                              "DLPack strided tensor span is too large");
    max_offset += dim_offset;
  }

  const auto storage_elements = max_offset + 1;
  validate_dlpack_condition(storage_elements <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                            matxInvalidSize, "DLPack tensor storage span exceeds host size range");
  return static_cast<size_t>(storage_elements);
}

template <typename T, int RANK>
auto make_dlpack_tensor_view(T *data_ptr,
                             const index_t (&shape)[RANK],
                             const index_t (&strides)[RANK],
                             size_t storage_elements) {
  DefaultDescriptor<RANK> desc{detail::to_array(shape), detail::to_array(strides)};
  auto storage = make_non_owning_storage<T>(data_ptr, storage_elements);
  return matx::tensor_t<T, RANK, decltype(desc)>{
      std::move(storage), std::move(desc), data_ptr};
}

template <typename T, int RANK, typename Owner>
auto make_dlpack_tensor_view(std::shared_ptr<Owner> owner,
                             T *data_ptr,
                             const index_t (&shape)[RANK],
                             const index_t (&strides)[RANK],
                             size_t storage_elements) {
  DefaultDescriptor<RANK> desc{detail::to_array(shape), detail::to_array(strides)};
  auto data = std::shared_ptr<T>(std::move(owner), data_ptr);
  auto storage = make_storage_from_shared_ptr<T>(std::move(data), storage_elements);
  return matx::tensor_t<T, RANK, decltype(desc)>{
      std::move(storage), std::move(desc), data_ptr};
}

template <typename TensorType>
void validate_dlpack_read_only_import(uint64_t flags) {
  if ((flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0U) {
    validate_dlpack_condition(std::is_const_v<typename TensorType::value_type>, matxInvalidType,
                              "Read-only DLPack tensors must be imported as const MatX tensors");
  }
}
} // namespace detail

/**
 * Create a tensor from a legacy DLPack managed tensor. This does not transfer ownership of the DLManagedTensor,
 * so the caller is responsible for calling the deleter method when the last MatX reference to the imported storage is
 * released.
 *
 * @deprecated Use `make_tensor(tensor, DLManagedTensor*)` to transfer ownership
 *   and guarantee source lifetime while MatX views are alive.
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param dlp_tensor
 *   Legacy DLPack tensor metadata and data pointer (borrowed, non-owning)
 **/
template <typename TensorType>
    requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
[[deprecated("Use make_tensor(tensor, DLManagedTensor*) to transfer ownership and ensure lifetime safety")]]
auto make_tensor( TensorType &tensor,
                  const DLManagedTensor dlp_tensor) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor(tensor&, DLManagedTensor): ptr={}", dlp_tensor.dl_tensor.data);

  using T = typename TensorType::value_type;
  const DLTensor &dt = dlp_tensor.dl_tensor;
  detail::validate_dlpack_tensor_type<T, TensorType::Rank()>(dt);

  index_t strides[TensorType::Rank()];
  index_t shape[TensorType::Rank()];
  const auto storage_elements = detail::dlpack_shape_and_strides<TensorType>(dt, shape, strides);
  auto *data_ptr = detail::dlpack_data_pointer<T>(dt);

  auto tmp = detail::make_dlpack_tensor_view<T, TensorType::Rank()>(
      data_ptr, shape, strides, storage_elements);
  tensor.Shallow(tmp);
}

/**
 * Create a tensor from a DLManagedTensor.
 *
 * This consumes `dlp_tensor`, the deleter method will be called when the last MatX reference to the imported storage is
 * released.
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param dlp_tensor
 *   Pointer to a heap-allocated `DLManagedTensor` whose ownership is
 *   transferred to MatX
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  DLManagedTensor *dlp_tensor) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  detail::validate_dlpack_condition(dlp_tensor != nullptr, matxInvalidParameter,
                                    "DLManagedTensor pointer cannot be null");

  auto owner = std::shared_ptr<DLManagedTensor>(dlp_tensor, [](DLManagedTensor *managed) {
    if (managed != nullptr && managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  });
  MATX_LOG_DEBUG("make_tensor(tensor&, DLManagedTensor*): ptr={}", owner->dl_tensor.data);

  using T = typename TensorType::value_type;
  const DLTensor &dt = owner->dl_tensor;
  detail::validate_dlpack_tensor_type<T, TensorType::Rank()>(dt);

  index_t strides[TensorType::Rank()];
  index_t shape[TensorType::Rank()];
  const auto storage_elements = detail::dlpack_shape_and_strides<TensorType>(dt, shape, strides);
  auto *data_ptr = detail::dlpack_data_pointer<T>(dt);

  auto tmp = detail::make_dlpack_tensor_view<T, TensorType::Rank()>(
      std::move(owner), data_ptr, shape, strides, storage_elements);

  tensor.Shallow(tmp);
}

/**
 * Create a tensor from a versioned DLPack managed tensor and transfer ownership.
 *
 * This consumes `dlp_tensor`, the deleter method will be called when the last MatX reference to the imported storage is
 * released.
 *
 * @param tensor
 *   Tensor object to store newly-created tensor into
 * @param dlp_tensor
 *   Pointer to a heap-allocated `DLManagedTensorVersioned` whose ownership is
 *   transferred to MatX
 **/
template <typename TensorType>
  requires (is_tensor<TensorType> && !is_dynamic_tensor_v<TensorType>)
auto make_tensor( TensorType &tensor,
                  DLManagedTensorVersioned *dlp_tensor) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  detail::validate_dlpack_condition(dlp_tensor != nullptr, matxInvalidParameter,
                                    "DLManagedTensorVersioned pointer cannot be null");

  auto owner = std::shared_ptr<DLManagedTensorVersioned>(dlp_tensor, [](DLManagedTensorVersioned *managed) {
    if (managed != nullptr && managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  });
  detail::validate_dlpack_condition(owner->version.major == DLPACK_MAJOR_VERSION, matxInvalidParameter,
                                    "Unsupported DLPack major version");
  MATX_LOG_DEBUG("make_tensor(tensor&, DLManagedTensorVersioned*): ptr={}", owner->dl_tensor.data);

  using T = typename TensorType::value_type;
  const DLTensor &dt = owner->dl_tensor;
  detail::validate_dlpack_tensor_type<T, TensorType::Rank()>(dt);
  detail::validate_dlpack_read_only_import<TensorType>(owner->flags);

  index_t strides[TensorType::Rank()];
  index_t shape[TensorType::Rank()];
  const auto storage_elements = detail::dlpack_shape_and_strides<TensorType>(dt, shape, strides);
  auto *data_ptr = detail::dlpack_data_pointer<T>(dt);

  auto tmp = detail::make_dlpack_tensor_view<T, TensorType::Rank()>(
      std::move(owner), data_ptr, shape, strides, storage_elements);

  tensor.Shallow(tmp);
}

} // namespace matx
