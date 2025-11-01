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
  requires is_tensor<TensorType>
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
  requires (is_tensor<TensorType> && !std::is_array_v<remove_cvref_t<ShapeType>>)
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
  requires is_tensor<TensorType>
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
  requires is_tensor<TensorType>
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
  requires (!is_matx_descriptor<ShapeType> && !std::is_array_v<remove_cvref_t<ShapeType>>)
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
  requires is_tensor<TensorType>
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
  cuda::std::array<index_t, 0> shape;
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
  requires is_tensor<TensorType>
auto make_tensor( TensorType &tensor,
                  typename TensorType::value_type *ptr) {
  MATX_LOG_DEBUG("make_tensor(tensor&, ptr, 0D): ptr={}", reinterpret_cast<void*>(ptr));
  auto tmp = make_tensor<typename TensorType::value_type>(ptr, false);
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
  requires (!is_matx_descriptor<ShapeType> && !std::is_array_v<remove_cvref_t<ShapeType>>)
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
  requires is_tensor<TensorType>
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
  requires (is_tensor<TensorType> &&
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
  requires is_tensor<TensorType>
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
  requires (is_tensor<TensorType> && is_matx_descriptor<typename TensorType::desc_type>)
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
  requires is_tensor<TensorType>
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
 * Create a static-sized tensor with implicit memory
 * @returns New tensor
 **/
template <typename T, index_t I, index_t ...Is>
auto make_static_tensor() {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_static_tensor<T,I,Is...>()");

  static_tensor_desc_t<I, Is...> desc{};
  auto storage = make_owning_storage<T>(desc.TotalSize());
  return tensor_t<T, desc.Rank(), decltype(desc)>{std::move(storage), std::move(desc)};
}

template <typename TensorType>
  requires is_tensor<TensorType>
auto make_tensor( TensorType &tensor,
                  const DLManagedTensor dlp_tensor) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  MATX_LOG_DEBUG("make_tensor(tensor&, DLManagedTensor): ptr={}", dlp_tensor.dl_tensor.data);

  using T = typename TensorType::value_type;
  const DLTensor &dt = dlp_tensor.dl_tensor;

  // MatX doesn't track the memory type or device ID, so we don't need to copy it
  MATX_ASSERT_STR_EXP(dt.ndim, TensorType::Rank(), matxInvalidDim, "DLPack rank doesn't match MatX rank!");

  switch (dt.dtype.code) {
    case kDLComplex: {
      switch (dt.dtype.bits) {
        case 128: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, cuda::std::complex<double>>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 64: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, cuda::std::complex<float>>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 32: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, matxFp16Complex> || std::is_same_v<T, matxBf16Complex>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
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
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, double>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 32: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, float>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 16: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, matxFp16> || std::is_same_v<T, matxBf16>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        default:
          MATX_THROW(matxInvalidSize, "Invalid float size from DLPack");
      }
      break;
    }
    case kDLInt: {
      switch (dt.dtype.bits) {
        case 64: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, int64_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 32: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, int32_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 16: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, int16_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 8: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, int8_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
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
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, uint64_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 32: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, uint32_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 16: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, uint16_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        case 8: {
          [[maybe_unused]] constexpr bool same = std::is_same_v<T, uint8_t>;
          MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
          break;
        }
        default:
          MATX_THROW(matxInvalidSize, "Invalid unsigned integer size from DLPack");
      }
      break;
    }
    case kDLBool: {
      [[maybe_unused]] constexpr bool same = std::is_same_v<T, bool>;
      MATX_ASSERT_STR(same, matxInvalidType, "DLPack/MatX type mismatch");
      break;
    }
  }

  index_t strides[TensorType::Rank()];
  index_t shape[TensorType::Rank()];

  for (int r = 0; r < TensorType::Rank(); r++) {
    strides[r] = dt.strides[r];
    shape[r]   = dt.shape[r];
  }

  auto tmp = make_tensor<typename TensorType::value_type, TensorType::Rank()>(
          reinterpret_cast<typename TensorType::value_type*>(dt.data), shape, strides, false);
  tensor.Shallow(tmp);
}

} // namespace matx
