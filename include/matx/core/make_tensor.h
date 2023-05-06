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
  
  T *ptr;
  DefaultDescriptor<RANK> desc{shape};

  size_t size = desc.TotalSize() * sizeof(T);
  matxAlloc((void**)&ptr, desc.TotalSize() * sizeof(T), space, stream);

  raw_pointer_buffer<T, matx_allocator<T>> rp(ptr, size, true);
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with a C array for the shape using implicitly-allocated memory
 *
 * @param tensor Tensor object to store newly-created tensor into
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is manged memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 **/
template <typename TensorType, std::enable_if_t< is_tensor_view_v<TensorType>, bool> = true>
void make_tensor( TensorType &tensor, 
                  const index_t (&shape)[TensorType::Rank()], 
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  auto tmp = make_tensor<typename TensorType::scalar_type, TensorType::Rank()>(shape, space, stream);
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
  
  T *ptr;
  DefaultDescriptor<RANK> desc{shape};

  size_t size = desc.TotalSize() * sizeof(T);
  matxAlloc((void**)&ptr, desc.TotalSize() * sizeof(T), space, stream);

  raw_pointer_buffer<T, matx_allocator<T>> rp(ptr, size, true);
  basic_storage<decltype(rp)> s{std::move(rp)};
  return new tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor from a conforming container type
 *
 * Conforming containers have sequential iterators defined (both const and non-const). std::array
 * and std::vector meet this criteria.
 *
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename T, typename ShapeType,
  std::enable_if_t< !is_matx_shape_v<ShapeType> &&
                    !is_matx_descriptor_v<ShapeType> &&
                    !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor( ShapeType &&shape, 
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  T *ptr;
  constexpr int rank = static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<rank> desc{std::move(shape)};

  size_t size = desc.TotalSize() * sizeof(T);
  matxAlloc((void**)&ptr, desc.TotalSize() * sizeof(T), space, stream);

  raw_pointer_buffer<T, matx_allocator<T>> rp(ptr, size, true);
  basic_storage<decltype(rp)> s{std::move(rp)};

  return tensor_t<T,
    std::tuple_size<typename remove_cvref<ShapeType>::type>::value,
    decltype(s),
    decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor from a conforming container type
 *
 * Conforming containers have sequential iterators defined (both const and non-const). std::array
 * and std::vector meet this criteria.
 *
 * @param tensor Tensor object to store newly-created tensor into
 * @param shape Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename TensorType,
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  typename TensorType::shape_container &&shape, 
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::scalar_type, typename TensorType::shape_container>(std::forward<typename TensorType::shape_container>(shape), space, stream);
  tensor.Shallow(tmp);
}

/**
 * Create a tensor from a conforming container type
 *
 * Conforming containers have sequential iterators defined (both const and non-const). std::array
 * and std::vector meet this criteria.  Caller is responsible for deleting tensor.
 *
 * @param shape  Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns Pointer to new tensor
 *
 **/
template <typename T, typename ShapeType,
  std::enable_if_t< !is_matx_shape_v<ShapeType> &&
                    !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor_p( ShapeType &&shape, 
                    matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                    cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  T *ptr;
  DefaultDescriptor<static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value)> desc{std::move(shape)};

  size_t size = desc.TotalSize() * sizeof(T);
  matxAlloc((void**)&ptr, desc.TotalSize() * sizeof(T), space, stream);

  raw_pointer_buffer<T, matx_allocator<T>> rp(ptr, size, true);
  basic_storage<decltype(rp)> s{std::move(rp)};
  return new tensor_t<T,
  std::tuple_size<typename remove_cvref<ShapeType>::type>::value,
  decltype(s),
  decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor from a conforming container type
 *
 * Conforming containers have sequential iterators defined (both const and non-const). std::array
 * and std::vector meet this criteria.  Caller is responsible for deleting tensor.
 *
 * @param tensor Tensor object to store newly-created tensor into
 * @param shape  Shape of tensor
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns Pointer to new tensor
 *
 **/
template <typename TensorType,
  std::enable_if_t< is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor_p( TensorType &tensor, 
                    typename TensorType::shape_container &&shape, 
                    matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                    cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::scalar_type, typename TensorType::shape_container>(std::forward<typename TensorType::shape_container>(shape), space, stream);
  tensor.Shallow(tmp);
}

/**
 * Create a 0D tensor with implicitly-allocated memory.
 *
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename T>
auto make_tensor( matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                  cudaStream_t stream = 0) {  
  std::array<index_t, 0> shape;

  return make_tensor<T, decltype(shape)>(std::move(shape), space, stream);
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
template <typename TensorType, 
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                  cudaStream_t stream = 0) {  
  auto tmp = make_tensor<typename TensorType::scalar_type>(space, stream);
  tensor.Shallow(tmp);
}

/**
 * Create a 0D tensor with user-defined memory.
 *
 * @param space  memory space to allocate in.  Default is managed memory memory.
 * @param stream cuda stream to allocate in (only applicable to async allocations)
 * @returns New tensor
 *
 **/
template <typename T>
auto make_tensor_p( matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                    cudaStream_t stream = 0) {
  
  std::array<index_t, 0> shape;
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
  
  DefaultDescriptor<RANK> desc{shape};
  raw_pointer_buffer<T, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize()*sizeof(T)), owning};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
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
 * @param owning
 *   If this class owns memory of data
 * @returns New tensor
 **/
template <typename TensorType,
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  typename TensorType::scalar_type *data, 
                  const index_t (&shape)[TensorType::Rank()], 
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::scalar_type, TensorType::Rank()>(data, shape, owning);
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
template <typename T, typename ShapeType,
  std::enable_if_t<!is_matx_descriptor_v<ShapeType> && !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor( T *data, 
                  ShapeType &&shape, 
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  constexpr int RANK = static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK>
    desc{std::forward<ShapeType>(shape)};
  raw_pointer_buffer<T, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize()*sizeof(T)), owning};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
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
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename TensorType,
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  typename TensorType::scalar_type *data, 
                  typename TensorType::shape_container &&shape, 
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::scalar_type, typename TensorType::shape_container>(data, std::forward<typename TensorType::shape_container>(shape), owning);
  tensor.Shallow(tmp);
}

/**
 * Create a 0D tensor with user-defined memory
 *
 * @param ptr
 *  Pointer to data
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename T>
auto make_tensor( T *ptr, 
                  bool owning = false) {
  std::array<index_t, 0> shape;
  return make_tensor<T, decltype(shape)>(ptr, std::move(shape), owning);
}

/**
 * Create a 0D tensor with user-defined memory
 *
 * @param tensor 
 *  Tensor object to store newly-created tensor into
 * @param ptr
 *  Pointer to data
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename TensorType,
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  typename TensorType::scalar_type *ptr, 
                  bool owning = false) {
  auto tmp = make_tensor<typename TensorType::scalar_type>(ptr, owning);
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
template <typename T, typename ShapeType,
  std::enable_if_t<!is_matx_descriptor_v<ShapeType> && !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor_p( T *const data, 
                    ShapeType &&shape, 
                    bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  constexpr int RANK = static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK>
    desc{std::forward<ShapeType>(shape)};
  raw_pointer_buffer<T, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize()*sizeof(T)), owning};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return new tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with user-defined memory, custom storage, and conforming shape type
 *
 * @param s
 *   Storage object
 * @param shape
 *   Shape of tensor
 * @returns New tensor
 **/
template <typename Storage, typename ShapeType,
  std::enable_if_t<is_matx_storage_v<Storage> &&
                  !is_matx_descriptor_v<ShapeType> &&
                  !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor( Storage &&s, 
                  ShapeType &&shape) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  constexpr int RANK = static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK>
    desc{std::forward<ShapeType>(shape)};
  using T = typename Storage::T;
  return tensor_t<T, RANK, Storage, decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with user-defined memory, custom storage, and conforming shape type
 *
 * @param tensor 
 *   Tensor object to store newly-created tensor into
 * @param s
 *   Storage object
 * @param shape
 *   Shape of tensor
 * @returns New tensor
 **/
template <typename TensorType,
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  typename TensorType::storage_type &&s, 
                  typename TensorType::shape_container &&shape) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::storage_type, typename TensorType::shape_container>(std::forward<typename TensorType::storage_type>(s), std::forward<typename TensorType::shape_container>(shape));
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
template <typename T, typename D, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<D>::type>, bool> = true>
auto make_tensor( T* const data, 
                  D &&desc, 
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  using Dstrip = typename remove_cvref<D>::type;
  raw_pointer_buffer<T, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize()*sizeof(T)), owning};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, Dstrip::Rank(), decltype(s), Dstrip>{std::move(s), std::forward<D>(desc)};
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
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename TensorType,
          std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  typename TensorType::scalar_type* const data, 
                  typename TensorType::desc_type &&desc, 
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::scalar_type, typename TensorType::desc_type>(data, std::forward<typename TensorType::desc_type>(desc), owning);
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
template <typename T, typename D, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<D>::type>, bool> = true>
auto make_tensor( D &&desc, 
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  T *ptr;
  using Dstrip = typename remove_cvref<D>::type;

  size_t size = desc.TotalSize() * sizeof(T);
  matxAlloc((void**)&ptr, desc.TotalSize() * sizeof(T), space, stream);

  raw_pointer_buffer<T, matx_allocator<T>> rp(ptr, size, true);
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, Dstrip::Rank(), decltype(s), Dstrip>{std::move(s), std::forward<D>(desc)};
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
template <typename TensorType,
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &&tensor, 
                  typename TensorType::desc_type &&desc, 
                  matxMemorySpace_t space = MATX_MANAGED_MEMORY, 
                  cudaStream_t stream = 0) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::scalar_type, typename TensorType::desc_type>(std::forward<typename TensorType::desc_type>(desc), space, stream);
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
  
  DefaultDescriptor<RANK>  desc{shape, strides};
  raw_pointer_buffer<T, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize()*sizeof(T)), owning};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T,RANK, decltype(s), decltype(desc)>{s, std::move(desc), data};
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
 * @param owning
 *    If this class owns memory of data
 * @returns New tensor
 **/
template <typename TensorType,
  std::enable_if_t<is_tensor_view_v<TensorType>, bool> = true>
auto make_tensor( TensorType &tensor, 
                  typename TensorType::scalar_type *const data, 
                  const index_t (&shape)[TensorType::Rank()], 
                  const index_t (&strides)[TensorType::Rank()], 
                  bool owning = false) {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  auto tmp = make_tensor<typename TensorType::scalar_type, TensorType::Rank()>(data, shape, strides, owning);
  tensor.Shallow(tmp);
}


/**
 * Create a static-sized tensor with implicit memory
 * @returns New tensor
 **/
template <typename T, index_t I, index_t ...Is>
auto make_static_tensor() {
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  
  static_tensor_desc_t<I, Is...> desc{};
  raw_pointer_buffer<T, matx_allocator<T>> rp{static_cast<size_t>(desc.TotalSize()*sizeof(T))};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, desc.Rank(), decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

} // namespace matx
