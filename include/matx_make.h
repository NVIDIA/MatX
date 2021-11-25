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

#include "matx_storage.h"
#include "matx_tensor_desc.h"

namespace matx {

/**
 * Create a tensor
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK, typename O = owning>
auto make_tensor(const index_t (&shape)[RANK]) {
  DefaultDescriptor<RANK> desc{shape};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

template <typename T, int RANK, typename O = owning>
auto make_tensor_p(const index_t (&shape)[RANK]) {
  DefaultDescriptor<RANK> desc{shape};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return  new tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK, typename O = owning>
auto make_tensor(const tensorShape_t<RANK> &shape) {
  DefaultDescriptor<RANK> desc{std::move(shape.AsArray())};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

template <typename T, int RANK, typename O = owning>
auto make_tensor_p(const tensorShape_t<RANK> &shape) {
  DefaultDescriptor<RANK> desc{std::move(shape.AsArray())};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return new tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK, typename O = owning>
auto make_tensor(T *ptr, const tensorShape_t<RANK> &shape) {
  DefaultDescriptor<RANK> desc{std::move(shape.AsArray())};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{ptr, static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, typename ShapeType, typename O = owning,
  std::enable_if_t< !is_matx_shape_v<ShapeType> && 
                    !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor(ShapeType &&shape) {
  DefaultDescriptor<static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value)> desc{std::move(shape)};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, 
  std::tuple_size<typename remove_cvref<ShapeType>::type>::value, 
  decltype(s), 
  decltype(desc)>{std::move(s), std::move(desc)};
}

template <typename T, typename ShapeType, typename O = owning,
  std::enable_if_t< !is_matx_shape_v<ShapeType> && 
                    !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor_p(ShapeType &&shape) {
  DefaultDescriptor<static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value)> desc{std::move(shape)};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return new tensor_t<T, 
  std::tuple_size<typename remove_cvref<ShapeType>::type>::value, 
  decltype(s), 
  decltype(desc)>{std::move(s), std::move(desc)};
}

template <typename T>
auto make_tensor() {
  std::array<T, 0> shape;
  return make_tensor<T, 0>(std::move(shape));
}

template <typename T>
auto make_tensor(T *ptr) {
  std::array<T, 0> shape;
  return make_tensor<T, 0>(ptr, std::move(shape));
}



/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, int RANK, typename O = owning>
auto make_tensor(T *const data, const index_t (&shape)[RANK]) {
  DefaultDescriptor<RANK> desc{shape};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, typename ShapeType, typename O = owning, 
  std::enable_if_t<!is_matx_descriptor_v<ShapeType> && !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor(T *const data, ShapeType &&shape) {
  constexpr int RANK = static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK> 
    desc{std::forward<ShapeType>(shape)};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, typename ShapeType, typename O = owning, 
  std::enable_if_t<!is_matx_descriptor_v<ShapeType> && !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor_p(T *const data, ShapeType &&shape) {
  constexpr int RANK = static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK> 
    desc{std::forward<ShapeType>(shape)};
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return new tensor_t<T, RANK, decltype(s), decltype(desc)>{std::move(s), std::move(desc)};
}

/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename Storage, typename ShapeType,  
  std::enable_if_t<is_matx_storage_v<Storage> && 
                  !is_matx_descriptor_v<ShapeType> && 
                  !std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
auto make_tensor(Storage s, ShapeType &&shape) {
  constexpr int RANK = static_cast<int>(std::tuple_size<typename remove_cvref<ShapeType>::type>::value);
  DefaultDescriptor<RANK> 
    desc{std::forward<ShapeType>(shape)};  
  using T = typename Storage::T;
  return tensor_t<T, RANK, Storage, decltype(desc)>{s, std::move(desc)};
}


/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 **/
template <typename T, typename D, typename O = owning, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<D>::type>, bool> = true>
auto make_tensor(T* const data, D &&desc) {    
  using Dstrip = typename remove_cvref<D>::type;
  raw_pointer_buffer<T, O, matx_allocator<T>> rp{data, static_cast<size_t>(desc.TotalSize())};
  basic_storage<decltype(rp)> s{std::move(rp)};
  return tensor_t<T, Dstrip::Rank(), decltype(s), Dstrip>{std::move(s), std::forward<D>(desc)};
}




// template <typename T, typename T2>
// int make_tensor(T&&t, T2&&t2) {}

/**
 * Create a tensor with user memory
 *
 * @param data
 *   Pointer to device data
 * @param shape
 *   Shape of tensor
 * @param strides
 *   Strides of tensor
 **/
template <typename T, int RANK>
auto make_tensor(T *const data, const index_t (&shape)[RANK], const index_t (&strides)[RANK]) {
  return tensor_t<T,RANK>{data, shape, strides};
}

} // namespace matx