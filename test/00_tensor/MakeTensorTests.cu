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

#include "matx.h"
#include "gtest/gtest.h"

#include <cuda/std/array>
#include <cstdlib>
#include <memory>

using namespace matx;

namespace {

struct CountingAllocator {
  static inline size_t allocations = 0;
  static inline size_t deallocations = 0;
  static inline size_t active_bytes = 0;

  static void Reset()
  {
    allocations = 0;
    deallocations = 0;
    active_bytes = 0;
  }

  void *allocate(size_t bytes)
  {
    allocations++;
    active_bytes += bytes;
    return std::malloc(bytes);
  }

  void deallocate(void *ptr, size_t bytes)
  {
    deallocations++;
    active_bytes -= bytes;
    std::free(ptr);
  }
};

template <typename TensorType>
void ExpectShape2(const TensorType &tensor, index_t dim0, index_t dim1)
{
  EXPECT_EQ(tensor.Rank(), 2);
  EXPECT_EQ(tensor.Size(0), dim0);
  EXPECT_EQ(tensor.Size(1), dim1);
  EXPECT_EQ(tensor.TotalSize(), dim0 * dim1);
}

} // namespace

TEST(MakeTensorTests, CreatesAllocatedAndPlacementTensors)
{
  MATX_ENTER_HANDLER();

  index_t shape2[2] = {2, 3};

  auto c_array_tensor = make_tensor<int, 2>(shape2, MATX_HOST_MALLOC_MEMORY);
  ExpectShape2(c_array_tensor, 2, 3);
  c_array_tensor(1, 2) = 12;
  EXPECT_EQ(c_array_tensor(1, 2), 12);

  tensor_t<int, 2> placed_c_array;
  make_tensor(placed_c_array, shape2, MATX_HOST_MALLOC_MEMORY);
  ExpectShape2(placed_c_array, 2, 3);
  placed_c_array(1, 1) = 11;
  EXPECT_EQ(placed_c_array(1, 1), 11);

  std::unique_ptr<tensor_t<int, 2>> c_array_ptr(
      make_tensor_p<int, 2>(shape2, MATX_HOST_MALLOC_MEMORY));
  ExpectShape2(*c_array_ptr, 2, 3);

  auto shape3 = cuda::std::array<index_t, 3>{2, 3, 4};
  auto container_tensor = make_tensor<int>(shape3, MATX_HOST_MALLOC_MEMORY);
  EXPECT_EQ(container_tensor.Rank(), 3);
  EXPECT_EQ(container_tensor.Size(0), 2);
  EXPECT_EQ(container_tensor.Size(1), 3);
  EXPECT_EQ(container_tensor.Size(2), 4);

  tensor_t<int, 3> placed_container;
  make_tensor(placed_container, cuda::std::array<index_t, 3>{2, 3, 4},
              MATX_HOST_MALLOC_MEMORY);
  EXPECT_EQ(placed_container.Size(0), 2);
  EXPECT_EQ(placed_container.Size(1), 3);
  EXPECT_EQ(placed_container.Size(2), 4);

  std::unique_ptr<tensor_t<int, 1>> container_ptr(
      make_tensor_p<int>(cuda::std::array<index_t, 1>{6},
                         MATX_HOST_MALLOC_MEMORY));
  EXPECT_EQ(container_ptr->Size(0), 6);

  auto scalar_tensor = make_tensor<int>({}, MATX_HOST_MALLOC_MEMORY);
  scalar_tensor() = 7;
  EXPECT_EQ(scalar_tensor(), 7);

  tensor_t<int, 0> placed_scalar;
  make_tensor(placed_scalar, MATX_HOST_MALLOC_MEMORY);
  placed_scalar() = 9;
  EXPECT_EQ(placed_scalar(), 9);

  std::unique_ptr<tensor_t<int, 0>> scalar_ptr(
      make_tensor_p<int>({}, MATX_HOST_MALLOC_MEMORY));
  (*scalar_ptr)() = 13;
  EXPECT_EQ((*scalar_ptr)(), 13);

  auto static_tensor = make_tensor<int, 2, 3>();
  ExpectShape2(static_tensor, 2, 3);
  static_tensor(1, 2) = 15;
  EXPECT_EQ(static_tensor(1, 2), 15);

  MATX_EXIT_HANDLER();
}

TEST(MakeTensorTests, CreatesPointerBackedViews)
{
  MATX_ENTER_HANDLER();

  int data[8] = {};
  index_t shape2[2] = {2, 3};
  index_t strides2[2] = {3, 1};

  Storage<int> storage(data, 6);
  auto storage_tensor =
      make_tensor(std::move(storage), cuda::std::array<index_t, 2>{2, 3});
  ExpectShape2(storage_tensor, 2, 3);
  storage_tensor(1, 2) = 17;
  EXPECT_EQ(data[5], 17);

  auto c_array_view = make_tensor(data, shape2);
  ExpectShape2(c_array_view, 2, 3);
  c_array_view(1, 1) = 21;
  EXPECT_EQ(data[4], 21);
  EXPECT_EQ(c_array_view.Data(), data);

  tensor_t<int, 2> placed_c_array_view;
  make_tensor(placed_c_array_view, data, shape2);
  ExpectShape2(placed_c_array_view, 2, 3);
  placed_c_array_view(0, 2) = 22;
  EXPECT_EQ(data[2], 22);

  auto container_view =
      make_tensor(data, cuda::std::array<index_t, 2>{2, 3});
  ExpectShape2(container_view, 2, 3);
  container_view(0, 1) = 23;
  EXPECT_EQ(data[1], 23);

  tensor_t<int, 2> placed_container_view;
  make_tensor(placed_container_view, data,
              cuda::std::array<index_t, 2>{2, 3});
  ExpectShape2(placed_container_view, 2, 3);
  placed_container_view(0, 0) = 24;
  EXPECT_EQ(data[0], 24);

  int scalar = 3;
  auto scalar_view = make_tensor(&scalar, {});
  EXPECT_EQ(scalar_view.Data(), &scalar);
  scalar_view() = 31;
  EXPECT_EQ(scalar, 31);

  tensor_t<int, 0> placed_scalar_view;
  make_tensor(placed_scalar_view, &scalar);
  EXPECT_EQ(placed_scalar_view.Data(), &scalar);
  placed_scalar_view() = 32;
  EXPECT_EQ(scalar, 32);

  std::unique_ptr<tensor_t<int, 2>> pointer_view_ptr(
      make_tensor_p(data, cuda::std::array<index_t, 2>{2, 3}));
  ExpectShape2(*pointer_view_ptr, 2, 3);
  (*pointer_view_ptr)(1, 0) = 33;
  EXPECT_EQ(data[3], 33);

  auto strided_view = make_tensor(data, shape2, strides2);
  ExpectShape2(strided_view, 2, 3);
  strided_view(1, 1) = 41;
  EXPECT_EQ(data[4], 41);

  tensor_t<int, 2> placed_strided_view;
  make_tensor(placed_strided_view, data, shape2, strides2);
  ExpectShape2(placed_strided_view, 2, 3);
  placed_strided_view(1, 2) = 42;
  EXPECT_EQ(data[5], 42);

  MATX_EXIT_HANDLER();
}

TEST(MakeTensorTests, CreatesDescriptorBackedTensors)
{
  MATX_ENTER_HANDLER();

  int data[6] = {};
  auto shape = cuda::std::array<index_t, 2>{2, 3};
  DefaultDescriptor<2> desc{shape};

  auto allocated = make_tensor<int>(desc, MATX_HOST_MALLOC_MEMORY);
  ExpectShape2(allocated, 2, 3);
  allocated(1, 2) = 51;
  EXPECT_EQ(allocated(1, 2), 51);

  auto view = make_tensor(data, desc);
  ExpectShape2(view, 2, 3);
  view(1, 2) = 52;
  EXPECT_EQ(data[5], 52);

  tensor_t<int, 2> placed_view;
  make_tensor(placed_view, data,
              DefaultDescriptor<2>{cuda::std::array<index_t, 2>{2, 3}});
  ExpectShape2(placed_view, 2, 3);
  placed_view(1, 1) = 53;
  EXPECT_EQ(data[4], 53);

  tensor_t<int, 2> placed_allocated;
  make_tensor(placed_allocated,
              DefaultDescriptor<2>{cuda::std::array<index_t, 2>{2, 3}},
              MATX_HOST_MALLOC_MEMORY);
  ExpectShape2(placed_allocated, 2, 3);
  placed_allocated(0, 2) = 54;
  EXPECT_EQ(placed_allocated(0, 2), 54);

  MATX_EXIT_HANDLER();
}

TEST(MakeTensorTests, CreatesCustomAllocatorTensors)
{
  MATX_ENTER_HANDLER();

  CountingAllocator::Reset();

  {
    index_t shape1[1] = {4};
    index_t shape2[2] = {2, 3};

    auto c_array_tensor = make_tensor<int>(shape1, CountingAllocator{});
    EXPECT_EQ(c_array_tensor.Size(0), 4);
    c_array_tensor(3) = 61;
    EXPECT_EQ(c_array_tensor(3), 61);

    auto container_tensor =
        make_tensor<int>(cuda::std::array<index_t, 2>{2, 3},
                         CountingAllocator{});
    ExpectShape2(container_tensor, 2, 3);
    container_tensor(1, 2) = 62;
    EXPECT_EQ(container_tensor(1, 2), 62);

    tensor_t<int, 2> placed_c_array;
    make_tensor(placed_c_array, shape2, CountingAllocator{});
    ExpectShape2(placed_c_array, 2, 3);
    placed_c_array(1, 1) = 63;
    EXPECT_EQ(placed_c_array(1, 1), 63);

    tensor_t<int, 2> placed_container;
    make_tensor(placed_container, cuda::std::array<index_t, 2>{2, 3},
                CountingAllocator{});
    ExpectShape2(placed_container, 2, 3);
    placed_container(0, 2) = 64;
    EXPECT_EQ(placed_container(0, 2), 64);

    EXPECT_EQ(CountingAllocator::allocations, 4);
    EXPECT_EQ(CountingAllocator::active_bytes,
              (4 + 6 + 6 + 6) * sizeof(int));
  }

  EXPECT_EQ(CountingAllocator::allocations, CountingAllocator::deallocations);
  EXPECT_EQ(CountingAllocator::active_bytes, 0);

  MATX_EXIT_HANDLER();
}
