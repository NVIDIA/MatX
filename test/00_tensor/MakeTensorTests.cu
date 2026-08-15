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
#include <cuda/std/mdspan>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <limits>


using namespace matx;

namespace {

// This kernel helps test operator[] on the GPU
// Since GoogleTests tests run on CPU, the GPU code is placed in 
// a CUDA kernel and launched from the test
#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202211L)
  __global__ void TestMultidimensionalSubscriptKernel(detail::tensor_impl_t<int, 2> tensor)
  {
    tensor[1, 2] = 27;
  }
#endif
  
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

// Checks whether make_tensor(span) creates a correct MatX
// view of the same memory described by the mdspan
// 1. Checks tensor type
// 2. Same memory for the MatX tensor and mdspan
// 3. Same dimensions 
// 4. Changes through either view are visible through the other
TEST(MakeTensorTests, CreatesLayoutRightStaticMdspanView)
{
  MATX_ENTER_HANDLER();

  using extents_type = cuda::std::extents<index_t, 2, 3>;

  int data[6] = {};
  cuda::std::mdspan<int, extents_type, cuda::std::layout_right> span{data};

  auto tensor = make_tensor(span);

  // Verify the mdspan overload returns the expected tensor type
  static_assert(decltype(tensor)::Rank() == 2);
  static_assert(
      std::is_same_v<typename decltype(tensor)::value_type, int>);

  // MatX must use the original pointer without copying the data
  EXPECT_EQ(tensor.Data(), span.data_handle());

  // Checks the dimensions
  ExpectShape2(tensor, 2, 3);

  // Checks if MatX can access different elements
  // Check that MatX row stride matches the mdspan’s row stride
  EXPECT_EQ(tensor.Stride(0), span.mapping().stride(0)); 
 
  // Check that MatX column stride matches the mdspan’s column stride
  EXPECT_EQ(tensor.Stride(1), span.mapping().stride(1)); 

  // If you write through MatX tensor, it will change the original memory
  tensor(1, 2) = 71;
  EXPECT_EQ(data[5], 71);

  // Writing to the original memory must be visible through MatX.
  data[1] = 72;
  EXPECT_EQ(tensor(0, 1), 72);

  MATX_EXIT_HANDLER();
}

// Test for row-major dynamic tensors
TEST(MakeTensorTests, CreatesLayoutRightDynamicMdspanView)
{
  MATX_ENTER_HANDLER();

  // This says the mdspan has 2 dimensions, but their sizes are given later.
  using extents_type = cuda::std::dextents<index_t, 2>;

  int data[6] = {};

  // Passing 2 and 3 provides the dimension sizes at runtime
  cuda::std::mdspan<int, extents_type, cuda::std::layout_right> span{
      data, 2, 3};

  auto tensor = make_tensor(span);

  // Verify the mdspan overload returns the expected tensor type
  static_assert(decltype(tensor)::Rank() == 2);
  static_assert(
      std::is_same_v<typename decltype(tensor)::value_type, int>);

  // MatX must use the original pointer without copying the data
  EXPECT_EQ(tensor.Data(), span.data_handle());

  // Checks the dimensions
  ExpectShape2(tensor, 2, 3);

  // Checks if MatX can access different elements
  // Check that MatX row stride matches the mdspan's row stride
  EXPECT_EQ(tensor.Stride(0), span.mapping().stride(0));

  // Check that MatX column stride matches the mdspan's column stride
  EXPECT_EQ(tensor.Stride(1), span.mapping().stride(1));

  // Writing through the MatX tensor changes the original memory
  // Writing to the original memory is visible through MatX
  tensor(1, 2) = 71;
  EXPECT_EQ(data[5], 71);

  // If you write on the original memory, it must be visible through MatX
  data[1] = 72;
  EXPECT_EQ(tensor(0, 1), 72);

  MATX_EXIT_HANDLER();
}

TEST(MakeTensorTests, CreatesLayoutLeftDynamicMdspanView)
{
  MATX_ENTER_HANDLER();

  // This says the mdspan has 2 dimensions, but their sizes are given later.
  using extents_type = cuda::std::dextents<index_t, 2>;

  int data[6] = {};

  // Passing 2 and 3 here gives this mdspan its dimensions at runtime.
  cuda::std::mdspan<int, extents_type, cuda::std::layout_left> span{
      data, 2, 3};

  auto tensor = make_tensor(span);

  // Verify the mdspan overload returns the expected tensor type
  static_assert(decltype(tensor)::Rank() == 2);
  static_assert(
      std::is_same_v<typename decltype(tensor)::value_type, int>);

  // MatX must use the original pointer without copying the data
  EXPECT_EQ(tensor.Data(), span.data_handle());

  // Checks the dimensions
  ExpectShape2(tensor, 2, 3);

  // Checks if MatX can access different elements
  // Check that MatX row stride matches the mdspan's row stride
  EXPECT_EQ(tensor.Stride(0), span.mapping().stride(0));

  // Check that MatX column stride matches the mdspan's column stride
  EXPECT_EQ(tensor.Stride(1), span.mapping().stride(1));

  // If you write on the MatX tensor, it will change the original memory
  tensor(1, 2) = 81;
  EXPECT_EQ(data[5], 81);

  // If you write on the original memory, it must be visible through MatX
  data[2] = 82;
  EXPECT_EQ(tensor(0, 1), 82);

  MATX_EXIT_HANDLER();
}


// Checks whether make_tensor(span) preserves custom strides
// that are neither row-major nor column-major
TEST(MakeTensorTests, CreatesLayoutStrideMdspanView)
{
  MATX_ENTER_HANDLER();

  using extents_type = cuda::std::extents<index_t, 2, 3>;

  // Allows custom strides
  using mapping_type =
      cuda::std::layout_stride::mapping<extents_type>;

  // The largest accessed position is 1 * 4 + 2 * 1 = 6
  int data[7] = {};

  // Move 4 positions for each row and 1 for each column
  const mapping_type mapping{
      extents_type{},
      cuda::std::array<index_t, 2>{4, 1}};

  cuda::std::mdspan<int, extents_type, cuda::std::layout_stride> span{
      data, mapping};

  auto tensor = make_tensor(span);

  static_assert(decltype(tensor)::Rank() == 2);
  static_assert(
      std::is_same_v<typename decltype(tensor)::value_type, int>);

  // Checks that MatX uses the same memory
  EXPECT_EQ(tensor.Data(), span.data_handle());

  // Checks the dimensions
  ExpectShape2(tensor, 2, 3);

  // Checks that MatX keeps the custom strides
  EXPECT_EQ(tensor.Stride(0), span.mapping().stride(0));
  EXPECT_EQ(tensor.Stride(1), span.mapping().stride(1));

  // Position (1, 2) maps to data[6]
  tensor(1, 2) = 91;
  EXPECT_EQ(data[6], 91);

  // Position (1, 0) maps to data[4]
  data[4] = 92;
  EXPECT_EQ(tensor(1, 0), 92);

  MATX_EXIT_HANDLER();
}

// Checks that destroying the MatX tensor does not destroy
// original data's memory 
TEST(MakeTensorTests, MdspanViewDoesNotOwnData)
{
  MATX_ENTER_HANDLER();

  // This unique pointer owns the original memory
  auto data = std::make_unique<int[]>(6);
  int *const original_data = data.get();

  {
    // The mdspan and MatX tensor only borrow the original memory
    cuda::std::mdspan<int, cuda::std::dextents<index_t, 2>> span{
        original_data, 2, 3};

    auto tensor = make_tensor(span);

    // Checks that MatX uses the original memory
    EXPECT_EQ(tensor.Data(), original_data);

    // Writes a value before the tensor is destroyed
    tensor(1, 2) = 101;
  }

  // The original memory must still exist after the tensor is destroyed
  EXPECT_EQ(data.get(), original_data);
  EXPECT_EQ(original_data[5], 101);

  MATX_EXIT_HANDLER();
}

// Checks that an mdspan extent or stride that is too large for MatX
// index_t throws an error instead of being converted to a smaller value.
TEST(MakeTensorTests, RejectsMdspanValuesOutsideMatXIndexRange)
{
  MATX_ENTER_HANDLER();

  int data = 0;
  const uint64_t oversized =
      static_cast<uint64_t>(std::numeric_limits<index_t>::max()) + 1ULL;

  // Checks an extent that is too large for MatX index_t
  {
    using extents_type = cuda::std::dextents<uint64_t, 1>;

    cuda::std::mdspan<int, extents_type> span{&data, oversized};

    EXPECT_THROW(
        { [[maybe_unused]] auto tensor = make_tensor(span); },
        matx::detail::matxException);
  }

  // Checks a stride that is too large for MatX index_t
  {
    using extents_type = cuda::std::extents<uint64_t, 1, 1>;
    using mapping_type =
        cuda::std::layout_stride::mapping<extents_type>;

    const mapping_type mapping{
        extents_type{},
        cuda::std::array<uint64_t, 2>{oversized, 1}};

    cuda::std::mdspan<int, extents_type, cuda::std::layout_stride> span{
        &data, mapping};

    EXPECT_THROW(
        { [[maybe_unused]] auto tensor = make_tensor(span); },
        matx::detail::matxException);
  }

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

#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202211L)
  
  // Tests if operator[] works on CPU
  TEST(MakeTensorTests, MultidimensionalSubscriptHost)
  {
    MATX_ENTER_HANDLER();

    auto tensor = make_tensor<int>({2, 3}, MATX_MANAGED_MEMORY);

    tensor[1, 2] = 17;


    // Checks if the new operator[] can read 
    // the value we wrote using operator[]
    EXPECT_EQ((tensor[1, 2]), 17);

    // Checks if the old operator() refers to the
    // same element as new operator[]
    EXPECT_EQ((tensor(1, 2)), 17);

    // Reads the tensor using the const reference to the tensor
    const auto &const_tensor = tensor;
    EXPECT_EQ((const_tensor[1, 2]), 17);

    MATX_EXIT_HANDLER();
  }

  // Tests if operator[] works on GPU
  TEST(MakeTensorTests, MultidimensionalSubscriptDevice)
  {
    MATX_ENTER_HANDLER();

    auto tensor = make_tensor<int>({2, 3}, MATX_MANAGED_MEMORY);

    detail::tensor_impl_t<int, 2> tensor_impl = tensor;
    TestMultidimensionalSubscriptKernel<<<1, 1>>>(tensor_impl);

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ((tensor[1, 2]), 27);
    EXPECT_EQ((tensor(1, 2)), 27);

    MATX_EXIT_HANDLER();
  }

#endif
