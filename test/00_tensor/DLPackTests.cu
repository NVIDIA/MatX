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

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "gtest/gtest.h"

#include <initializer_list>
#include <limits>

using namespace matx;

template <typename T>
struct DLPackOwningImportContext {
  int *deleter_calls;
  T *data;
  int64_t *shape;
  int64_t *strides;
};

template <typename T>
void DLPackOwningImportDeleter(DLManagedTensor *mt) {
  auto *ctx = static_cast<DLPackOwningImportContext<T> *>(mt->manager_ctx);
  if (ctx != nullptr) {
    (*ctx->deleter_calls)++;
    delete[] ctx->data;
    delete[] ctx->shape;
    delete[] ctx->strides;
    delete ctx;
  }
  delete mt;
}

template <typename T>
DLManagedTensor *MakeManagedTensorForOwningImportTest(int *deleter_calls, int64_t size) {
  auto *mt = new DLManagedTensor{};
  auto *ctx = new DLPackOwningImportContext<T>{};

  ctx->deleter_calls = deleter_calls;
  ctx->data = new T[static_cast<size_t>(size)];
  ctx->shape = new int64_t[1];
  ctx->strides = new int64_t[1];
  ctx->shape[0] = size;
  ctx->strides[0] = 1;

  mt->dl_tensor.data = static_cast<void *>(ctx->data);
  mt->dl_tensor.device.device_type = kDLCPU;
  mt->dl_tensor.device.device_id = 0;
  mt->dl_tensor.ndim = 1;
  mt->dl_tensor.dtype = detail::TypeToDLPackType<T>();
  mt->dl_tensor.shape = ctx->shape;
  mt->dl_tensor.strides = ctx->strides;
  mt->dl_tensor.byte_offset = 0;
  mt->manager_ctx = ctx;
  mt->deleter = &DLPackOwningImportDeleter<T>;

  return mt;
}

template <typename T>
void SetManagedTensorShapeAndStrides(
    DLManagedTensor *mt, std::initializer_list<int64_t> shape,
    std::initializer_list<int64_t> strides) {
  auto *ctx = static_cast<DLPackOwningImportContext<T> *>(mt->manager_ctx);

  delete[] ctx->shape;
  delete[] ctx->strides;
  ctx->shape = new int64_t[shape.size()];
  ctx->strides = strides.size() == 0 ? nullptr : new int64_t[strides.size()];

  size_t idx = 0;
  for (auto dim : shape) {
    ctx->shape[idx++] = dim;
  }

  idx = 0;
  for (auto stride : strides) {
    ctx->strides[idx++] = stride;
  }

  mt->dl_tensor.ndim = static_cast<int32_t>(shape.size());
  mt->dl_tensor.shape = ctx->shape;
  mt->dl_tensor.strides = ctx->strides;
}

template <typename T>
struct DLPackVersionedOwningImportContext {
  int *deleter_calls;
  T *data;
  int64_t *shape;
  int64_t *strides;
};

template <typename T>
void DLPackVersionedOwningImportDeleter(DLManagedTensorVersioned *mt) {
  auto *ctx = static_cast<DLPackVersionedOwningImportContext<T> *>(mt->manager_ctx);
  if (ctx != nullptr) {
    (*ctx->deleter_calls)++;
    delete[] ctx->data;
    delete[] ctx->shape;
    delete[] ctx->strides;
    delete ctx;
  }
  delete mt;
}

template <typename T>
DLManagedTensorVersioned *MakeVersionedManagedTensorForOwningImportTest(
    int *deleter_calls, int64_t size, uint64_t flags = 0) {
  auto *mt = new DLManagedTensorVersioned{};
  auto *ctx = new DLPackVersionedOwningImportContext<T>{};

  ctx->deleter_calls = deleter_calls;
  ctx->data = new T[static_cast<size_t>(size)];
  ctx->shape = new int64_t[1];
  ctx->strides = new int64_t[1];
  ctx->shape[0] = size;
  ctx->strides[0] = 1;

  mt->version.major = DLPACK_MAJOR_VERSION;
  mt->version.minor = DLPACK_MINOR_VERSION;
  mt->dl_tensor.data = static_cast<void *>(ctx->data);
  mt->dl_tensor.device.device_type = kDLCPU;
  mt->dl_tensor.device.device_id = 0;
  mt->dl_tensor.ndim = 1;
  mt->dl_tensor.dtype = detail::TypeToDLPackType<T>();
  mt->dl_tensor.shape = ctx->shape;
  mt->dl_tensor.strides = ctx->strides;
  mt->dl_tensor.byte_offset = 0;
  mt->manager_ctx = ctx;
  mt->flags = flags;
  mt->deleter = &DLPackVersionedOwningImportDeleter<T>;

  return mt;
}

template <typename T>
void SetVersionedManagedTensorShapeAndStrides(
    DLManagedTensorVersioned *mt, std::initializer_list<int64_t> shape,
    std::initializer_list<int64_t> strides) {
  auto *ctx = static_cast<DLPackVersionedOwningImportContext<T> *>(mt->manager_ctx);

  delete[] ctx->shape;
  delete[] ctx->strides;
  ctx->shape = new int64_t[shape.size()];
  ctx->strides = strides.size() == 0 ? nullptr : new int64_t[strides.size()];

  size_t idx = 0;
  for (auto dim : shape) {
    ctx->shape[idx++] = dim;
  }

  idx = 0;
  for (auto stride : strides) {
    ctx->strides[idx++] = stride;
  }

  mt->dl_tensor.ndim = static_cast<int32_t>(shape.size());
  mt->dl_tensor.shape = ctx->shape;
  mt->dl_tensor.strides = ctx->strides;
}

// Verifies that a view whose data pointer is offset from the start of its base
// allocation still exports the memory space of that allocation. GetPointerKind()
// is an exact-key lookup into the allocation map, so a view's offset pointer is
// never found there and ToDlPack must classify the memory from the base pointer
// held by the storage. Both the legacy and versioned paths are checked since
// they funnel through the same ToDlPackImpl.
template <typename ViewType>
void CheckOffsetViewDLPackExport(ViewType &v, const void *base,
                                 DLDeviceType expected_device_type)
{
  using TestType = typename ViewType::value_type;

  auto *data_ptr = const_cast<void *>(static_cast<const void *>(v.Data()));
  ASSERT_NE(data_ptr, base) << "view must start at a non-zero offset into its allocation";

  auto check_dl_tensor = [&](auto *dl) {
    ASSERT_EQ(dl->dl_tensor.ndim, v.Rank());

    // The view's offset pointer is exported as-is, with no additional byte offset
    ASSERT_EQ(dl->dl_tensor.data, data_ptr);
    ASSERT_EQ(dl->dl_tensor.byte_offset, 0U);

    ASSERT_EQ(dl->dl_tensor.device.device_type, expected_device_type);

    auto dlt = detail::TypeToDLPackType<TestType>();
    ASSERT_EQ(dl->dl_tensor.dtype.code, dlt.code);
    ASSERT_EQ(dl->dl_tensor.dtype.bits, dlt.bits);
    ASSERT_EQ(dl->dl_tensor.dtype.lanes, dlt.lanes);

    for (int32_t r = 0; r < v.Rank(); r++) {
      ASSERT_EQ(dl->dl_tensor.shape[r], v.Size(r));
      ASSERT_EQ(dl->dl_tensor.strides[r], v.Stride(r));
    }
  };

  // A view shares its storage with the tensor it came from, so compare the
  // reference count against what it was before the export rather than to 1
  const auto ref_count = v.GetRefCount();

  {
    auto *dl = v.ToDlPack();
    ASSERT_EQ(v.GetRefCount(), ref_count + 1);
    ASSERT_NO_FATAL_FAILURE(check_dl_tensor(dl));
    dl->deleter(dl);
    ASSERT_EQ(v.GetRefCount(), ref_count);
  }

  {
    auto *dlv = v.ToDlPackVersioned();
    ASSERT_EQ(v.GetRefCount(), ref_count + 1);
    ASSERT_NO_FATAL_FAILURE(check_dl_tensor(dlv));
    dlv->deleter(dlv);
    ASSERT_EQ(v.GetRefCount(), ref_count);
  }
}

// Calls check(tensor, expected_device_type) once for a {5, 10, 20} tensor in
// each memory space a tensor can be exported from. Shared by the full-tensor
// and offset-view export tests so the list of spaces lives in one place
template <typename TestType, typename CheckFunc>
void ForEachExportableMemorySpace(CheckFunc &&check)
{
  {
    // Default memory space -> exercises whatever the allocator handed back. On
    // devices where cudaDevAttrConcurrentManagedAccess == 0 (e.g. Jetson), the
    // allocator falls back to host pinned memory, so derive the expected device
    // type from the base allocation
    auto t = make_tensor<TestType>({5, 10, 20});
    auto kind = GetPointerKind(t.GetStorage().data());

    // Default allocation can only be MATX_MANAGED_MEMORY or MATX_HOST_MEMORY.
    // Assert that explicitly so a future allocator change can't silently
    // produce a wrong expectation below
    ASSERT_TRUE(kind == MATX_MANAGED_MEMORY || kind == MATX_HOST_MEMORY);
    ASSERT_NO_FATAL_FAILURE(check(t, kind == MATX_HOST_MEMORY ? kDLCUDAHost : kDLCUDA));
  }

  {
    // Explicit host pinned memory. This is the case that separates a base
    // pointer lookup from the cuPointerGetAttributes fallback: the fallback
    // cannot tell pinned memory from plain host memory and reports kDLCPU
    auto t = make_tensor<TestType>({5, 10, 20}, MATX_HOST_MEMORY);
    ASSERT_NO_FATAL_FAILURE(check(t, kDLCUDAHost));
  }

  {
    // Explicit plain malloc host memory -> kDLCPU
    auto t = make_tensor<TestType>({5, 10, 20}, MATX_HOST_MALLOC_MEMORY);
    ASSERT_NO_FATAL_FAILURE(check(t, kDLCPU));
  }

  {
    // Explicit device memory -> kDLCUDA
    auto t = make_tensor<TestType>({5, 10, 20}, MATX_DEVICE_MEMORY);
    ASSERT_NO_FATAL_FAILURE(check(t, kDLCUDA));
  }

  {
    // Async device memory on the default stream -> kDLCUDA
    auto t = make_tensor<TestType>({5, 10, 20}, MATX_ASYNC_DEVICE_MEMORY);
    ASSERT_NO_FATAL_FAILURE(check(t, kDLCUDA));
  }
}

template <typename TensorType>
class DLPackTestsAll : public ::testing::Test {
};
template <typename TensorType>
class DLPackTestsFloatNonComplex : public ::testing::Test {
};

TYPED_TEST_SUITE(DLPackTestsAll, MatXAllTypesCUDAExec);
TYPED_TEST_SUITE(DLPackTestsFloatNonComplex, MatXFloatNonComplexTypesCUDAExec);

TYPED_TEST(DLPackTestsAll, ExportLegacyDLPack)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto check_dl_export = [](auto &t, DLDeviceType expected_device_type) {
    auto dl = t.ToDlPack();

    ASSERT_EQ(dl->dl_tensor.ndim, 3);
    ASSERT_EQ(dl->dl_tensor.data, t.Data());
    ASSERT_EQ(dl->dl_tensor.device.device_id, 0);
    ASSERT_EQ(dl->dl_tensor.device.device_type, expected_device_type);
    auto dlt = detail::TypeToDLPackType<TestType>();
    ASSERT_EQ(dl->dl_tensor.dtype.code, dlt.code);
    ASSERT_EQ(dl->dl_tensor.dtype.bits, dlt.bits);
    ASSERT_EQ(dl->dl_tensor.dtype.lanes, dlt.lanes);
    ASSERT_EQ(dl->dl_tensor.shape[0], t.Size(0));
    ASSERT_EQ(dl->dl_tensor.shape[1], t.Size(1));
    ASSERT_EQ(dl->dl_tensor.shape[2], t.Size(2));
    ASSERT_EQ(dl->dl_tensor.strides[0], t.Stride(0));
    ASSERT_EQ(dl->dl_tensor.strides[1], t.Stride(1));
    ASSERT_EQ(dl->dl_tensor.strides[2], t.Stride(2));
    ASSERT_EQ(t.GetRefCount(), 2);
    dl->deleter(dl);
    ASSERT_EQ(t.GetRefCount(), 1);
  };

  ASSERT_NO_FATAL_FAILURE(ForEachExportableMemorySpace<TestType>(check_dl_export));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsAll, ExportOffsetSlice)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto check_offset_slice = [](auto &t, DLDeviceType expected_device_type) {
    const void *base = static_cast<const void *>(t.GetStorage().data());

    // Starts partway into the allocation, so Data() is no longer the base
    // pointer the allocator recorded
    auto s = t.Slice({1, 2, 3}, {4, 8, 15});

    ASSERT_EQ(s.Size(0), 3);
    ASSERT_EQ(s.Size(1), 6);
    ASSERT_EQ(s.Size(2), 12);

    // Pin the absolute layout of the view. CheckOffsetViewDLPackExport compares
    // the exported shape and strides against the view's own accessors, so
    // without this a regression in Slice would move both sides together
    ASSERT_EQ(s.Stride(0), 200);
    ASSERT_EQ(s.Stride(1), 20);
    ASSERT_EQ(s.Stride(2), 1);
    ASSERT_EQ(s.Data() - static_cast<const TestType *>(base), 1 * 200 + 2 * 20 + 3);

    ASSERT_NO_FATAL_FAILURE(CheckOffsetViewDLPackExport(s, base, expected_device_type));
  };

  ASSERT_NO_FATAL_FAILURE(ForEachExportableMemorySpace<TestType>(check_offset_slice));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsAll, ExportPermutedOffsetSlice)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto check_permuted_offset_slice = [](auto &t, DLDeviceType expected_device_type) {
    const void *base = static_cast<const void *>(t.GetStorage().data());
    auto s = t.Slice({1, 2, 3}, {4, 8, 15});
    auto p = s.Permute({2, 0, 1});

    // Permuting reorders sizes and strides but keeps the slice's offset pointer
    ASSERT_EQ(p.Data(), s.Data());
    ASSERT_EQ(p.Size(0), s.Size(2));
    ASSERT_EQ(p.Size(1), s.Size(0));
    ASSERT_EQ(p.Size(2), s.Size(1));
    ASSERT_EQ(p.Stride(0), s.Stride(2));
    ASSERT_EQ(p.Stride(1), s.Stride(0));
    ASSERT_EQ(p.Stride(2), s.Stride(1));
    ASSERT_NO_FATAL_FAILURE(CheckOffsetViewDLPackExport(p, base, expected_device_type));
  };

  ASSERT_NO_FATAL_FAILURE(ForEachExportableMemorySpace<TestType>(check_permuted_offset_slice));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, ExportVersionedConstTensorSetsReadOnlyFlag)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto t_const = make_tensor<const TestType>({5, 10});
  auto *dlv = t_const.ToDlPackVersioned();

  ASSERT_EQ(dlv->dl_tensor.ndim, 2);
  ASSERT_EQ(dlv->dl_tensor.data, const_cast<void *>(static_cast<const void *>(t_const.Data())));
  ASSERT_NE((dlv->flags & DLPACK_FLAG_BITMASK_READ_ONLY), 0U);

  auto dlt = detail::TypeToDLPackType<TestType>();
  ASSERT_EQ(dlv->dl_tensor.dtype.code, dlt.code);
  ASSERT_EQ(dlv->dl_tensor.dtype.bits, dlt.bits);
  ASSERT_EQ(dlv->dl_tensor.dtype.lanes, dlt.lanes);

  dlv->deleter(dlv);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportLifetimeLegacy)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  int deleter_calls = 0;
  auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 8);

  {
    tensor_t<TestType, 1> t;
    make_tensor(t, dl);
    ASSERT_EQ(deleter_calls, 0);

    {
      auto t_copy = t;
      ASSERT_EQ(deleter_calls, 0);
      ASSERT_EQ(t_copy.Data(), t.Data());
    }

    ASSERT_EQ(deleter_calls, 0);
  }

  ASSERT_EQ(deleter_calls, 1);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportLifetimeVersioned)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  int deleter_calls = 0;
  auto *dl = MakeVersionedManagedTensorForOwningImportTest<TestType>(&deleter_calls, 8);

  {
    tensor_t<TestType, 1> t;
    make_tensor(t, dl);
    ASSERT_EQ(deleter_calls, 0);

    {
      auto t_copy = t;
      ASSERT_EQ(deleter_calls, 0);
      ASSERT_EQ(t_copy.Data(), t.Data());
    }

    ASSERT_EQ(deleter_calls, 0);
  }

  ASSERT_EQ(deleter_calls, 1);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportByteOffset)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  int deleter_calls = 0;
  auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 10);
  auto *ctx = static_cast<DLPackOwningImportContext<TestType> *>(dl->manager_ctx);
  ctx->shape[0] = 9;
  dl->dl_tensor.byte_offset = sizeof(TestType);

  {
    tensor_t<TestType, 1> t;
    make_tensor(t, dl);
    ASSERT_EQ(deleter_calls, 0);

    auto *expected_ptr =
        reinterpret_cast<TestType *>(reinterpret_cast<char *>(dl->dl_tensor.data) + sizeof(TestType));
    ASSERT_EQ(t.Data(), expected_ptr);
    ASSERT_EQ(t.Size(0), 9);
  }

  ASSERT_EQ(deleter_calls, 1);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportRejectsInvalidShapeMetadata)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  {
    int deleter_calls = 0;
    auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    auto *ctx = static_cast<DLPackOwningImportContext<TestType> *>(dl->manager_ctx);
    ctx->shape[0] = -1;

    tensor_t<TestType, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  {
    int deleter_calls = 0;
    auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    auto *ctx = static_cast<DLPackOwningImportContext<TestType> *>(dl->manager_ctx);
    ctx->shape[0] = 0;

    tensor_t<TestType, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  {
    int deleter_calls = 0;
    auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    SetManagedTensorShapeAndStrides<TestType>(
        dl, {std::numeric_limits<index_t>::max(), 2}, {2, 1});

    tensor_t<TestType, 2> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportRejectsInvalidDataPointerMetadata)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  {
    int deleter_calls = 0;
    auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    auto *ctx = static_cast<DLPackOwningImportContext<TestType> *>(dl->manager_ctx);
    delete[] ctx->data;
    ctx->data = nullptr;
    dl->dl_tensor.data = nullptr;

    tensor_t<TestType, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  {
    int deleter_calls = 0;
    auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    dl->dl_tensor.byte_offset = 1;

    tensor_t<TestType, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportRejectsInvalidStrideMetadata)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  {
    int deleter_calls = 0;
    auto *dl = MakeVersionedManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    auto *ctx = static_cast<DLPackVersionedOwningImportContext<TestType> *>(dl->manager_ctx);
    ctx->strides[0] = -1;

    tensor_t<TestType, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  {
    int deleter_calls = 0;
    auto *dl = MakeVersionedManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    auto *ctx = static_cast<DLPackVersionedOwningImportContext<TestType> *>(dl->manager_ctx);
    ctx->strides[0] = 0;

    tensor_t<TestType, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  {
    int deleter_calls = 0;
    auto *dl = MakeVersionedManagedTensorForOwningImportTest<TestType>(&deleter_calls, 4);
    SetVersionedManagedTensorShapeAndStrides<TestType>(
        dl, {2, 2}, {std::numeric_limits<index_t>::max(), 1});

    tensor_t<TestType, 2> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
    ASSERT_EQ(deleter_calls, 1);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportUsesStridedStorageSpan)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  int deleter_calls = 0;
  auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 7);
  auto *ctx = static_cast<DLPackOwningImportContext<TestType> *>(dl->manager_ctx);
  for (int i = 0; i < 7; i++) {
    ctx->data[i] = static_cast<TestType>(i);
  }
  SetManagedTensorShapeAndStrides<TestType>(dl, {2, 3}, {4, 1});

  {
    tensor_t<TestType, 2> t;
    make_tensor(t, dl);
    ASSERT_EQ(deleter_calls, 0);
    ASSERT_EQ(t.TotalSize(), 6);
    ASSERT_EQ(t.GetStorage().size(), 7);
    ASSERT_EQ(static_cast<double>(t(1, 2)), static_cast<double>(ctx->data[6]));
  }

  ASSERT_EQ(deleter_calls, 1);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, OwningImportNullStridesContiguous)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  int deleter_calls = 0;
  auto *dl = MakeManagedTensorForOwningImportTest<TestType>(&deleter_calls, 6);
  auto *ctx = static_cast<DLPackOwningImportContext<TestType> *>(dl->manager_ctx);
  for (int i = 0; i < 6; i++) {
    ctx->data[i] = static_cast<TestType>(i);
  }

  delete[] ctx->shape;
  delete[] ctx->strides;
  ctx->shape = new int64_t[2]{2, 3};
  ctx->strides = nullptr; // Legacy DLPack contiguous indicator.
  dl->dl_tensor.ndim = 2;
  dl->dl_tensor.shape = ctx->shape;
  dl->dl_tensor.strides = ctx->strides;

  {
    tensor_t<TestType, 2> t;
    make_tensor(t, dl);
    ASSERT_EQ(deleter_calls, 0);
    ASSERT_EQ(t.Size(0), 2);
    ASSERT_EQ(t.Size(1), 3);
    ASSERT_EQ(t.Stride(0), 3);
    ASSERT_EQ(t.Stride(1), 1);
    ASSERT_EQ(static_cast<double>(t(1, 2)), static_cast<double>(ctx->data[5]));
  }

  ASSERT_EQ(deleter_calls, 1);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(DLPackTestsFloatNonComplex, VersionedReadOnlyRequiresConstType)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  int mutable_deleter_calls = 0;
  int const_deleter_calls = 0;

  auto *dl_mutable = MakeVersionedManagedTensorForOwningImportTest<TestType>(
      &mutable_deleter_calls, 8, DLPACK_FLAG_BITMASK_READ_ONLY);
  {
    tensor_t<TestType, 1> t;
    ASSERT_THROW({ make_tensor(t, dl_mutable); }, matx::detail::matxException);
  }
  ASSERT_EQ(mutable_deleter_calls, 1);

  auto *dl_const = MakeVersionedManagedTensorForOwningImportTest<TestType>(
      &const_deleter_calls, 8, DLPACK_FLAG_BITMASK_READ_ONLY);
  {
    tensor_t<const TestType, 1> t_const;
    make_tensor(t_const, dl_const);
    ASSERT_EQ(const_deleter_calls, 0);
  }
  ASSERT_EQ(const_deleter_calls, 1);

  MATX_EXIT_HANDLER();
}

TEST(DLPackVectorTests, ExportVectorLanesLegacyAndVersioned)
{
  MATX_ENTER_HANDLER();

  auto t = make_tensor<float4>({4});

  auto *dl = t.ToDlPack();
  ASSERT_EQ(dl->dl_tensor.dtype.code, kDLFloat);
  ASSERT_EQ(dl->dl_tensor.dtype.bits, 32);
  ASSERT_EQ(dl->dl_tensor.dtype.lanes, 4);
  dl->deleter(dl);

  auto *dlv = t.ToDlPackVersioned();
  ASSERT_EQ(dlv->dl_tensor.dtype.code, kDLFloat);
  ASSERT_EQ(dlv->dl_tensor.dtype.bits, 32);
  ASSERT_EQ(dlv->dl_tensor.dtype.lanes, 4);
  dlv->deleter(dlv);

  MATX_EXIT_HANDLER();
}

TEST(DLPackVectorTests, ImportVectorLaneMatchLegacy)
{
  MATX_ENTER_HANDLER();

  int deleter_calls = 0;
  auto *dl = MakeManagedTensorForOwningImportTest<float4>(&deleter_calls, 8);
  {
    tensor_t<float4, 1> t;
    make_tensor(t, dl);
    ASSERT_EQ(deleter_calls, 0);
  }
  ASSERT_EQ(deleter_calls, 1);

  MATX_EXIT_HANDLER();
}

TEST(DLPackVectorTests, ImportVectorLaneMatchVersioned)
{
  MATX_ENTER_HANDLER();

  int deleter_calls = 0;
  auto *dl = MakeVersionedManagedTensorForOwningImportTest<float4>(&deleter_calls, 8);
  {
    tensor_t<float4, 1> t;
    make_tensor(t, dl);
    ASSERT_EQ(deleter_calls, 0);
  }
  ASSERT_EQ(deleter_calls, 1);

  MATX_EXIT_HANDLER();
}

TEST(DLPackVectorTests, ImportVectorLaneMismatchThrows)
{
  MATX_ENTER_HANDLER();

  int deleter_calls = 0;
  auto *dl = MakeManagedTensorForOwningImportTest<float4>(&deleter_calls, 8);
  dl->dl_tensor.dtype.lanes = 2;
  {
    tensor_t<float4, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
  }
  ASSERT_EQ(deleter_calls, 1);

  MATX_EXIT_HANDLER();
}

TEST(DLPackVectorTests, ImportVectorBaseTypeMismatchThrows)
{
  MATX_ENTER_HANDLER();

  int deleter_calls = 0;
  auto *dl = MakeManagedTensorForOwningImportTest<float4>(&deleter_calls, 8);
  dl->dl_tensor.dtype.code = kDLInt;
  {
    tensor_t<float4, 1> t;
    ASSERT_THROW({ make_tensor(t, dl); }, matx::detail::matxException);
  }
  ASSERT_EQ(deleter_calls, 1);

  MATX_EXIT_HANDLER();
}
