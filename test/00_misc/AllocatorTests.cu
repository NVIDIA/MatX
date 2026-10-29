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

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"
#include <iostream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <unordered_map>

using namespace matx;

class AsyncDeviceAllocatorTest : public ::testing::TestWithParam<bool> {};

TEST_P(AsyncDeviceAllocatorTest, DeviceCapabilityAndStreamLifetime) {
    MATX_ENTER_HANDLER();

    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    int pools_supported = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, device), cudaSuccess);
    const auto expected_kind = pools_supported ? MATX_ASYNC_DEVICE_MEMORY : MATX_DEVICE_MEMORY;

    cudaStream_t nondefault_stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&nondefault_stream, cudaStreamNonBlocking), cudaSuccess);
    for (cudaStream_t stream : {cudaStream_t{}, nondefault_stream}) {
        size_t initial, total, maximum;
        matxGetMemoryStats(&initial, &total, &maximum);
        void *ptr = nullptr;
        constexpr size_t bytes = 1024;
        matxAlloc(&ptr, bytes, MATX_ASYNC_DEVICE_MEMORY, stream);
        EXPECT_NE(ptr, nullptr);
        EXPECT_EQ(GetPointerKind(ptr), expected_kind);

        size_t current;
        matxGetMemoryStats(&current, &total, &maximum);
        EXPECT_EQ(current, initial + bytes);

        // Exercise the allocation on its stream, then free it with work pending.
        // The synchronous fallback must also safely handle a nonblocking stream.
        EXPECT_EQ(cudaMemsetAsync(ptr, 0x5a, bytes, stream), cudaSuccess);
        unsigned char result = 0;
        EXPECT_EQ(cudaMemcpyAsync(&result, ptr, sizeof(result), cudaMemcpyDeviceToHost, stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(result, 0x5a);
        EXPECT_EQ(cudaMemsetAsync(ptr, 0, bytes, stream), cudaSuccess);
        if (GetParam()) {
            matxFree(ptr, stream);
        } else {
            matxFree(ptr);
        }
        EXPECT_FALSE(IsAllocated(ptr));
        EXPECT_EQ(GetPointerKind(ptr), MATX_INVALID_MEMORY);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        matxGetMemoryStats(&current, &total, &maximum);
        EXPECT_EQ(current, initial);
    }
    EXPECT_EQ(cudaStreamDestroy(nondefault_stream), cudaSuccess);

    MATX_EXIT_HANDLER();
}

TEST_P(AsyncDeviceAllocatorTest, UnsupportedMemoryPoolsFallback) {
    MATX_ENTER_HANDLER();

    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    cudaStream_t nondefault_stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&nondefault_stream, cudaStreamNonBlocking), cudaSuccess);
    for (cudaStream_t stream : {cudaStream_t{}, nondefault_stream}) {
        size_t initial, total, maximum;
        matxGetMemoryStats(&initial, &total, &maximum);
        void *ptr = nullptr;
        int queries = 0;
        // Use the production allocation implementation, forcing only the
        // capability result. This must exercise cudaMalloc even on pooled GPUs.
        GetAllocMap().allocate_internal(&ptr, 1024, MATX_ASYNC_DEVICE_MEMORY, stream,
            [&](int queried_device) {
                EXPECT_EQ(queried_device, device);
                ++queries;
                return false;
            });
        EXPECT_EQ(queries, 1);
        EXPECT_NE(ptr, nullptr);
        EXPECT_EQ(GetPointerKind(ptr), MATX_DEVICE_MEMORY);

        EXPECT_EQ(cudaMemsetAsync(ptr, 0x5a, 1024, stream), cudaSuccess);
        if (GetParam()) {
            matxFree(ptr, stream);
        } else {
            matxFree(ptr);
        }
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_FALSE(IsAllocated(ptr));
        EXPECT_EQ(GetPointerKind(ptr), MATX_INVALID_MEMORY);
        size_t current;
        matxGetMemoryStats(&current, &total, &maximum);
        EXPECT_EQ(current, initial);
    }
    EXPECT_EQ(cudaStreamDestroy(nondefault_stream), cudaSuccess);

    MATX_EXIT_HANDLER();
}

INSTANTIATE_TEST_SUITE_P(FreeOverloads, AsyncDeviceAllocatorTest, ::testing::Bool());

TEST(DeviceAllocatorTest, MemoryPoolSupportQueriedOnStreamDevice) {
    MATX_ENTER_HANDLER();
    if (CudaDeviceCount() < 2) {
        GTEST_SKIP() << "Requires two CUDA devices";
    }

    const auto restore = detail::CudaDeviceGuard::RestoreCurrent();
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    cudaStream_t stream_on_device1;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_on_device1, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    auto allocate_on_device1 = [&]() {
        // cudaMallocAsync allocates from the stream's device, so pool support
        // must be checked there rather than on the current device.
        void *ptr = nullptr;
        int queried_device = -1;
        GetAllocMap().allocate_internal(&ptr, 1024, MATX_ASYNC_DEVICE_MEMORY, stream_on_device1,
            [&](int device) {
                queried_device = device;
                return false;
            });
        EXPECT_EQ(queried_device, 1);

        // The cudaMalloc fallback must also land on the stream's device.
        cudaPointerAttributes attrs{};
        EXPECT_EQ(cudaPointerGetAttributes(&attrs, ptr), cudaSuccess);
        EXPECT_EQ(attrs.device, 1);
        matxFree(ptr);
    };

    // Afterwards the caller's context must be current again: device 0's primary
    // context, or a context the application created on device 0, which a device
    // switch back with cudaSetDevice would have replaced.
    const CUcontext primary = CurrentDriverContext();
    allocate_on_device1();
    EXPECT_EQ(CurrentDriverContext(), primary);
    std::thread([&]() {
        try {
            const CUcontext created = CreateDriverContext(0);
            allocate_on_device1();
            EXPECT_EQ(CurrentDriverContext(), created);
            EXPECT_EQ(cuCtxDestroy(created), CUDA_SUCCESS);
        } catch (const std::exception &e) {
            ADD_FAILURE() << e.what();
        }
    }).join();

    EXPECT_EQ(cudaStreamDestroy(stream_on_device1), cudaSuccess);
    MATX_EXIT_HANDLER();
}

TEST(DeviceAllocatorTest, FallbackSwitchesDeviceWithoutContext) {
    MATX_ENTER_HANDLER();
    if (CudaDeviceCount() < 2) {
        GTEST_SKIP() << "Requires two CUDA devices";
    }

    const auto restore = detail::CudaDeviceGuard::RestoreCurrent();
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    cudaStream_t stream_on_device1;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_on_device1, cudaStreamNonBlocking), cudaSuccess);

    // Driver API code can empty a thread's context stack after cudaSetDevice(1).
    // cudaGetDevice then still reports device 1, but the next runtime call runs
    // on device 0, so the fallback must switch devices rather than trust
    // cudaGetDevice to say device 1 is already current.
    std::thread([&]() {
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        ASSERT_EQ(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
        try {
            void *ptr = nullptr;
            GetAllocMap().allocate_internal(&ptr, 1024, MATX_ASYNC_DEVICE_MEMORY, stream_on_device1,
                [](int) { return false; });
            cudaPointerAttributes attrs{};
            EXPECT_EQ(cudaPointerGetAttributes(&attrs, ptr), cudaSuccess);
            EXPECT_EQ(attrs.device, 1);
            matxFree(ptr);
        } catch (const std::exception &e) {
            ADD_FAILURE() << e.what();
        }
    }).join();

    EXPECT_EQ(cudaStreamDestroy(stream_on_device1), cudaSuccess);
    MATX_EXIT_HANDLER();
}

namespace {
struct NamedLookup {
    const char *name;
    detail::StreamDeviceLookup lookup;
};

// The lookups a multi-GPU system makes: cuStreamGetCtx_v2 on 12.5+ drivers and
// cuStreamGetCtx before that. The single-GPU fast path is off so single-GPU
// systems reach them too. On drivers older than 12.5, both entries take the
// cuStreamGetCtx path.
std::vector<NamedLookup> MultiDeviceLookups() {
    auto lookup = detail::GetStreamDeviceLookup();
    lookup.single_device = false;
    std::vector<NamedLookup> lookups{{"installed driver", lookup}};
    lookup.get_ctx_v2 = nullptr;
    lookups.push_back({"cuStreamGetCtx", lookup});
    return lookups;
}
} // namespace

TEST(DeviceAllocatorTest, StreamDeviceLookupResolvesStreams) {
    MATX_ENTER_HANDLER();
    const int device_count = CudaDeviceCount();
    if (device_count == 0) {
        GTEST_SKIP() << "CUDA device required";
    }

    // cuGetProcAddress would return the two-argument cuStreamGetCtx from a
    // driver older than 12.5, so the three-argument version is only used on
    // newer drivers. Availability follows the driver, not the toolkit.
    int driver_version = 0;
    ASSERT_EQ(cuDriverGetVersion(&driver_version), CUDA_SUCCESS);
    EXPECT_EQ(detail::GetStreamDeviceLookup().get_ctx_v2 != nullptr, driver_version >= 12050)
        << "driver version " << driver_version;

    const cudaStream_t default_handles[] = {cudaStream_t{}, cudaStreamLegacy, cudaStreamPerThread};
    const auto lookups = MultiDeviceLookups();
    const auto restore = detail::CudaDeviceGuard::RestoreCurrent();
    std::vector<cudaStream_t> streams(static_cast<size_t>(device_count));
    for (int device = 0; device < device_count; ++device) {
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&streams[static_cast<size_t>(device)], cudaStreamNonBlocking),
                  cudaSuccess);
    }

    // A created stream belongs to the device it was created on, whichever
    // device is current. The default-stream handles name the current device.
    for (int current = device_count - 1; current >= 0; --current) {
        SCOPED_TRACE(current);
        ASSERT_EQ(cudaSetDevice(current), cudaSuccess);
        for (int device = 0; device < device_count; ++device) {
            const cudaStream_t stream = streams[static_cast<size_t>(device)];
            EXPECT_EQ(detail::GetStreamDevice(stream), device);
            for (const auto &l : lookups) {
                SCOPED_TRACE(l.name);
                EXPECT_EQ(detail::GetStreamDevice(stream, l.lookup), device);
            }
        }
        for (cudaStream_t stream : default_handles) {
            EXPECT_EQ(detail::GetStreamDevice(stream), current);
            for (const auto &l : lookups) {
                SCOPED_TRACE(l.name);
                EXPECT_EQ(detail::GetStreamDevice(stream, l.lookup), current);
            }
        }
    }

    // A new thread starts with no current context. Looking up a created stream
    // must not leave the stream's context current, and the default-stream
    // handles, which a driver lookup could not resolve, are on device 0.
    bool threw = false;
    CUcontext ctx_before = nullptr, ctx_after = nullptr;
    CUresult before = CUDA_ERROR_UNKNOWN, after = CUDA_ERROR_UNKNOWN;
    std::vector<int> expected, found;
    std::thread([&]() {
        try {
            before = cuCtxGetCurrent(&ctx_before);
            for (const auto &l : lookups) {
                for (int device = 0; device < device_count; ++device) {
                    expected.push_back(device);
                    found.push_back(detail::GetStreamDevice(streams[static_cast<size_t>(device)], l.lookup));
                }
            }
            after = cuCtxGetCurrent(&ctx_after);
            for (const auto &l : lookups) {
                for (cudaStream_t stream : default_handles) {
                    expected.push_back(0);
                    found.push_back(detail::GetStreamDevice(stream, l.lookup));
                }
            }
        } catch (...) {
            threw = true;
        }
    }).join();
    EXPECT_FALSE(threw);
    EXPECT_EQ(before, CUDA_SUCCESS);
    EXPECT_EQ(ctx_before, nullptr);
    EXPECT_EQ(after, CUDA_SUCCESS);
    EXPECT_EQ(ctx_after, nullptr);
    EXPECT_EQ(found, expected);

    for (cudaStream_t stream : streams) {
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
    MATX_EXIT_HANDLER();
}

#if CUDA_VERSION >= 12050
TEST(DeviceAllocatorTest, GreenContextStreamDevice) {
    MATX_ENTER_HANDLER();
    const int device_count = CudaDeviceCount();
    if (device_count == 0) {
        GTEST_SKIP() << "CUDA device required";
    }
    int driver_version = 0;
    ASSERT_EQ(cuDriverGetVersion(&driver_version), CUDA_SUCCESS);
    if (driver_version < 12050) {
        GTEST_SKIP() << "cuGreenCtxStreamCreate requires a 12.5 or newer driver";
    }

    const auto restore = detail::CudaDeviceGuard::RestoreCurrent();
    const int device = device_count - 1;
    CUdevResource sm_resource;
    ASSERT_EQ(cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM), CUDA_SUCCESS);
    CUdevResourceDesc desc;
    ASSERT_EQ(cuDevResourceGenerateDesc(&desc, &sm_resource, 1), CUDA_SUCCESS);
    CUgreenCtx green_ctx;
    ASSERT_EQ(cuGreenCtxCreate(&green_ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM), CUDA_SUCCESS);
    CUstream green_stream;
    ASSERT_EQ(cuGreenCtxStreamCreate(&green_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0), CUDA_SUCCESS);
    // On multi-GPU systems, make a different device current so the lookup
    // cannot pass by returning the current device.
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    // cuStreamGetCtx rejects green-context streams on 12.x drivers, so only the
    // cuStreamGetCtx_v2 lookup is checked.
    const auto stream = reinterpret_cast<cudaStream_t>(green_stream);
    for (const auto &l : MultiDeviceLookups()) {
        if (l.lookup.get_ctx_v2 == nullptr) {
            continue;
        }
        SCOPED_TRACE(l.name);
        EXPECT_EQ(detail::GetStreamDevice(stream, l.lookup), device);
    }
    EXPECT_EQ(cuStreamDestroy(green_stream), CUDA_SUCCESS);
    EXPECT_EQ(cuGreenCtxDestroy(green_ctx), CUDA_SUCCESS);
    MATX_EXIT_HANDLER();
}
#endif

TEST(DeviceAllocatorTest, StreamDeviceLookupKeepsGraphCapture) {
    MATX_ENTER_HANDLER();
    if (CudaDeviceCount() == 0) {
        GTEST_SKIP() << "CUDA device required";
    }

    // Allocations can happen while their stream is being captured into a CUDA
    // graph. A query that fails on a capturing stream invalidates the capture,
    // as cuStreamGetDevice does, so every lookup must leave the capture valid.
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    for (cudaStreamCaptureMode mode : {cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal,
                                       cudaStreamCaptureModeRelaxed}) {
        SCOPED_TRACE(static_cast<int>(mode));
        for (const auto &l : MultiDeviceLookups()) {
            SCOPED_TRACE(l.name);
            ASSERT_EQ(cudaStreamBeginCapture(stream, mode), cudaSuccess);
            int found = -1;
            bool threw = false;
            try {
                found = detail::GetStreamDevice(stream, l.lookup);
            } catch (...) {
                threw = true;
            }
            cudaGraph_t graph = nullptr;
            EXPECT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
            EXPECT_FALSE(threw);
            EXPECT_EQ(found, device);
            if (graph != nullptr) {
                EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
            }
        }
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    MATX_EXIT_HANDLER();
}

TEST(DeviceAllocatorTest, AsyncAllocationDuringGraphCapture) {
    MATX_ENTER_HANDLER();
    if (CudaDeviceCount() == 0) {
        GTEST_SKIP() << "CUDA device required";
    }
    if (!detail::GetCurrentDeviceAttributes().memory_pools_supported) {
        GTEST_SKIP() << "The cudaMalloc fallback cannot be captured";
    }

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    // Cover matxAlloc, and the stream-device lookup that allocations take when
    // some device lacks memory pools.
    for (bool use_matx_alloc : {true, false}) {
        SCOPED_TRACE(use_matx_alloc);
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        bool threw = false;
        try {
            void *ptr = nullptr;
            if (use_matx_alloc) {
                matxAlloc(&ptr, 1024, MATX_ASYNC_DEVICE_MEMORY, stream);
            }
            else {
                GetAllocMap().allocate_internal(&ptr, 1024, MATX_ASYNC_DEVICE_MEMORY, stream,
                    [](int) { return true; }, false);
            }
            EXPECT_EQ(cudaMemsetAsync(ptr, 0x5a, 1024, stream), cudaSuccess);
            matxFree(ptr, stream);
        } catch (...) {
            threw = true;
        }
        cudaGraph_t graph = nullptr;
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
        EXPECT_FALSE(threw);

        // Launch the graph to check that the captured allocation is usable.
        cudaGraphExec_t exec = nullptr;
        ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
        EXPECT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
        EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    MATX_EXIT_HANDLER();
}

namespace {
// Stand-ins for TryGetDeviceAttributes in the all-devices memory pool check.
cudaError_t AllDevicesHavePools(int, const detail::DeviceAttributes *&attrs) {
    static const detail::DeviceAttributes with_pools = [] {
        detail::DeviceAttributes a{};
        a.memory_pools_supported = true;
        return a;
    }();
    attrs = &with_pools;
    return cudaSuccess;
}

cudaError_t NoDeviceHasPools(int, const detail::DeviceAttributes *&attrs) {
    static const detail::DeviceAttributes without_pools{};
    attrs = &without_pools;
    return cudaSuccess;
}

// Fails the way a real query does, leaving an error for cudaGetLastError.
cudaError_t AttributeQueryFails(int, const detail::DeviceAttributes *&) {
    int value = 0;
    return cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, -1);
}
} // namespace

TEST(DeviceAllocatorTest, AllDevicesMemoryPoolCheck) {
    MATX_ENTER_HANDLER();
    const int device_count = CudaDeviceCount();
    if (device_count == 0) {
        GTEST_SKIP() << "CUDA device required";
    }

    // When every device has memory pools, async allocations skip finding the
    // stream's device.
    bool expected = true;
    for (int device = 0; device < device_count; ++device) {
        int supported = 0;
        ASSERT_EQ(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device), cudaSuccess);
        expected = expected && supported != 0;
    }
    EXPECT_EQ(detail::AllDevicesSupportMemoryPools(), expected);
    EXPECT_TRUE(detail::QueryAllDevicesSupportMemoryPools(&AllDevicesHavePools));
    EXPECT_FALSE(detail::QueryAllDevicesSupportMemoryPools(&NoDeviceHasPools));

    // A device whose attributes cannot be read sends allocations to the
    // per-stream check rather than failing them all, and its error is cleared.
    const detail::DeviceAttributes *unused = nullptr;
    ASSERT_NE(AttributeQueryFails(0, unused), cudaSuccess);
    ASSERT_NE(cudaGetLastError(), cudaSuccess);
    bool all_supported = true;
    EXPECT_NO_THROW(all_supported = detail::QueryAllDevicesSupportMemoryPools(&AttributeQueryFails));
    EXPECT_FALSE(all_supported);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    MATX_EXIT_HANDLER();
}

#if GTEST_HAS_DEATH_TEST && !defined(_WIN32)
TEST(DeviceAllocatorTest, HostPathsWithoutVisibleDevices) {
    // Host allocations and host execution must not need a GPU, and
    // TryGetComputeCapability must report 0 rather than throw. The threadsafe
    // death test style runs the test binary again in a new process, and only
    // that process hides the devices. CUDA has not started there yet, so the
    // empty list applies.
    const auto previous_style = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT({
        if (setenv("CUDA_VISIBLE_DEVICES", "", 1) != 0) {
            std::exit(2);
        }
        // A visible device means CUDA started before the list was hidden, and
        // the checks below would not show anything.
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
            std::exit(3);
        }
        if (detail::TryGetComputeCapability() != 0) {
            std::exit(4);
        }
        void *ptr = nullptr;
        matxAlloc(&ptr, 64, MATX_HOST_MALLOC_MEMORY);
        matxFree(ptr);
        auto t = make_tensor<float>({16}, MATX_HOST_MALLOC_MEMORY);
        (t = ones<float>({16})).run(SingleThreadedHostExecutor{});
        std::exit(t(3) == 1.0f ? 0 : 1);
    }, testing::ExitedWithCode(0), "");
    GTEST_FLAG_SET(death_test_style, previous_style);
}
#endif

TEST(DeviceAllocatorTest, CachedMemoryPoolSupportAcrossDevices) {
    MATX_ENTER_HANDLER();

    const auto restore = detail::CudaDeviceGuard::RestoreCurrent();

    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    std::vector<int> pools_supported(static_cast<size_t>(device_count));
    for (int device = 0; device < device_count; ++device) {
        ASSERT_EQ(cudaDeviceGetAttribute(&pools_supported[static_cast<size_t>(device)],
            cudaDevAttrMemoryPoolsSupported, device), cudaSuccess);
    }

    // Revisit devices in reverse order to exercise cached capabilities after
    // switching devices, including systems with different device capabilities.
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < device_count; ++i) {
            const int device = pass == 0 ? i : device_count - 1 - i;
            ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
            void *ptr = nullptr;
            matxAlloc(&ptr, 1024, MATX_ASYNC_DEVICE_MEMORY);
            EXPECT_EQ(GetPointerKind(ptr), pools_supported[static_cast<size_t>(device)]
                ? MATX_ASYNC_DEVICE_MEMORY : MATX_DEVICE_MEMORY);
            matxFree(ptr);
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
    }

    MATX_EXIT_HANDLER();
}

// Example 1: Simple custom allocator - NO INHERITANCE REQUIRED!
class TrackedAllocator {
private:
    static inline std::unordered_map<void*, size_t> allocations_;
    static inline size_t total_allocated_ = 0;
    
public:
    // Duck-typed interface - just implement these methods!
    void* allocate(size_t bytes) {
        void* ptr = std::malloc(bytes);
        allocations_[ptr] = bytes;
        total_allocated_ += bytes;
        std::cout << "TrackedAllocator::allocate(" << bytes << ") -> " << ptr << std::endl;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_allocated_ -= it->second;
            allocations_.erase(it);
        }
        std::cout << "TrackedAllocator::deallocate(" << ptr << ", " << bytes << ")" << std::endl;
        std::free(ptr);
    }
    
    // Utility methods for testing
    static size_t getTotalAllocated() { return total_allocated_; }
    static size_t getActiveAllocations() { return allocations_.size(); }
    static void reset() { 
        allocations_.clear(); 
        total_allocated_ = 0; 
    }
};

// Example 2: Pool allocator - also NO INHERITANCE!
class PoolAllocator {
private:
    std::vector<char> pool_;
    size_t offset_ = 0;
    
public:
    PoolAllocator(size_t pool_size) : pool_(pool_size) {
        std::cout << "PoolAllocator created with " << pool_size << " bytes" << std::endl;
    }
    
    // Duck-typed interface
    void* allocate(size_t bytes) {
        if (offset_ + bytes > pool_.size()) {
            throw std::bad_alloc();
        }
        void* ptr = pool_.data() + offset_;
        offset_ += bytes;
        std::cout << "PoolAllocator::allocate(" << bytes << ") -> " << ptr 
                  << " (offset now " << offset_ << ")" << std::endl;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        // Simple pool - just log the deallocation
        std::cout << "PoolAllocator::deallocate(" << ptr << ", " << bytes << ")" << std::endl;
    }
    
    size_t getBytesUsed() const { return offset_; }
};

// Example 3: Debug allocator with alignment
struct AlignedDebugAllocator {
    size_t alignment_;
    
    AlignedDebugAllocator(size_t alignment = 64) : alignment_(alignment) {}
    
    // Duck-typed interface
    void* allocate(size_t bytes) {
        size_t aligned_bytes = (bytes + alignment_ - 1) & ~(alignment_ - 1);
#ifdef _WIN32
        void* ptr = _aligned_malloc(aligned_bytes, alignment_);
#else
        void* ptr = std::aligned_alloc(alignment_, aligned_bytes);
#endif
        std::cout << "AlignedDebugAllocator::allocate(" << bytes 
                  << ") aligned to " << aligned_bytes << " -> " << ptr << std::endl;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        std::cout << "AlignedDebugAllocator::deallocate(" << ptr << ", " << bytes << ")" << std::endl;
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

class DuckTypingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        TrackedAllocator::reset();
    }
    
    void TearDown() override {
        TrackedAllocator::reset();
    }
};

TEST_F(DuckTypingAllocatorTest, TrackedAllocatorObject) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing TrackedAllocator with Storage (object) ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Use allocator object directly - no inheritance required!
        Storage<float> storage(1000, alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 1000);
        EXPECT_GT(TrackedAllocator::getTotalAllocated(), 0);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Test the storage actually works
        storage.data()[0] = 42.0f;
        storage.data()[999] = 99.0f;
        EXPECT_EQ(storage.data()[0], 42.0f);
        EXPECT_EQ(storage.data()[999], 99.0f);
    }
    
    // After destruction, memory should be freed
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, TrackedAllocatorPointer) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing TrackedAllocator with Storage (pointer) ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Use allocator pointer - also works!
        Storage<double> storage(500, &alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 500);
        EXPECT_GT(TrackedAllocator::getTotalAllocated(), 0);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, FactoryFunction) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing TrackedAllocator with make_owning_storage ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Use with factory function
        auto storage = make_owning_storage<int>(2000, alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 2000);
        EXPECT_GT(TrackedAllocator::getTotalAllocated(), 0);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, PoolAllocator) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing PoolAllocator ===" << std::endl;
    
    PoolAllocator pool(10000);  // 10KB pool
    
    {
        Storage<float> storage1(100, pool);  // 400 bytes
        Storage<int> storage2(200, pool);    // 800 bytes
        
        EXPECT_NE(storage1.data(), nullptr);
        EXPECT_NE(storage2.data(), nullptr);
        EXPECT_EQ(storage1.size(), 100);
        EXPECT_EQ(storage2.size(), 200);
        
        // Pool should have allocated bytes
        EXPECT_GT(pool.getBytesUsed(), 0);
        
        // Test the storages work
        storage1.data()[0] = 3.14f;
        storage2.data()[0] = 42;
        EXPECT_EQ(storage1.data()[0], 3.14f);
        EXPECT_EQ(storage2.data()[0], 42);
    }
    
    std::cout << "Pool used " << pool.getBytesUsed() << " bytes total" << std::endl;
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, AlignedAllocator) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing AlignedDebugAllocator ===" << std::endl;
    
    AlignedDebugAllocator aligned_alloc(128);  // 128-byte alignment
    
    {
        Storage<double> storage(64, aligned_alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 64);
        
        // Check alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(storage.data());
        EXPECT_EQ(addr % 128, 0);  // Should be 128-byte aligned
        
        std::cout << "Storage pointer: " << storage.data() 
                  << " (aligned to 128 bytes)" << std::endl;
    }
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, SharedOwnership) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing Shared Ownership ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        Storage<int> storage1(100, alloc);
        EXPECT_EQ(storage1.use_count(), 1);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        {
            Storage<int> storage2 = storage1;  // Copy constructor
            EXPECT_EQ(storage1.use_count(), 2);
            EXPECT_EQ(storage2.use_count(), 2);
            EXPECT_EQ(storage1.data(), storage2.data());  // Same data
            EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);  // Still one allocation
        }
        
        // After storage2 is destroyed, storage1 should still work
        EXPECT_EQ(storage1.use_count(), 1);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        storage1.data()[0] = 42;
        EXPECT_EQ(storage1.data()[0], 42);
    }
    
    // Now memory should be freed
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, CompareWithDefault) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Comparing Custom vs Default Allocator ===" << std::endl;
    
    TrackedAllocator custom_alloc;
    
    {
        Storage<float> default_storage(1000);           // Default allocator
        Storage<float> custom_storage(1000, custom_alloc);  // Custom allocator
        
        EXPECT_NE(default_storage.data(), nullptr);
        EXPECT_NE(custom_storage.data(), nullptr);
        EXPECT_EQ(default_storage.size(), 1000);
        EXPECT_EQ(custom_storage.size(), 1000);
        
        // Only custom allocator should be tracked
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Both should work the same way
        default_storage.data()[0] = 1.0f;
        custom_storage.data()[0] = 2.0f;
        EXPECT_EQ(default_storage.data()[0], 1.0f);
        EXPECT_EQ(custom_storage.data()[0], 2.0f);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

// Demonstrate that it works with new make_tensor API!
TEST_F(DuckTypingAllocatorTest, WithTensorAPI) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing with New make_tensor API ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // NEW: Use make_tensor directly with custom allocator - much simpler!
        auto tensor = make_tensor<float>({1000}, alloc);
        
        EXPECT_NE(tensor.Data(), nullptr);
        EXPECT_EQ(tensor.Size(0), 1000);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Test tensor operations
        tensor(0) = 3.14f;
        tensor(999) = 2.71f;
        EXPECT_EQ(tensor(0), 3.14f);
        EXPECT_EQ(tensor(999), 2.71f);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

// Test all variants of new make_tensor with custom allocators
TEST_F(DuckTypingAllocatorTest, MakeTensorAllVariants) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing All make_tensor Variants with Custom Allocators ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        std::cout << "Testing C-array shape variant..." << std::endl;
        index_t shape[2] = {10, 20};
        auto tensor1 = make_tensor<double>(shape, alloc);
        
        EXPECT_NE(tensor1.Data(), nullptr);
        EXPECT_EQ(tensor1.Size(0), 10);
        EXPECT_EQ(tensor1.Size(1), 20);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        std::cout << "Testing shape type variant..." << std::endl;
        auto tensor2 = make_tensor<float>({5, 8}, alloc);
        
        EXPECT_NE(tensor2.Data(), nullptr);
        EXPECT_EQ(tensor2.Size(0), 5);
        EXPECT_EQ(tensor2.Size(1), 8);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 2);
        
        std::cout << "Testing 1D tensor with C-array..." << std::endl;
        index_t shape1d[1] = {100};
        auto tensor3 = make_tensor<int>(shape1d, alloc);
        
        EXPECT_NE(tensor3.Data(), nullptr);
        EXPECT_EQ(tensor3.Size(0), 100);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 3);
        
        std::cout << "Testing 3D tensor with initializer list..." << std::endl;
        auto tensor4 = make_tensor<char>({2, 3, 4}, alloc);
        
        EXPECT_NE(tensor4.Data(), nullptr);
        EXPECT_EQ(tensor4.Size(0), 2);
        EXPECT_EQ(tensor4.Size(1), 3);
        EXPECT_EQ(tensor4.Size(2), 4);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 4);
        
        // Test that all tensors work correctly
        tensor1(0, 0) = 1.23;
        tensor2(0, 0) = 4.56f;
        tensor3(0) = 789;
        tensor4(0, 0, 0) = 'A';
        
        EXPECT_DOUBLE_EQ(tensor1(0, 0), 1.23);
        EXPECT_FLOAT_EQ(tensor2(0, 0), 4.56f);
        EXPECT_EQ(tensor3(0), 789);
        EXPECT_EQ(tensor4(0, 0, 0), 'A');
    }
    
    // All tensors should be freed
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

// Test with PMR-style allocator pointer
TEST_F(DuckTypingAllocatorTest, MakeTensorWithAllocatorPointer) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing make_tensor with Allocator Pointer ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Test passing allocator pointer (like PMR usage)
        auto tensor1 = make_tensor<float>({50, 60}, &alloc);
        
        EXPECT_NE(tensor1.Data(), nullptr);
        EXPECT_EQ(tensor1.Size(0), 50);
        EXPECT_EQ(tensor1.Size(1), 60);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Test with allocator pointer and return by value
        auto tensor2 = make_tensor<double>({1000}, &alloc);
        
        EXPECT_NE(tensor2.Data(), nullptr);
        EXPECT_EQ(tensor2.Size(0), 1000);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 2);
        
        // Verify they work
        tensor1(0, 0) = 3.14f;
        tensor2(0) = 2.718;
        EXPECT_FLOAT_EQ(tensor1(0, 0), 3.14f);
        EXPECT_DOUBLE_EQ(tensor2(0), 2.718);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}
