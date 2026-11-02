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
#include "utilities.h"

#include "gtest/gtest.h"

#include <latch>
#include <thread>
#include <vector>

using namespace matx;

namespace {

int RawAttr(cudaDeviceAttr attr, int device)
{
  int value = 0;
  EXPECT_EQ(cudaDeviceGetAttribute(&value, attr, device), cudaSuccess);
  return value;
}

// Attribute that QueryFailingOne reports as failed. Every other query goes to CUDA.
cudaDeviceAttr failing_attr = cudaDevAttrMaxThreadsPerBlock;

cudaError_t QueryFailingOne(int *value, cudaDeviceAttr attr, int device)
{
  if (attr == failing_attr) {
    return cudaErrorInvalidValue;
  }
  return cudaDeviceGetAttribute(value, attr, device);
}

// Attributes with only the compute capability set, so an EPT query can target
// an architecture without that GPU present.
detail::DeviceAttributes AttrsWithComputeCapability(int major, int minor)
{
  detail::DeviceAttributes attrs{};
  attrs.cc_major = major;
  attrs.cc_minor = minor;
  return attrs;
}

} // namespace

TEST(DeviceAttributeCacheTest, MatchesRawQueriesOnEveryDevice)
{
  MATX_ENTER_HANDLER();
  const int device_count = CudaDeviceCount();
  if (device_count == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  EXPECT_EQ(detail::GetDeviceCount(), device_count);
  for (int device = 0; device < device_count; ++device) {
    SCOPED_TRACE(device);
    const auto &attrs = detail::GetDeviceAttributes(device);
    // Later lookups return the cached entry.
    EXPECT_EQ(&detail::GetDeviceAttributes(device), &attrs);
    EXPECT_EQ(attrs.cc_major, RawAttr(cudaDevAttrComputeCapabilityMajor, device));
    EXPECT_EQ(attrs.cc_minor, RawAttr(cudaDevAttrComputeCapabilityMinor, device));
    EXPECT_EQ(attrs.max_smem_per_block, RawAttr(cudaDevAttrMaxSharedMemoryPerBlock, device));
    EXPECT_EQ(attrs.max_smem_per_block_optin, RawAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    EXPECT_EQ(attrs.max_smem_per_sm, RawAttr(cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    EXPECT_EQ(attrs.max_regs_per_sm, RawAttr(cudaDevAttrMaxRegistersPerMultiprocessor, device));
    EXPECT_EQ(attrs.l2_cache_bytes, RawAttr(cudaDevAttrL2CacheSize, device));
    EXPECT_EQ(attrs.memory_bus_width_bits, RawAttr(cudaDevAttrGlobalMemoryBusWidth, device));
    EXPECT_EQ(attrs.fp32_to_fp64_perf_ratio, RawAttr(cudaDevAttrSingleToDoublePrecisionPerfRatio, device));
    EXPECT_EQ(attrs.concurrent_managed_access, RawAttr(cudaDevAttrConcurrentManagedAccess, device) != 0);
    EXPECT_EQ(attrs.memory_pools_supported, RawAttr(cudaDevAttrMemoryPoolsSupported, device) != 0);
    EXPECT_EQ(attrs.ComputeCapability(), attrs.cc_major * 100 + attrs.cc_minor * 10);
  }
  MATX_EXIT_HANDLER();
}

TEST(DeviceAttributeCacheTest, CurrentDeviceFollowsSetDevice)
{
  MATX_ENTER_HANDLER();
  const int device_count = CudaDeviceCount();
  if (device_count == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  const auto restore = detail::CudaDeviceGuard::RestoreCurrent();
#ifdef MATX_EN_MATHDX
  // Operators build these helpers in their constructors, so build them with
  // device 0 current and check that later use follows the current device.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  detail::cuBLASDxHelper<float> gemm;
  gemm.set_m(16);
  gemm.set_n(16);
  gemm.set_k(16);
  gemm.set_is_complex(false);
  detail::cuFFTDxHelper<cuda::std::complex<float>> fft;
  detail::cuFFTDx2DHelper<cuda::std::complex<float>> fft2;
  detail::cuSolverDxHelper<float> solver;
#endif

  // Everything that targets the current device must follow cudaSetDevice.
  // Visit devices in reverse so a stale lookup would be caught on multi-GPU
  // systems. With one device, the lookup skips cudaGetDevice and must still
  // return device 0's entry.
  for (int device = device_count - 1; device >= 0; --device) {
    SCOPED_TRACE(device);
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    const auto &attrs = detail::GetDeviceAttributes(device);
    const int cc = attrs.ComputeCapability();
    EXPECT_EQ(&detail::GetCurrentDeviceAttributes(), &attrs);
    EXPECT_EQ(detail::GetComputeCapabilityMajor(), RawAttr(cudaDevAttrComputeCapabilityMajor, device));
    EXPECT_EQ(detail::GetComputeCapabilityMinor(), RawAttr(cudaDevAttrComputeCapabilityMinor, device));
    EXPECT_EQ(detail::GetComputeCapability(), cc);
#ifdef MATX_EN_JIT
    EXPECT_EQ(detail::get_runtime_cuda_arch(), std::to_string(attrs.cc_major * 10 + attrs.cc_minor));
#endif
#ifdef MATX_EN_MATHDX
    EXPECT_EQ(gemm.get_cc(), cc);
    EXPECT_NE(gemm.GetSymbolName().find("_CC" + std::to_string(cc) + "_"), std::string::npos);
    EXPECT_EQ(fft.get_cc(), cc);
    EXPECT_EQ(fft2.get_cc(), cc);
    EXPECT_EQ(solver.get_cc(), cc);
#endif
  }
  MATX_EXIT_HANDLER();
}

#ifdef MATX_EN_MATHDX
TEST(DeviceAttributeCacheTest, CuSolverDxTraitsFollowCurrentDevice)
{
  MATX_ENTER_HANDLER();
  const int device_count = CudaDeviceCount();
  if (device_count == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  // The trait getters must load the traits for the current device themselves,
  // without an earlier IsSupported() call. Release builds used to compile the
  // load out with an assert, so the getters returned zeros, or the traits of
  // another device's architecture after a device switch.
  const auto configure = [](detail::cuSolverDxHelper<float> &helper) {
    helper.set_m(32);
    helper.set_n(32);
    helper.set_function(CUSOLVERDX_FUNCTION_POTRF);
    helper.set_fill_mode(SolverFillMode::LOWER);
  };

  const auto restore = detail::CudaDeviceGuard::RestoreCurrent();
  // Build the helper with device 0 current, as an operator's constructor would.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  detail::cuSolverDxHelper<float> solver;
  configure(solver);
  for (int device = device_count - 1; device >= 0; --device) {
    SCOPED_TRACE(device);
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    const auto block_dim = solver.GetBlockDim();
    const int shm_required = solver.GetShmRequired();

    // A helper built on this device that loads its traits through IsSupported().
    detail::cuSolverDxHelper<float> reference;
    configure(reference);
    ASSERT_TRUE(reference.IsSupported());
    EXPECT_GT(block_dim[0], 0);
    EXPECT_EQ(block_dim[0], reference.GetBlockDim()[0]);
    EXPECT_EQ(shm_required, reference.GetShmRequired());
  }
  MATX_EXIT_HANDLER();
}
#endif

TEST(DeviceAttributeCacheTest, ConcurrentFirstLookupsAgree)
{
  MATX_ENTER_HANDLER();
  const int device_count = CudaDeviceCount();
  if (device_count == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  // A private cache guarantees the threads race on the first fill, which the
  // process-wide cache may already have done.
  detail::DeviceAttributeCache cache;
  constexpr int num_threads = 16;
  std::latch start(num_threads);
  std::vector<cudaError_t> errors(num_threads, cudaErrorUnknown);
  std::vector<const detail::DeviceAttributes *> results(num_threads, nullptr);
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      start.arrive_and_wait();
      errors[t] = cache.Get(t % device_count, results[t]);
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  for (int t = 0; t < num_threads; ++t) {
    SCOPED_TRACE(t);
    const int device = t % device_count;
    ASSERT_EQ(errors[t], cudaSuccess);
    ASSERT_NE(results[t], nullptr);
    EXPECT_EQ(results[t], results[device]);
    EXPECT_EQ(results[t]->cc_major, detail::GetDeviceAttributes(device).cc_major);
    EXPECT_EQ(results[t]->max_smem_per_sm, detail::GetDeviceAttributes(device).max_smem_per_sm);
  }
  MATX_EXIT_HANDLER();
}

TEST(DeviceAttributeCacheTest, InvalidDeviceOrdinalIsRejected)
{
  MATX_ENTER_HANDLER();
  const int device_count = CudaDeviceCount();
  if (device_count == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  for (int device : {-1, device_count}) {
    SCOPED_TRACE(device);
    const detail::DeviceAttributes *attrs = nullptr;
    EXPECT_EQ(detail::TryGetDeviceAttributes(device, attrs), cudaErrorInvalidDevice);
    EXPECT_EQ(attrs, nullptr);
    EXPECT_THROW(detail::GetDeviceAttributes(device), detail::matxException);
  }

  // A rejected lookup must not disturb valid lookups or leave a CUDA error behind.
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(detail::GetDeviceAttributes(0).cc_major, RawAttr(cudaDevAttrComputeCapabilityMajor, 0));
  MATX_EXIT_HANDLER();
}

TEST(DeviceAttributeCacheTest, AttributeQueryFailures)
{
  MATX_ENTER_HANDLER();
  if (CudaDeviceCount() == 0) {
    GTEST_SKIP() << "CUDA device required";
  }

  // A failed required attribute fails the whole query.
  for (cudaDeviceAttr attr : {cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
                              cudaDevAttrMaxSharedMemoryPerBlock,
                              cudaDevAttrMaxSharedMemoryPerBlockOptin,
                              cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                              cudaDevAttrMaxRegistersPerMultiprocessor}) {
    SCOPED_TRACE(static_cast<int>(attr));
    failing_attr = attr;
    detail::DeviceAttributes attrs{};
    EXPECT_EQ(detail::QueryDeviceAttributes(0, attrs, &QueryFailingOne), cudaErrorInvalidValue);
  }

  // A failed optional attribute is stored as 0, the conservative value, and
  // the rest of the query succeeds.
  using IntField = int detail::DeviceAttributes::*;
  using BoolField = bool detail::DeviceAttributes::*;
  const auto &cached = detail::GetDeviceAttributes(0);
  const struct {
    cudaDeviceAttr attr;
    IntField field;
  } int_cases[] = {
    {cudaDevAttrL2CacheSize, &detail::DeviceAttributes::l2_cache_bytes},
    {cudaDevAttrGlobalMemoryBusWidth, &detail::DeviceAttributes::memory_bus_width_bits},
    {cudaDevAttrSingleToDoublePrecisionPerfRatio, &detail::DeviceAttributes::fp32_to_fp64_perf_ratio},
  };
  for (const auto &c : int_cases) {
    SCOPED_TRACE(static_cast<int>(c.attr));
    failing_attr = c.attr;
    detail::DeviceAttributes attrs{};
    ASSERT_EQ(detail::QueryDeviceAttributes(0, attrs, &QueryFailingOne), cudaSuccess);
    EXPECT_EQ(attrs.*c.field, 0);
    EXPECT_EQ(attrs.cc_major, cached.cc_major);
    EXPECT_EQ(attrs.max_smem_per_sm, cached.max_smem_per_sm);
  }

  const struct {
    cudaDeviceAttr attr;
    BoolField field;
  } bool_cases[] = {
    {cudaDevAttrConcurrentManagedAccess, &detail::DeviceAttributes::concurrent_managed_access},
    {cudaDevAttrMemoryPoolsSupported, &detail::DeviceAttributes::memory_pools_supported},
  };
  for (const auto &c : bool_cases) {
    SCOPED_TRACE(static_cast<int>(c.attr));
    failing_attr = c.attr;
    detail::DeviceAttributes attrs{};
    ASSERT_EQ(detail::QueryDeviceAttributes(0, attrs, &QueryFailingOne), cudaSuccess);
    EXPECT_FALSE(attrs.*c.field);
    EXPECT_EQ(attrs.cc_major, cached.cc_major);
  }
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  MATX_EXIT_HANDLER();
}

TEST(DeviceAttributeCacheTest, TensorVectorWidthFollowsComputeCapability)
{
  MATX_ENTER_HANDLER();
  // Wrap aligned host buffers so the test needs no GPU. The length and
  // alignment allow every vector width up to 32 bytes.
  constexpr index_t size = 1024;
  alignas(64) static float float_data[size];
  alignas(64) static double double_data[size];
  auto f = make_tensor<float>(float_data, {size});
  auto d = make_tensor<double>(double_data, {size});

  // CC 10.0 and later allow 32-byte vector loads. Earlier devices allow 16.
  const struct {
    int major;
    int minor;
    int vec_bytes;
  } cases[] = {{8, 9, 16}, {10, 0, 32}};
  for (const auto &c : cases) {
    SCOPED_TRACE(::testing::Message() << "CC " << c.major << "." << c.minor);
    const detail::EPTQueryInput query{false, AttrsWithComputeCapability(c.major, c.minor)};
    EXPECT_EQ(query.max_vec_width_bytes, c.vec_bytes);

    const int float_width = c.vec_bytes / static_cast<int>(sizeof(float));
    const auto float_ept = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(f, query);
    EXPECT_EQ(float_ept[0], detail::ElementsPerThread::ONE);
    EXPECT_EQ(float_ept[1], static_cast<detail::ElementsPerThread>(float_width));
    EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(f, query), float_width);

    const int double_width = c.vec_bytes / static_cast<int>(sizeof(double));
    const auto double_ept = detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(d, query);
    EXPECT_EQ(double_ept[0], detail::ElementsPerThread::ONE);
    EXPECT_EQ(double_ept[1], static_cast<detail::ElementsPerThread>(double_width));
    EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(d, query), double_width);
  }
  MATX_EXIT_HANDLER();
}
