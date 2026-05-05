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

#include <cuda/std/tuple>

using namespace matx;

namespace {

bool HasCudaDevice()
{
  int device_count = 0;
  const auto err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    return false;
  }

  return true;
}

template <typename KernelProvider>
void ExpectAllSupportedEPTs(KernelProvider &&provider, bool expect_supported)
{
  using detail::ElementsPerThread;

  if (expect_supported) {
    EXPECT_TRUE(provider(ElementsPerThread::THIRTY_TWO) != nullptr);
    EXPECT_TRUE(provider(ElementsPerThread::SIXTEEN) != nullptr);
    EXPECT_TRUE(provider(ElementsPerThread::EIGHT) != nullptr);
    EXPECT_TRUE(provider(ElementsPerThread::FOUR) != nullptr);
    EXPECT_TRUE(provider(ElementsPerThread::TWO) != nullptr);
    EXPECT_TRUE(provider(ElementsPerThread::ONE) != nullptr);
  }
  else {
    EXPECT_EQ(provider(ElementsPerThread::THIRTY_TWO), nullptr);
    EXPECT_EQ(provider(ElementsPerThread::SIXTEEN), nullptr);
    EXPECT_EQ(provider(ElementsPerThread::EIGHT), nullptr);
    EXPECT_EQ(provider(ElementsPerThread::FOUR), nullptr);
    EXPECT_EQ(provider(ElementsPerThread::TWO), nullptr);
    EXPECT_EQ(provider(ElementsPerThread::ONE), nullptr);
  }

  EXPECT_EQ(provider(ElementsPerThread::INVALID), nullptr);
}

template <typename Op, size_t RANK>
void ExpectProviderForSizes(const cuda::std::array<index_t, RANK> &sizes, bool is_jit = false, bool global_kernel = false)
{
  static_assert(Op::Rank() == static_cast<int>(RANK));
  auto provider = detail::create_kernel_provider<Op>(sizes, is_jit, global_kernel);
  ExpectAllSupportedEPTs(provider, RANK <= 4);
}

} // namespace

TEST(CudaExecutorCommonTests, BaseConstructorsKeepStreamAndRejectUnprofiledTiming)
{
  detail::CudaExecutorBase default_exec;
  EXPECT_EQ(default_exec.getStream(), reinterpret_cast<cudaStream_t>(0));
  default_exec.start_timer();
  default_exec.stop_timer();
  EXPECT_THROW({ default_exec.get_time_ms(); }, detail::matxException);

  detail::CudaExecutorBase int_stream_exec{7};
  EXPECT_EQ(int_stream_exec.getStream(), reinterpret_cast<cudaStream_t>(7));

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(11);
  detail::CudaExecutorBase stream_exec{stream};
  EXPECT_EQ(stream_exec.getStream(), stream);
}

TEST(CudaExecutorCommonTests, ProfiledBaseRecordsElapsedTimeWhenDeviceIsAvailable)
{
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device required for profiling event coverage";
  }

  cudaStream_t stream{};
  MATX_CUDA_CHECK(cudaStreamCreate(&stream));
  {
    detail::CudaExecutorBase exec{stream, true};
    exec.start_timer();
    exec.stop_timer();
    EXPECT_TRUE(exec.get_time_ms() >= 0.0f);
  }
  MATX_CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(CudaExecutorCommonTests, KernelProviderReturnsRankSpecificPointers)
{
  int *ptr = nullptr;
  auto t0 = make_tensor<int>(ptr, {}, false);
  auto t1 = make_tensor<int>(ptr, {32}, false);
  auto t2 = make_tensor<int>(ptr, {8, 32}, false);
  auto t3 = make_tensor<int>(ptr, {4, 8, 32}, false);
  auto t4 = make_tensor<int>(ptr, {2, 4, 8, 32}, false);
  auto t5 = make_tensor<int>(ptr, {2, 2, 2, 2, 2}, false);

  ExpectProviderForSizes<decltype(t0)>(cuda::std::array<index_t, 0>{});
  ExpectProviderForSizes<decltype(t1)>(cuda::std::array<index_t, 1>{32});
  ExpectProviderForSizes<decltype(t2)>(cuda::std::array<index_t, 2>{8, 32});
  ExpectProviderForSizes<decltype(t3)>(cuda::std::array<index_t, 3>{4, 8, 32});
  ExpectProviderForSizes<decltype(t4)>(cuda::std::array<index_t, 4>{2, 4, 8, 32});
  ExpectProviderForSizes<decltype(t5)>(cuda::std::array<index_t, 5>{2, 2, 2, 2, 2});
}

TEST(CudaExecutorCommonTests, KernelProviderCoversStrideAndJitGridPaths)
{
  int *ptr = nullptr;
  auto t2 = make_tensor<int>(ptr, {70000, 2}, false);
  auto t3 = make_tensor<int>(ptr, {70000, 70000, 2}, false);
  auto t4 = make_tensor<int>(ptr, {70000, 70000, 2, 2}, false);

  ExpectProviderForSizes<decltype(t2)>(cuda::std::array<index_t, 2>{70000, 2});
  ExpectProviderForSizes<decltype(t3)>(cuda::std::array<index_t, 3>{70000, 70000, 2});
  ExpectProviderForSizes<decltype(t4)>(cuda::std::array<index_t, 4>{70000, 70000, 2, 2});

  auto non_global_jit = make_tensor<int>(ptr, {8, 32}, false);
  ExpectProviderForSizes<decltype(non_global_jit)>(cuda::std::array<index_t, 2>{8, 32}, true, false);
  ExpectProviderForSizes<decltype(non_global_jit)>(cuda::std::array<index_t, 2>{8, 32}, true, true);
}

TEST(CudaExecutorCommonTests, FindBestLaunchParamsUsesCompiledKernelAttributes)
{
  if (!HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device required for launch-parameter coverage";
  }

  int *ptr = nullptr;
  auto op = make_tensor<int>(ptr, {128}, false);
  auto provider = detail::create_kernel_provider<decltype(op)>(cuda::std::array<index_t, 1>{128});
  auto result = detail::find_best_launch_params(op, provider, 256, false);

  const auto ept = cuda::std::get<0>(result);
  const auto shm_size = cuda::std::get<1>(result);
  const auto block_size = cuda::std::get<2>(result);
  const auto groups_per_block = cuda::std::get<3>(result);

  EXPECT_TRUE(static_cast<int>(ept) >= static_cast<int>(detail::ElementsPerThread::ONE));
  EXPECT_TRUE(shm_size >= 0);
  EXPECT_TRUE(block_size > 0);
  EXPECT_TRUE(groups_per_block >= 1);
}
