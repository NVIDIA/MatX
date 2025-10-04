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
#include <thread>
#include <chrono>

using namespace matx;

// Test basic profiling functionality
TEST(ProfilingTests, BasicTimerTest)
{
  MATX_ENTER_HANDLER();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Create executor with profiling enabled
  cudaExecutor exec{stream, true};
  
  // Create a simple tensor operation
  auto t1 = make_tensor<float>({1024, 1024});
  auto t2 = make_tensor<float>({1024, 1024});
  auto t3 = make_tensor<float>({1024, 1024});
  
  // Initialize tensors
  (t1 = 1.0f).run(exec);
  (t2 = 2.0f).run(exec);
  
  // Start profiling
  exec.start_timer();
  
  // Perform operation
  (t3 = t1 + t2).run(exec);
  
  // Stop profiling
  exec.stop_timer();
  
  // Get elapsed time
  float elapsed_ms = exec.get_time_ms();
  
  // Time should be greater than 0
  EXPECT_GT(elapsed_ms, 0.0f);
  
  // Clean up
  cudaStreamDestroy(stream);
  
  MATX_EXIT_HANDLER();
}

// Test profiling with disabled state (should throw assertion)
TEST(ProfilingTests, ProfilingDisabledTest)
{
  MATX_ENTER_HANDLER();
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Create executor with profiling disabled (default)
  cudaExecutor exec{stream, false};
  
  // Create a simple tensor operation
  auto t1 = make_tensor<float>({256});
  (t1 = 1.0f).run(exec);
  
  // start_timer and stop_timer should be safe to call even when profiling is disabled
  // They should just be no-ops
  EXPECT_NO_THROW(exec.start_timer());
  (t1 = 2.0f).run(exec);
  EXPECT_NO_THROW(exec.stop_timer());
  
  // Getting time should fail with assertion when profiling is disabled
  EXPECT_THROW(exec.get_time_ms(), matx::detail::matxException);
  
  // Verify that with profiling enabled, everything works correctly
  cudaExecutor exec_prof{stream, true};
  
  exec_prof.start_timer();
  (t1 = 3.0f).run(exec_prof);
  exec_prof.stop_timer();
  
  float time_prof = exec_prof.get_time_ms();
  EXPECT_GT(time_prof, 0.0f);
  
  // Clean up
  cudaStreamDestroy(stream);
  
  MATX_EXIT_HANDLER();
}

// Test multiple timer cycles
TEST(ProfilingTests, MultipleTimerCyclesTest)
{
  MATX_ENTER_HANDLER();
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Create executor with profiling enabled
  cudaExecutor exec{stream, true};
  
  // Create tensors for operations
  auto t1 = make_tensor<float>({512, 512});
  auto t2 = make_tensor<float>({512, 512});
  auto t3 = make_tensor<float>({512, 512});
  
  // Initialize tensors
  (t1 = 1.0f).run(exec);
  (t2 = 2.0f).run(exec);
  
  float total_time = 0.0f;
  const int num_cycles = 5;
  
  for (int i = 0; i < num_cycles; i++) {
    // Start profiling
    exec.start_timer();
    
    // Perform operation
    (t3 = t1 + t2 * static_cast<float>(i)).run(exec);
    
    // Stop profiling
    exec.stop_timer();
    
    // Get elapsed time
    float elapsed_ms = exec.get_time_ms();
    EXPECT_GT(elapsed_ms, 0.0f);
    
    total_time += elapsed_ms;
  }
  
  // Total time should be greater than any individual cycle
  EXPECT_GT(total_time, 0.0f);
  
  // Clean up
  cudaStreamDestroy(stream);
  
  MATX_EXIT_HANDLER();
}

// Test profiling with complex operations
TEST(ProfilingTests, ComplexOperationProfilingTest)
{
  MATX_ENTER_HANDLER();
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Create executor with profiling enabled
  cudaExecutor exec{stream, true};
  
  // Create tensors for complex operations
  const int N = 1024;
  auto t1 = make_tensor<float>({N, N});
  auto t2 = make_tensor<float>({N, N});
  auto t3 = make_tensor<float>({N, N});
  auto t4 = make_tensor<float>({N, N});
  
  // Initialize tensors
  (t1 = ones<float>({N, N})).run(exec);
  (t2 = ones<float>({N, N}) * 2.0f).run(exec);
  
  // Start profiling
  exec.start_timer();
  
  // Perform multiple operations
  (t3 = t1 + t2).run(exec);
  (t4 = t3 * 2.0f + sqrt(t1)).run(exec);
  (t3 = sin(t4) + cos(t2)).run(exec);
  
  // Stop profiling
  exec.stop_timer();
  
  // Get elapsed time
  float elapsed_ms = exec.get_time_ms();
  
  // Time should be greater than 0
  EXPECT_GT(elapsed_ms, 0.0f);
  
  // Clean up
  cudaStreamDestroy(stream);
  
  MATX_EXIT_HANDLER();
}

// Test profiling with async operations and synchronization
TEST(ProfilingTests, AsyncProfilingTest)
{
  MATX_ENTER_HANDLER();
  
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  
  // Create executors with profiling enabled
  cudaExecutor exec1{stream1, true};
  cudaExecutor exec2{stream2, true};
  
  // Create tensors
  auto t1 = make_tensor<float>({2048});
  auto t2 = make_tensor<float>({2048});
  auto t3 = make_tensor<float>({2048});
  auto t4 = make_tensor<float>({2048});
  
  // Initialize tensors
  (t1 = 1.0f).run(exec1);
  (t2 = 2.0f).run(exec1);
  (t3 = 3.0f).run(exec2);
  (t4 = 4.0f).run(exec2);
  
  // Start profiling on both streams
  exec1.start_timer();
  exec2.start_timer();
  
  // Perform operations on different streams
  (t1 = t1 + t2).run(exec1);
  (t3 = t3 * t4).run(exec2);
  
  // Stop profiling
  exec1.stop_timer();
  exec2.stop_timer();
  
  // Get elapsed times
  float elapsed_ms1 = exec1.get_time_ms();
  float elapsed_ms2 = exec2.get_time_ms();
  
  // Both times should be greater than 0
  EXPECT_GT(elapsed_ms1, 0.0f);
  EXPECT_GT(elapsed_ms2, 0.0f);
  
  // Clean up
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  
  MATX_EXIT_HANDLER();
}

// Test profiling overhead by comparing with/without profiling
TEST(ProfilingTests, ProfilingOverheadTest)
{
  MATX_ENTER_HANDLER();
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Create executors with and without profiling
  cudaExecutor exec_prof{stream, true};
  cudaExecutor exec_no_prof{stream, false};
  
  // Create tensors
  const int N = 4096;
  auto t1 = make_tensor<float>({N});
  auto t2 = make_tensor<float>({N});
  auto t3 = make_tensor<float>({N});
  
  // Initialize tensors
  (t1 = 1.0f).run(exec_prof);
  (t2 = 2.0f).run(exec_prof);
  
  // Warm up
  (t3 = t1 + t2).run(exec_prof);
  cudaStreamSynchronize(stream);
  
  // Time with profiling
  auto start_cpu = std::chrono::high_resolution_clock::now();
  
  exec_prof.start_timer();
  for (int i = 0; i < 100; i++) {
    (t3 = t1 + t2).run(exec_prof);
  }
  exec_prof.stop_timer();
  
  float gpu_time_ms = exec_prof.get_time_ms();
  
  auto end_cpu = std::chrono::high_resolution_clock::now();
  auto cpu_time_prof = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count()) / 1000.0f;
  
  // Time without profiling (CPU timing only)
  start_cpu = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < 100; i++) {
    (t3 = t1 + t2).run(exec_no_prof);
  }
  cudaStreamSynchronize(stream);
  
  end_cpu = std::chrono::high_resolution_clock::now();
  auto cpu_time_no_prof = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count()) / 1000.0f;
  
  // GPU time should be reasonable
  EXPECT_GT(gpu_time_ms, 0.0f);
  
  // The overhead should be minimal (allow for some variance)
  // Note: This is a soft check as timing can vary
  float overhead_ratio = cpu_time_prof / cpu_time_no_prof;
  EXPECT_LT(overhead_ratio, 2.0f); // Profiling shouldn't more than double the time
  
  // Clean up
  cudaStreamDestroy(stream);
  
  MATX_EXIT_HANDLER();
}

// Test profiling with default stream
TEST(ProfilingTests, DefaultStreamProfilingTest)
{
  MATX_ENTER_HANDLER();
  
  // Create executor with default stream and profiling enabled
  cudaExecutor exec{0, true}; // Using 0 for default stream
  
  // Create a simple tensor operation
  auto t1 = make_tensor<float>({256, 256});
  auto t2 = make_tensor<float>({256, 256});
  
  // Initialize tensors
  (t1 = 1.5f).run(exec);
  
  // Start profiling
  exec.start_timer();
  
  // Perform operation
  (t2 = t1 * 2.0f + 1.0f).run(exec);
  
  // Stop profiling
  exec.stop_timer();
  
  // Get elapsed time
  float elapsed_ms = exec.get_time_ms();
  
  // Time should be greater than 0
  EXPECT_GT(elapsed_ms, 0.0f);
  
  MATX_EXIT_HANDLER();
}
