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

#include "matx/core/matx_tensor.h"

#include <gtest/gtest.h>

using namespace matx;

TEST(MinimalTensor, BasicMathOnHost)
{
  auto a = make_tensor<float>({3}, MATX_HOST_MALLOC_MEMORY);
  auto b = make_tensor<float>({3}, MATX_HOST_MALLOC_MEMORY);
  auto c = make_tensor<float>({3}, MATX_HOST_MALLOC_MEMORY);

  for (index_t i = 0; i < 3; ++i) {
    a(i) = static_cast<float>(i + 1);
    b(i) = static_cast<float>(i + 4);
  }

  EXPECT_EQ(a.Size(0), 3);
  EXPECT_FLOAT_EQ(a(2), 3.0f);

  (c = a + b).run(HostExecutor{});
  EXPECT_FLOAT_EQ(c(0), 5.0f);
  EXPECT_FLOAT_EQ(c(1), 7.0f);
  EXPECT_FLOAT_EQ(c(2), 9.0f);

  (c = a * b).run(HostExecutor{});
  EXPECT_FLOAT_EQ(c(0), 4.0f);
  EXPECT_FLOAT_EQ(c(1), 10.0f);
  EXPECT_FLOAT_EQ(c(2), 18.0f);

  (c = -a).run(HostExecutor{});
  EXPECT_FLOAT_EQ(c(0), -1.0f);
  EXPECT_FLOAT_EQ(c(1), -2.0f);
  EXPECT_FLOAT_EQ(c(2), -3.0f);

  (c = a + 5.0f).run(HostExecutor{});
  EXPECT_FLOAT_EQ(c(0), 6.0f);
  EXPECT_FLOAT_EQ(c(1), 7.0f);
  EXPECT_FLOAT_EQ(c(2), 8.0f);
}

TEST(MinimalTensor, ScalarOnHost)
{
  auto a = make_tensor<int>({}, MATX_HOST_MALLOC_MEMORY);
  auto b = make_tensor<int>({}, MATX_HOST_MALLOC_MEMORY);
  a() = 3;

  (b = -a).run(HostExecutor{});
  EXPECT_EQ(b(), -3);
}

TEST(MinimalTensor, ComposedMathOnCuda)
{
  auto a = make_tensor<int>({2, 2});
  auto b = make_tensor<int>({2, 2});
  auto c = make_tensor<int>({2, 2});

  for (index_t i = 0; i < 2; ++i) {
    for (index_t j = 0; j < 2; ++j) {
      a(i, j) = static_cast<int>(i + j);
      b(i, j) = 2;
    }
  }

  (c = -(a + b)).run(cudaExecutor{});
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(c(0, 0), -2);
  EXPECT_EQ(c(0, 1), -3);
  EXPECT_EQ(c(1, 0), -3);
  EXPECT_EQ(c(1, 1), -4);
}
