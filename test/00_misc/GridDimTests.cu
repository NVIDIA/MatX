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

#include "matx/core/get_grid_dims.h"

#include "gtest/gtest.h"

using namespace matx;

namespace {

void ExpectDim3(const dim3 &actual, unsigned int x, unsigned int y, unsigned int z)
{
  EXPECT_EQ(actual.x, x);
  EXPECT_EQ(actual.y, y);
  EXPECT_EQ(actual.z, z);
}

} // namespace

TEST(GridDimTests, ComputesStandardGridDimsAcrossRanks)
{
  dim3 blocks{};
  dim3 threads{};

  bool stride = detail::get_grid_dims<0>(blocks, threads, cuda::std::array<index_t, 0>{}, 1, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 1);
  ExpectDim3(threads, 1, 1, 1);

  stride = detail::get_grid_dims<1>(blocks, threads, cuda::std::array<index_t, 1>{17}, 2, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 2, 1, 1);
  ExpectDim3(threads, 8, 1, 1);

  stride = detail::get_grid_dims<2>(blocks, threads, cuda::std::array<index_t, 2>{7, 4}, 1, 16);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 2, 1);
  ExpectDim3(threads, 4, 4, 1);

  stride = detail::get_grid_dims<3>(blocks, threads, cuda::std::array<index_t, 3>{5, 1, 1}, 1, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 1);
  ExpectDim3(threads, 1, 1, 8);

  stride = detail::get_grid_dims<4>(blocks, threads, cuda::std::array<index_t, 4>{5, 3, 2, 4}, 1, 64);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 3);
  ExpectDim3(threads, 4, 8, 2);

  stride = detail::get_grid_dims<5>(blocks, threads, cuda::std::array<index_t, 5>{2, 3, 4, 5, 6}, 1, 64);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 12, 1, 1);
  ExpectDim3(threads, 64, 1, 1);
}

TEST(GridDimTests, ClampsStandardGridDimsAtCudaLimits)
{
  dim3 blocks{};
  dim3 threads{};

  bool stride = detail::get_grid_dims<2>(blocks, threads, cuda::std::array<index_t, 2>{70000, 1}, 1, 1);
  EXPECT_TRUE(stride);
  ExpectDim3(blocks, 1, 65535, 1);
  ExpectDim3(threads, 1, 1, 1);

  stride = detail::get_grid_dims<3>(blocks, threads, cuda::std::array<index_t, 3>{70000, 70000, 1}, 1, 1);
  EXPECT_TRUE(stride);
  ExpectDim3(blocks, 1, 65535, 65535);
  ExpectDim3(threads, 1, 1, 1);

  stride = detail::get_grid_dims<3>(blocks, threads, cuda::std::array<index_t, 3>{1000, 1, 1}, 1, 1024);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 16);
  ExpectDim3(threads, 1, 1, 64);

  stride = detail::get_grid_dims<4>(blocks, threads, cuda::std::array<index_t, 4>{70000, 70000, 2, 1}, 1, 1);
  EXPECT_TRUE(stride);
  ExpectDim3(blocks, 1, 65535, 65535);
  ExpectDim3(threads, 1, 1, 1);

  stride = detail::get_grid_dims<4>(blocks, threads, cuda::std::array<index_t, 4>{1000, 1, 1, 1}, 1, 1024);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 16);
  ExpectDim3(threads, 1, 1, 64);
}

TEST(GridDimTests, ComputesBlockGridDimsAcrossRanks)
{
  dim3 blocks{};
  dim3 threads{};

  bool stride = detail::get_grid_dims_block<0>(blocks, threads, cuda::std::array<index_t, 0>{}, 1, 1, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 1);
  ExpectDim3(threads, 1, 1, 1);

  stride = detail::get_grid_dims_block<1>(blocks, threads, cuda::std::array<index_t, 1>{17}, 2, 1, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 1);
  ExpectDim3(threads, 8, 1, 1);

  stride = detail::get_grid_dims_block<1>(blocks, threads, cuda::std::array<index_t, 1>{17}, 2, 1, 128, true);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 1);
  ExpectDim3(threads, 128, 1, 1);

  stride = detail::get_grid_dims_block<2>(blocks, threads, cuda::std::array<index_t, 2>{8, 17}, 2, 4, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 2, 1, 1);
  ExpectDim3(threads, 8, 4, 1);

  stride = detail::get_grid_dims_block<2>(blocks, threads, cuda::std::array<index_t, 2>{8, 17}, 2, 1, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 8, 1, 1);
  ExpectDim3(threads, 8, 1, 1);

  stride = detail::get_grid_dims_block<3>(blocks, threads, cuda::std::array<index_t, 3>{3, 5, 9}, 1, 1, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 5, 3, 1);
  ExpectDim3(threads, 8, 1, 1);

  stride = detail::get_grid_dims_block<4>(blocks, threads, cuda::std::array<index_t, 4>{3, 4, 5, 9}, 1, 1, 8);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 5, 4, 3);
  ExpectDim3(threads, 8, 1, 1);
}

TEST(GridDimTests, ClampsBlockGridDimsAtCudaLimits)
{
  dim3 blocks{};
  dim3 threads{};

  bool stride = detail::get_grid_dims_block<3>(
      blocks, threads, cuda::std::array<index_t, 3>{70000, 140000, 10}, 1, 2, 8);
  EXPECT_TRUE(stride);
  ExpectDim3(blocks, 65535, 65535, 1);
  ExpectDim3(threads, 8, 2, 1);

  stride = detail::get_grid_dims_block<4>(
      blocks, threads, cuda::std::array<index_t, 4>{70000, 70000, 140000, 10}, 1, 2, 8);
  EXPECT_TRUE(stride);
  ExpectDim3(blocks, 65535, 65535, 65535);
  ExpectDim3(threads, 8, 2, 1);
}

TEST(GridDimTests, ComputesBlock2DGridDims)
{
  dim3 blocks{};
  dim3 threads{};

  bool stride = detail::get_grid_dims_block_2d<2>(blocks, threads, cuda::std::array<index_t, 2>{8, 16}, 32);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 1, 1, 1);
  ExpectDim3(threads, 32, 1, 1);

  stride = detail::get_grid_dims_block_2d<3>(blocks, threads, cuda::std::array<index_t, 3>{7, 8, 16}, 64);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 7, 1, 1);
  ExpectDim3(threads, 64, 1, 1);

  stride = detail::get_grid_dims_block_2d<4>(blocks, threads, cuda::std::array<index_t, 4>{3, 5, 8, 16}, 128);
  EXPECT_FALSE(stride);
  ExpectDim3(blocks, 5, 3, 1);
  ExpectDim3(threads, 128, 1, 1);
}

TEST(GridDimTests, RejectsInvalidBlockGridDims)
{
#ifndef NDEBUG
  auto invalid_groups_per_block = []() {
    dim3 blocks{};
    dim3 threads{};
    detail::get_grid_dims_block<2>(blocks, threads, cuda::std::array<index_t, 2>{7, 17}, 1, 4, 8);
  };
  EXPECT_THROW(invalid_groups_per_block(), matx::detail::matxException);
#endif

  auto oversized_batch_dim = []() {
    dim3 blocks{};
    dim3 threads{};
    detail::get_grid_dims_block_2d<3>(blocks, threads, cuda::std::array<index_t, 3>{70000, 8, 16}, 32);
  };
  EXPECT_THROW(oversized_batch_dim(), matx::detail::matxException);
}
