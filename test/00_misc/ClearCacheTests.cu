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
#include <vector>
#include <unordered_map>

using namespace matx;

TEST(ClearCacheTests, TestCase) {
    MATX_ENTER_HANDLER();

    size_t initial_free_mem = 0;
    size_t total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&initial_free_mem, &total_mem);
    ASSERT_EQ(err, cudaSuccess);

    // The cuBLAS handle will allocate an associated workspace of 4 MiB on pre-Hopper and
    // 32 MiB on Hopper+.
    {
        auto c = matx::make_tensor<float, 2>({1024, 1024});
        auto a = matx::make_tensor<float, 2>({1024, 1024});
        auto b = matx::make_tensor<float, 2>({1024, 1024});
        (c = matx::matmul(a, b)).run();
        cudaDeviceSynchronize();    
    }

    // Manually allocate 4 MiB
    const size_t four_MiB = 4 * 1024 * 1024;
    void *ptr;
    matxAlloc(&ptr, four_MiB, MATX_DEVICE_MEMORY);

    size_t post_alloc_free_mem = 0;
    err = cudaMemGetInfo(&post_alloc_free_mem, &total_mem);
    ASSERT_EQ(err, cudaSuccess);

    matx::ClearCachesAndAllocations();

    size_t post_clear_free_mem = 0;
    err = cudaMemGetInfo(&post_clear_free_mem, &total_mem);
    ASSERT_EQ(err, cudaSuccess);

    const ssize_t allocated = static_cast<ssize_t>(initial_free_mem) - static_cast<ssize_t>(post_alloc_free_mem);
    const ssize_t freed = static_cast<ssize_t>(post_clear_free_mem) - static_cast<ssize_t>(post_alloc_free_mem);

    // The cuBLAS cache and allocator data structure should have allocated at least 8 MiB
    // in total and thus at least 8 MiB should be freed when clearing the caches/allocations.
    ASSERT_GE(allocated, 2 * four_MiB);
    ASSERT_GE(freed, 2 * four_MiB);

    MATX_EXIT_HANDLER();
}