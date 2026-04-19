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
#include "matx/kernels/tensor_accessor.h"
#include "gtest/gtest.h"

using namespace matx;

// ---- Fast path ----------------------------------------------------------

TEST(TensorAccessor, FastPathRank1)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({16});
    (t = range<0>({16}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), /*IsUnitStride=*/true> acc(t);
    for (index_t i = 0; i < 16; ++i) {
        ASSERT_EQ(acc(i), static_cast<int>(i));
    }
}

TEST(TensorAccessor, FastPathRank2)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({4, 8});
    // t(r, c) = 8*r + c (row-major flat index).
    (t = 8 * range<0>({4, 8}, 0, 1) + range<1>({4, 8}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    for (index_t r = 0; r < 4; ++r) {
        for (index_t c = 0; c < 8; ++c) {
            ASSERT_EQ(acc(r, c), t(r, c));
        }
    }
}

TEST(TensorAccessor, FastPathRank3)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({3, 4, 5});
    (t = 20 * range<0>({3, 4, 5}, 0, 1)
           + 5 * range<1>({3, 4, 5}, 0, 1)
           +     range<2>({3, 4, 5}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    for (index_t i = 0; i < 3; ++i) {
        for (index_t j = 0; j < 4; ++j) {
            for (index_t k = 0; k < 5; ++k) {
                ASSERT_EQ(acc(i, j, k), t(i, j, k));
            }
        }
    }
}

// Write-through: assigning via fast-path operator() modifies the underlying tensor.
TEST(TensorAccessor, FastPathWriteThrough)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({4, 4});
    (t = 0).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    for (index_t r = 0; r < 4; ++r) {
        for (index_t c = 0; c < 4; ++c) {
            acc(r, c) = static_cast<int>(100 + r * 10 + c);
        }
    }
    cudaDeviceSynchronize();
    for (index_t r = 0; r < 4; ++r) {
        for (index_t c = 0; c < 4; ++c) {
            ASSERT_EQ(t(r, c), static_cast<int>(100 + r * 10 + c));
        }
    }
}

// ---- Slow path ----------------------------------------------------------

// IsUnitStride=false falls through to op_(indices...) regardless of the
// underlying tensor's stride. Verifies read correctness on a plain view.
TEST(TensorAccessor, SlowPathOnTensorView)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({4, 8});
    (t = 8 * range<0>({4, 8}, 0, 1) + range<1>({4, 8}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), /*IsUnitStride=*/false> acc(t);
    for (index_t r = 0; r < 4; ++r) {
        for (index_t c = 0; c < 8; ++c) {
            ASSERT_EQ(acc(r, c), t(r, c));
        }
    }
}

// Non-unit-stride slice: last-dim stride = 2. Fast path would give wrong
// offsets, so these must be accessed through the slow path.
TEST(TensorAccessor, SlowPathNonUnitStride)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({4, 16});
    (t = 16 * range<0>({4, 16}, 0, 1) + range<1>({4, 16}, 0, 1)).run(exec);
    exec.sync();
    auto strided = slice(t, {0, 0}, {matxEnd, matxEnd}, {1, 2});
    EXPECT_EQ(strided.Stride(1), 2);  // sanity: actually non-unit
    detail::TensorAccessor<decltype(strided), false> acc(strided);
    for (index_t r = 0; r < strided.Size(0); ++r) {
        for (index_t c = 0; c < strided.Size(1); ++c) {
            ASSERT_EQ(acc(r, c), strided(r, c));
        }
    }
}

// ConstVal is a computed op with no Data() / Stride(), so only the slow path
// can work on it. IsUnitStride=false instantiates the slow path.
TEST(TensorAccessor, SlowPathOnConstVal)
{
    auto z = zeros<int>({4, 8});
    detail::TensorAccessor<decltype(z), false> acc(z);
    for (index_t r = 0; r < 4; ++r) {
        for (index_t c = 0; c < 8; ++c) {
            ASSERT_EQ(acc(r, c), 0);
        }
    }
}

// ---- bind_first_n -------------------------------------------------------

TEST(TensorAccessor, BindFirstN_Zero)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({4, 8});
    (t = 8 * range<0>({4, 8}, 0, 1) + range<1>({4, 8}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    cuda::std::array<index_t, 0> idx{};
    auto bound = detail::bind_first_n<0>(acc, idx);
    // N=0 returns the accessor unchanged.
    for (index_t r = 0; r < 4; ++r) {
        for (index_t c = 0; c < 8; ++c) {
            ASSERT_EQ(bound(r, c), t(r, c));
        }
    }
}

TEST(TensorAccessor, BindFirstN_BatchDim)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({3, 5, 7});
    (t = 35 * range<0>({3, 5, 7}, 0, 1)
           + 7 * range<1>({3, 5, 7}, 0, 1)
           +     range<2>({3, 5, 7}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    for (index_t b = 0; b < 3; ++b) {
        cuda::std::array<index_t, 3> idx{b, 0, 0}; // bind_first_n uses arr[0..N-1]
        auto bound = detail::bind_first_n<1>(acc, idx);
        for (index_t j = 0; j < 5; ++j) {
            for (index_t k = 0; k < 7; ++k) {
                ASSERT_EQ(bound(j, k), t(b, j, k));
            }
        }
    }
}

// Bind-all-dims-of-rank-1 is the case that previously read an uninitialized
// outer_strides_[0] in BoundAccessor's fast-path constructor. With the fix,
// the last-dim stride is taken as 1 rather than from the unpopulated slot.
TEST(TensorAccessor, BindAllDims_Rank1_FastPath)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({16});
    (t = range<0>({16}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    for (index_t i = 0; i < 16; ++i) {
        cuda::std::array<index_t, 1> idx{i};
        auto bound = detail::bind_first_n<1>(acc, idx);
        // After binding the single dim, the remaining rank is 0; call nullary operator().
        ASSERT_EQ(bound(), static_cast<int>(i)) << "at i=" << i;
    }
}

TEST(TensorAccessor, BindAllDims_Rank2_FastPath)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({3, 5});
    (t = 5 * range<0>({3, 5}, 0, 1) + range<1>({3, 5}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    for (index_t r = 0; r < 3; ++r) {
        for (index_t c = 0; c < 5; ++c) {
            cuda::std::array<index_t, 2> idx{r, c};
            auto bound = detail::bind_first_n<2>(acc, idx);
            ASSERT_EQ(bound(), t(r, c));
        }
    }
}

// Same as BindFirstN_BatchDim but with IsUnitStride=false, forcing slow path.
// Exercises BoundAccessor's slow_forward path.
TEST(TensorAccessor, BindFirstN_SlowPath)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({3, 5, 7});
    (t = 35 * range<0>({3, 5, 7}, 0, 1)
           + 7 * range<1>({3, 5, 7}, 0, 1)
           +     range<2>({3, 5, 7}, 0, 1)).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), false> acc(t);
    for (index_t b = 0; b < 3; ++b) {
        cuda::std::array<index_t, 3> idx{b, 0, 0};
        auto bound = detail::bind_first_n<1>(acc, idx);
        for (index_t j = 0; j < 5; ++j) {
            for (index_t k = 0; k < 7; ++k) {
                ASSERT_EQ(bound(j, k), t(b, j, k));
            }
        }
    }
}

// Write-through via a bound accessor.
TEST(TensorAccessor, BoundAccessor_WriteThrough)
{
    auto exec = cudaExecutor{};
    auto t = make_tensor<int>({3, 5});
    (t = 0).run(exec);
    exec.sync();
    detail::TensorAccessor<decltype(t), true> acc(t);
    for (index_t b = 0; b < 3; ++b) {
        cuda::std::array<index_t, 2> idx{b, 0};
        auto bound = detail::bind_first_n<1>(acc, idx);
        for (index_t j = 0; j < 5; ++j) {
            bound(j) = static_cast<int>(1000 + b * 10 + j);
        }
    }
    cudaDeviceSynchronize();
    for (index_t b = 0; b < 3; ++b) {
        for (index_t j = 0; j < 5; ++j) {
            ASSERT_EQ(t(b, j), static_cast<int>(1000 + b * 10 + j));
        }
    }
}

// Slow-path bind on a ConstVal: remaining dim access returns the ConstVal's value
// regardless of bound leading index.
TEST(TensorAccessor, BindFirstN_ConstVal)
{
    auto z = zeros<int>({4, 8});
    detail::TensorAccessor<decltype(z), false> acc(z);
    for (index_t b = 0; b < 4; ++b) {
        cuda::std::array<index_t, 2> idx{b, 0};
        auto bound = detail::bind_first_n<1>(acc, idx);
        for (index_t j = 0; j < 8; ++j) {
            ASSERT_EQ(bound(j), 0);
        }
    }
}
