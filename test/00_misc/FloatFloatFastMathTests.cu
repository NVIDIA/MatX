////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "matx.h"
#include "gtest/gtest.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <memory>

using namespace matx;

namespace {
constexpr std::size_t RESULT_COUNT = 20;
constexpr std::size_t DOUBLE_RESULT_COUNT = 4;

struct CudaFreeDeleter {
    template <typename T>
    void operator()(T *ptr) const noexcept
    {
        if (ptr != nullptr) {
            (void)cudaFree(ptr);
        }
    }
};
} // namespace

static_assert(static_cast<double>(fltflt{1.0f, 0x1p-24f}) == 1.0 + 0x1p-24);

__global__ void FltFltGradualUnderflowKernel(unsigned int *result_bits,
                                             unsigned long long *double_result_bits,
                                             double min_subnormal_as_double,
                                             unsigned int ftz_probe_bits,
                                             float normal_sqrt_input)
{
    // FLT_MIN - largest-subnormal == smallest-subnormal. An FTZ add instead
    // treats the second operand as zero and incorrectly returns FLT_MIN.
    const float min_normal = __uint_as_float(0x00800000U);
    const float neg_largest_subnormal = __uint_as_float(0x807FFFFFU);
    const fltflt result = fltflt_two_sum(min_normal, neg_largest_subnormal);

    result_bits[0] = __float_as_uint(result.hi);
    result_bits[1] = __float_as_uint(result.lo);

    const float min_subnormal = __uint_as_float(ftz_probe_bits);
    const fltflt tiny{min_subnormal, 0.0f};
    const fltflt neg_tiny = -tiny;
    result_bits[2] = __float_as_uint(neg_tiny.hi);

    const fltflt converted{min_subnormal_as_double};
    result_bits[3] = __float_as_uint(converted.hi);

    result_bits[4] = (tiny > fltflt{0.0f, 0.0f}) ? 1U : 0U;

    const fltflt floored = fltflt_floor(fltflt{__uint_as_float(0x80000001U), 0.0f});
    result_bits[5] = __float_as_uint(floored.hi);
    result_bits[6] = __float_as_uint(floored.lo);

    // Exercise a subnormal low component using normalized float-float values.
    const fltflt low_component{1.0f, min_subnormal};
    const fltflt neg_low_component = -low_component;
    result_bits[7] = __float_as_uint(neg_low_component.lo);
    result_bits[8] = (low_component > fltflt{1.0f, 0.0f}) ? 1U : 0U;

    // Consume the bit pattern as a runtime kernel argument so Clang cannot
    // constant-fold this premise check using ordinary C++ FP semantics.
    result_bits[9] = (__uint_as_float(ftz_probe_bits) == 0.0f) ? 1U : 0U;

    double_result_bits[0] = __double_as_longlong(static_cast<double>(tiny));
    double_result_bits[1] = __double_as_longlong(fltflt_to_double(tiny));

    // The subnormal low component contributes exactly at bit 27 of the FP64
    // significand and must survive both float-to-double widening operations.
    const fltflt subnormal_low{__uint_as_float(0x01800000U), min_subnormal};
    double_result_bits[2] = __double_as_longlong(static_cast<double>(subnormal_low));
    double_result_bits[3] = __double_as_longlong(fltflt_to_double(subnormal_low));

    // A flushed rsqrt input produces infinity and contaminates the refinement.
    const fltflt sqrt_result = fltflt_sqrt(tiny);
    result_bits[10] = __float_as_uint(sqrt_result.hi);
    result_bits[11] = __float_as_uint(sqrt_result.lo);

    // Exercise the subnormal zero guards in all three division overloads.
    const float largest_subnormal =
        __uint_as_float(0x00800000U - ftz_probe_bits);
    const fltflt subnormal_value{largest_subnormal, 0.0f};
    const fltflt div_fltflt_fltflt = fltflt_div(subnormal_value, subnormal_value);
    const fltflt div_fltflt_float = fltflt_div(subnormal_value, largest_subnormal);
    const fltflt div_float_fltflt = fltflt_div(largest_subnormal, subnormal_value);
    result_bits[12] = __float_as_uint(div_fltflt_fltflt.hi);
    result_bits[13] = __float_as_uint(div_fltflt_fltflt.lo);
    result_bits[14] = __float_as_uint(div_fltflt_float.hi);
    result_bits[15] = __float_as_uint(div_fltflt_float.lo);
    result_bits[16] = __float_as_uint(div_float_fltflt.hi);
    result_bits[17] = __float_as_uint(div_float_fltflt.lo);

    // Separately verify float-float sqrt precision away from the underflow
    // boundary. Keep the input runtime so the compiler cannot precompute it.
    const fltflt normal_sqrt_result =
        fltflt_sqrt(fltflt{normal_sqrt_input, 0.0f});
    result_bits[18] = __float_as_uint(normal_sqrt_result.hi);
    result_bits[19] = __float_as_uint(normal_sqrt_result.lo);
}

TEST(FloatFloatFastMathTests, PreservesGradualUnderflow)
{
    unsigned int *device_result_ptr = nullptr;
    ASSERT_EQ(cudaSuccess,
              cudaMalloc(&device_result_ptr, RESULT_COUNT * sizeof(unsigned int)));
    std::unique_ptr<unsigned int, CudaFreeDeleter> device_result{device_result_ptr};

    unsigned long long *device_double_result_ptr = nullptr;
    ASSERT_EQ(cudaSuccess,
              cudaMalloc(&device_double_result_ptr,
                         DOUBLE_RESULT_COUNT * sizeof(unsigned long long)));
    std::unique_ptr<unsigned long long, CudaFreeDeleter>
        device_double_result{device_double_result_ptr};

    FltFltGradualUnderflowKernel<<<1, 1>>>(device_result.get(), device_double_result.get(),
                                          0x1p-149, 0x00000001U, 2.0f);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());

    std::array<unsigned int, RESULT_COUNT> result_bits{};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(result_bits.data(), device_result.get(),
                         RESULT_COUNT * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
    EXPECT_EQ(0x00000001U, result_bits[0]);
    EXPECT_EQ(0x00000000U, result_bits[1]);
    EXPECT_EQ(0x80000001U, result_bits[2]);
    EXPECT_EQ(0x00000001U, result_bits[3]);
    EXPECT_EQ(0x00000001U, result_bits[4]);
    EXPECT_EQ(0xBF800000U, result_bits[5]);
    EXPECT_EQ(0x00000000U, result_bits[6]);
    EXPECT_EQ(0x80000001U, result_bits[7]);
    EXPECT_EQ(0x00000001U, result_bits[8]);
    EXPECT_EQ(0x00000001U, result_bits[9]);
    const double sqrt_result =
        static_cast<double>(std::bit_cast<float>(result_bits[10])) +
        static_cast<double>(std::bit_cast<float>(result_bits[11]));
    const double sqrt_reference = std::sqrt(0x1p-149);
    // At this boundary the refinement residual itself underflows in FP32, so
    // require FP32 accuracy while still detecting an FTZ-produced NaN.
    EXPECT_NEAR(sqrt_reference, sqrt_result, 0x1p-97); // Two FP32 ULPs.
    EXPECT_EQ(0x3F800000U, result_bits[12]);
    EXPECT_EQ(0x00000000U, result_bits[13]);
    EXPECT_EQ(0x3F800000U, result_bits[14]);
    EXPECT_EQ(0x00000000U, result_bits[15]);
    EXPECT_EQ(0x3F800000U, result_bits[16]);
    EXPECT_EQ(0x00000000U, result_bits[17]);
    const double normal_sqrt_result =
        static_cast<double>(std::bit_cast<float>(result_bits[18])) +
        static_cast<double>(std::bit_cast<float>(result_bits[19]));
    const double normal_sqrt_reference = std::sqrt(2.0);
    EXPECT_NEAR(normal_sqrt_reference, normal_sqrt_result,
                normal_sqrt_reference * 0x1p-44);

    std::array<unsigned long long, DOUBLE_RESULT_COUNT> double_result_bits{};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(double_result_bits.data(), device_double_result.get(),
                         DOUBLE_RESULT_COUNT * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
    EXPECT_EQ(0x36A0000000000000ULL, double_result_bits[0]);
    EXPECT_EQ(0x36A0000000000000ULL, double_result_bits[1]);
    EXPECT_EQ(0x3830000008000000ULL, double_result_bits[2]);
    EXPECT_EQ(0x3830000008000000ULL, double_result_bits[3]);

    EXPECT_EQ(cudaSuccess, cudaFree(device_result.release()));
    EXPECT_EQ(cudaSuccess, cudaFree(device_double_result.release()));
}
