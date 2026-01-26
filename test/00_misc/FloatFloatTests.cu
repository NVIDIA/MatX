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
#include "utilities.h"
#include "gtest/gtest.h"

#include <bit>

using namespace matx;

using FltFltExecutorTypes = TupleToTypes<cuda::std::tuple<matx::cudaExecutor, matx::SingleThreadedHostExecutor>>::type;

// GTest's generated TestBody() method has private/protected access, which causes an
// build error in the current nvcc ("an extended __host__ __device__ lambda cannot be defined inside...").
// Use file-scope functors instead.
struct FltFltAdd {
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, fltflt b) const
    {
      return static_cast<double>(a + b);
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(float a, fltflt b) const
    {
      return static_cast<double>(a + b);
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, float b) const
    {
      return static_cast<double>(a + b);
    }
};

struct FltFltMul {
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, fltflt b) const
    {
      return static_cast<double>(a * b);
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(float a, fltflt b) const
    {
      return static_cast<double>(a * b);
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, float b) const
    {
      return static_cast<double>(a * b);
    }
};

struct FltFltFma {
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, fltflt b, fltflt c) const
    {
      return static_cast<double>(fltflt_fma(a, b, c));
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(float a, fltflt b, fltflt c) const
    {
      return static_cast<double>(fltflt_fma(a, b, c));
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, float b, fltflt c) const
    {
      return static_cast<double>(fltflt_fma(a, b, c));
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, fltflt b, float c) const
    {
      return static_cast<double>(fltflt_fma(a, b, c));
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, float b, float c) const
    {
      return static_cast<double>(fltflt_fma(a, b, c));
    }
    __MATX_HOST__ __MATX_DEVICE__ double operator()(float a, fltflt b, float c) const
    {
      return static_cast<double>(fltflt_fma(a, b, c));
    }
};

struct FltFltSub {
  __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, fltflt b) const
  {
    return static_cast<double>(a - b);
  }
  __MATX_HOST__ __MATX_DEVICE__ double operator()(float a, fltflt b) const
  {
    return static_cast<double>(a - b);
  }
  __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, float b) const
  {
    return static_cast<double>(a - b);
  }
};

struct FltFltDiv {
  __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, fltflt b) const
  {
    return static_cast<double>(a / b);
  }
  __MATX_HOST__ __MATX_DEVICE__ double operator()(float a, fltflt b) const
  {
    return static_cast<double>(a / b);
  }
  __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a, float b) const
  {
    return static_cast<double>(a / b);
  }
};

struct FltFltSqrt {
  __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a) const
  {
    return static_cast<double>(static_cast<double>(fltflt_sqrt(a)));
  }
};

struct FltFltAbs {
  __MATX_HOST__ __MATX_DEVICE__ double operator()(fltflt a) const
  {
    return static_cast<double>(static_cast<double>(fltflt_abs(a)));
  }
};

struct FltFltCmpEq {
  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, fltflt b) const
  {
    return a == b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, float b) const
  {
    return a == b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(float a, fltflt b) const
  {
    return a == b;
  }
};

struct FltFltCmpNeq {
  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, fltflt b) const
  {
    return a != b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, float b) const
  {
    return a != b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(float a, fltflt b) const
  {
    return a != b;
  }
};

struct FltFltCmpLt {
  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, fltflt b) const
  {
    return a < b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, float b) const
  {
    return a < b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(float a, fltflt b) const
  {
    return a < b;
  }
};

struct FltFltCmpGt {
  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, fltflt b) const
  {
    return a > b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, float b) const
  {
    return a > b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(float a, fltflt b) const
  {
    return a > b;
  }
};

struct FltFltCmpLe {
  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, fltflt b) const
  {
    return a <= b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, float b) const
  {
    return a <= b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(float a, fltflt b) const
  {
    return a <= b;
  }
};

struct FltFltCmpGe {
  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, fltflt b) const
  {
    return a >= b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(fltflt a, float b) const
  {
    return a >= b;
  }

  __MATX_HOST__ __MATX_DEVICE__ bool operator()(float a, fltflt b) const
  {
    return a >= b;
  }
};

template <typename Exec>
class FltFltExecutorTests : public ::testing::Test {
protected:
  Exec exec{};
};

TYPED_TEST_SUITE(FltFltExecutorTests, FltFltExecutorTypes);

// The general strategy in the float-float test is as follows:
//   1. Compute an operation, such as addition, using float-float arithmetic based on
//      double precision inputs known to require more mantissa bits than offered even by
//      double (such as pi and e).
//   2. Compute the reference result using double precision arithmetic.
//   3. Compute the effective number of mantissa bits in the float-float result using
//      the numMatchingMantissaBits() function.
//   4. Repeat the effective mantissa bit calculation using single-precision arithmetic.
//   5. Verify that the effective number of mantissa bits for the single-precision result
//      is <= 24 (to verify that this is not a degenerate case for which the single-precision
//      result has high-precision, such as a case where all trailing bits are 0).
//   6. Verify that the effective number of mantissa bits for the float-float result
//      is >= 44. We keep the operations simple (e.g., a single float-float addition), so
//      we expect close to the ideal ~48 mantissa bits of precision.

static int numMatchingMantissaBits(double a, double b) {
    if (a == b) { return 53; /* exact match */ }

    int e_a, e_b;
    const double f_a = std::frexp(a, &e_a); // [0.5, 1)
    const double f_b = std::frexp(b, &e_b);

    uint64_t m_a = static_cast<uint64_t>(f_a * (1ULL << 53));
    uint64_t m_b = static_cast<uint64_t>(f_b * (1ULL << 53));

    int delta = e_a - e_b;
    if (delta > 0) {
        m_b >>= delta;
    } else if (delta < 0) {
        m_a >>= -delta;
    }

    // Shift the implicit mantissa bit to the most significant bit position. This
    // allows us to just count the leading zeros of the difference.
    const uint64_t diff = (m_a ^ m_b) << (64 - 53);

    const int leadingZeros = std::countl_zero(diff);
    const int matchingBits = std::min(leadingZeros, 53);
    return matchingBits + delta;
}

TYPED_TEST(FltFltExecutorTests, Addition) {
    auto pi = make_tensor<fltflt>({});
    auto one = make_tensor<float>({});
    (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (one = 1.0f).run(this->exec);
    auto add_result = make_tensor<double>({});
    // Add a left-hand side and right-hand side, resp., of 1.0 to the pi tensor.
    auto add_result_lhs1 = make_tensor<double>({});
    auto add_result_rhs1 = make_tensor<double>({});
    (add_result = matx::apply(FltFltAdd{}, pi, pi)).run(this->exec);
    (add_result_lhs1 = matx::apply(FltFltAdd{}, one, pi)).run(this->exec);
    (add_result_rhs1 = matx::apply(FltFltAdd{}, pi, one)).run(this->exec);
    this->exec.sync();

    // Whether we add 1.0 to the left or right, we should get the same result.
    EXPECT_EQ(add_result_lhs1(), add_result_rhs1());

    const double add_result_ref_f64 = std::numbers::pi + std::numbers::pi;
    const double add_result_ref_f32 = static_cast<float>(std::numbers::pi) + static_cast<float>(std::numbers::pi);
    const double pi_plus_1_ref_f64 = std::numbers::pi + 1.0;
    const float pi_plus_1_ref_f32 = static_cast<float>(std::numbers::pi) + 1.0f;

    // Verify that we only get up to 24 mantissa bits of precision from the single-precision results
    EXPECT_LE(numMatchingMantissaBits(add_result_ref_f32, add_result_ref_f64), 24);
    EXPECT_LE(numMatchingMantissaBits(pi_plus_1_ref_f32, pi_plus_1_ref_f64), 24);

    // Verify that we get >= 44 mantissa bits of precision for the float-float results
    EXPECT_GE(numMatchingMantissaBits(add_result(), add_result_ref_f64), 44);
    EXPECT_GE(numMatchingMantissaBits(add_result_lhs1(), pi_plus_1_ref_f64), 44);
    EXPECT_GE(numMatchingMantissaBits(add_result_rhs1(), pi_plus_1_ref_f64), 44);
}

TYPED_TEST(FltFltExecutorTests, Multiplication) {
    auto pi = make_tensor<fltflt>({});
    auto pi_f32 = make_tensor<float>({});
    (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (pi_f32 = std::numbers::pi_v<float>).run(this->exec);
    auto mul_result = make_tensor<double>({});
    auto mul_result_lhs_f32 = make_tensor<double>({});
    auto mul_result_rhs_f32 = make_tensor<double>({});
    (mul_result = matx::apply(FltFltMul{}, pi, pi)).run(this->exec);
    (mul_result_lhs_f32 = matx::apply(FltFltMul{}, pi_f32, pi)).run(this->exec);
    (mul_result_rhs_f32 = matx::apply(FltFltMul{}, pi, pi_f32)).run(this->exec);
    this->exec.sync();

    const double pi_sqr_ref_f64 = std::numbers::pi * std::numbers::pi;
    const double pi_sqr_ref_f32 = std::numbers::pi_v<float> * std::numbers::pi_v<float>;
    const double pi_sqr_ref_one_f32 = std::numbers::pi_v<float> * std::numbers::pi;

    // We expect equivalent results with float * fltflt and fltflt * float
    EXPECT_EQ(mul_result_lhs_f32(), mul_result_rhs_f32());

    EXPECT_LE(numMatchingMantissaBits(mul_result_lhs_f32(), pi_sqr_ref_f32), 24);

    EXPECT_GE(numMatchingMantissaBits(mul_result(), pi_sqr_ref_f64), 44);
    EXPECT_GE(numMatchingMantissaBits(mul_result_lhs_f32(), pi_sqr_ref_one_f32), 44);
    EXPECT_GE(numMatchingMantissaBits(mul_result_rhs_f32(), pi_sqr_ref_one_f32), 44);
}

TYPED_TEST(FltFltExecutorTests, FusedMultiplyAdd) {
    auto pi = make_tensor<fltflt>({});
    auto e = make_tensor<fltflt>({});
    auto sqrt2 = make_tensor<fltflt>({});
    auto pi_f32 = make_tensor<float>({});
    auto e_f32 = make_tensor<float>({});
    auto sqrt2_f32 = make_tensor<float>({});
    (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);
    (sqrt2 = static_cast<fltflt>(std::numbers::sqrt2)).run(this->exec);
    (pi_f32 = std::numbers::pi_v<float>).run(this->exec);
    (e_f32 = std::numbers::e_v<float>).run(this->exec);
    (sqrt2_f32 = std::numbers::sqrt2_v<float>).run(this->exec);

    auto fma_result = make_tensor<double>({});
    auto fma_result_a_f32 = make_tensor<double>({});
    auto fma_result_b_f32 = make_tensor<double>({});
    auto fma_result_c_f32 = make_tensor<double>({});
    auto fma_result_bc_f32 = make_tensor<double>({});
    auto fma_result_ac_f32 = make_tensor<double>({});
    (fma_result = matx::apply(FltFltFma{}, pi, e, sqrt2)).run(this->exec);
    (fma_result_a_f32 = matx::apply(FltFltFma{}, pi_f32, e, sqrt2)).run(this->exec);
    (fma_result_b_f32 = matx::apply(FltFltFma{}, pi, e_f32, sqrt2)).run(this->exec);
    (fma_result_c_f32 = matx::apply(FltFltFma{}, pi, e, sqrt2_f32)).run(this->exec);
    (fma_result_bc_f32 = matx::apply(FltFltFma{}, pi, e_f32, sqrt2_f32)).run(this->exec);
    (fma_result_ac_f32 = matx::apply(FltFltFma{}, pi_f32, e, sqrt2_f32)).run(this->exec);
    this->exec.sync();

    const double fma_ref_f64 = std::numbers::pi * std::numbers::e + std::numbers::sqrt2;
    const double fma_ref_f32 = std::numbers::pi_v<float> * std::numbers::e_v<float> + std::numbers::sqrt2_v<float>;
    const double fma_ref_a_f32 = std::numbers::pi_v<float> * std::numbers::e + std::numbers::sqrt2;
    const double fma_ref_b_f32 = std::numbers::pi * std::numbers::e_v<float> + std::numbers::sqrt2;
    const double fma_ref_c_f32 = std::numbers::pi * std::numbers::e + std::numbers::sqrt2_v<float>;
    const double fma_ref_bc_f32 = std::numbers::pi * std::numbers::e_v<float> + std::numbers::sqrt2_v<float>;
    const double fma_ref_ac_f32 = std::numbers::pi_v<float> * std::numbers::e + std::numbers::sqrt2_v<float>;

    EXPECT_LE(numMatchingMantissaBits(fma_ref_f32, fma_ref_f64), 24);

    EXPECT_GE(numMatchingMantissaBits(fma_result(), fma_ref_f64), 44);
    EXPECT_GE(numMatchingMantissaBits(fma_result_a_f32(), fma_ref_a_f32), 44);
    EXPECT_GE(numMatchingMantissaBits(fma_result_b_f32(), fma_ref_b_f32), 44);
    EXPECT_GE(numMatchingMantissaBits(fma_result_c_f32(), fma_ref_c_f32), 44);
    EXPECT_GE(numMatchingMantissaBits(fma_result_bc_f32(), fma_ref_bc_f32), 44);
    EXPECT_GE(numMatchingMantissaBits(fma_result_ac_f32(), fma_ref_ac_f32), 44);
}

TYPED_TEST(FltFltExecutorTests, Subtraction) {
  auto pi = make_tensor<fltflt>({});
  auto e = make_tensor<fltflt>({});
  auto e_f32 = make_tensor<float>({});
  (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
  (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);
  (e_f32 = std::numbers::e_v<float>).run(this->exec);
  auto sub_result = make_tensor<double>({});
  auto sub_result_lhs_f32 = make_tensor<double>({});
  auto sub_result_rhs_f32 = make_tensor<double>({});
  (sub_result = matx::apply(FltFltSub{}, pi, e)).run(this->exec);
  (sub_result_lhs_f32 = matx::apply(FltFltSub{}, e_f32, pi)).run(this->exec);
  (sub_result_rhs_f32 = matx::apply(FltFltSub{}, pi, e_f32)).run(this->exec);
  this->exec.sync();

  const double pi_minus_e_ref_f64 = std::numbers::pi - std::numbers::e;
  const double pi_minus_e_ref_f32 = std::numbers::pi_v<float> - std::numbers::e_v<float>;
  const double pi_e_f32_minus_pi_ref = std::numbers::e_v<float> - std::numbers::pi;
  const double pi_minus_e_f32_ref = std::numbers::pi - std::numbers::e_v<float>;

  EXPECT_LE(numMatchingMantissaBits(pi_minus_e_ref_f32, pi_minus_e_ref_f64), 24);

  EXPECT_GE(numMatchingMantissaBits(sub_result(), pi_minus_e_ref_f64), 44);
  EXPECT_GE(numMatchingMantissaBits(sub_result_lhs_f32(), pi_e_f32_minus_pi_ref), 44);
  EXPECT_GE(numMatchingMantissaBits(sub_result_rhs_f32(), pi_minus_e_f32_ref), 44);

  // Also test unary negation
  const fltflt val1{1.0f, 2.0e-7f};
  const fltflt val2{-1.0f, -2.0e-7f};
  const fltflt neg_val1 = -val1;
  EXPECT_EQ(neg_val1, val2);
  EXPECT_EQ(neg_val1.hi, -val1.hi);
  EXPECT_EQ(neg_val1.lo, -val1.lo);
}

TYPED_TEST(FltFltExecutorTests, Division) {
  auto pi = make_tensor<fltflt>({});
  auto e = make_tensor<fltflt>({});
  auto pi_f32 = make_tensor<float>({});
  auto e_f32 = make_tensor<float>({});
  (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
  (pi_f32 = std::numbers::pi_v<float>).run(this->exec);
  (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);
  (e_f32 = std::numbers::e_v<float>).run(this->exec);

  auto div_result = make_tensor<double>({});
  auto div_result_num_f32 = make_tensor<double>({});
  auto div_result_denom_f32 = make_tensor<double>({});
  (div_result = matx::apply(FltFltDiv{}, pi, e)).run(this->exec);
  (div_result_num_f32 = matx::apply(FltFltDiv{}, pi_f32, e)).run(this->exec);
  (div_result_denom_f32 = matx::apply(FltFltDiv{}, pi, e_f32)).run(this->exec);
  this->exec.sync();

  const double pi_div_e_ref_f64 = std::numbers::pi / std::numbers::e;
  const double pi_div_e_ref_f32 = std::numbers::pi_v<float> / std::numbers::e_v<float>;
  const double pi_f32_div_e_ref = std::numbers::pi_v<float> / std::numbers::e;
  const double pi_div_e_f32_ref = std::numbers::pi / std::numbers::e_v<float>;

  EXPECT_LE(numMatchingMantissaBits(pi_div_e_ref_f32, pi_div_e_ref_f64), 24);

  EXPECT_GE(numMatchingMantissaBits(div_result(), pi_div_e_ref_f64), 44);
  EXPECT_GE(numMatchingMantissaBits(div_result_num_f32(), pi_f32_div_e_ref), 44);
  EXPECT_GE(numMatchingMantissaBits(div_result_denom_f32(), pi_div_e_f32_ref), 44);
}

TYPED_TEST(FltFltExecutorTests, SquareRoot) {
  auto pi = make_tensor<fltflt>({});
  (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);

  auto sqrt_result = make_tensor<double>({});
  (sqrt_result = matx::apply(FltFltSqrt{}, pi)).run(this->exec);
  this->exec.sync();

  const double pi_sqrt_ref_f64 = std::sqrt(std::numbers::pi);
  const float pi_sqrt_ref_f32 = std::sqrt(std::numbers::pi_v<float>);

  EXPECT_LE(numMatchingMantissaBits(pi_sqrt_ref_f32, pi_sqrt_ref_f64), 24);

  EXPECT_GE(numMatchingMantissaBits(sqrt_result(), pi_sqrt_ref_f64), 44);
}

TYPED_TEST(FltFltExecutorTests, MatXSqrtOperator) {
  auto pi = make_tensor<fltflt>({});
  (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);

  auto sqrt_result = make_tensor<fltflt>({});
  (sqrt_result = sqrt(pi)).run(this->exec);
  this->exec.sync();

  const double pi_sqrt_ref_f64 = std::sqrt(std::numbers::pi);
  const float pi_sqrt_ref_f32 = std::sqrt(std::numbers::pi_v<float>);

  // Ensure this is a non-degenerate case for fp32.
  EXPECT_LE(numMatchingMantissaBits(pi_sqrt_ref_f32, pi_sqrt_ref_f64), 24);

  // Expect float-float sqrt to retain high precision.
  EXPECT_GE(numMatchingMantissaBits(static_cast<double>(sqrt_result()), pi_sqrt_ref_f64), 44);
}

TYPED_TEST(FltFltExecutorTests, AbsoluteValue) {
  auto pi = make_tensor<fltflt>({});
  auto neg_pi = make_tensor<fltflt>({});
  auto e = make_tensor<fltflt>({});
  auto neg_e = make_tensor<fltflt>({});
  (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
  (neg_pi = static_cast<fltflt>(-1.0 * std::numbers::pi)).run(this->exec);
  (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);
  (neg_e = static_cast<fltflt>(-1.0 * std::numbers::e)).run(this->exec);

  auto abs_result_pi = make_tensor<double>({});
  auto abs_result_neg_pi = make_tensor<double>({});
  auto abs_result_e = make_tensor<double>({});
  auto abs_result_neg_e = make_tensor<double>({});
  (abs_result_pi = matx::apply(FltFltAbs{}, pi)).run(this->exec);
  (abs_result_neg_pi = matx::apply(FltFltAbs{}, neg_pi)).run(this->exec);
  (abs_result_e = matx::apply(FltFltAbs{}, e)).run(this->exec);
  (abs_result_neg_e = matx::apply(FltFltAbs{}, neg_e)).run(this->exec);
  this->exec.sync();

  // The float-float representation of pi has a positive hi value and negative lo value. It is
  // approximately 3.141593e+00 + -8.742278e-08. The representation of e is 2.718282e+00 + 8.254840e-08,
  // so both terms are positive. We test that taking the absolute value of the negative of both values
  // provides the expected results.

  EXPECT_GE(numMatchingMantissaBits(abs_result_pi(), std::numbers::pi), 44);
  EXPECT_GE(numMatchingMantissaBits(abs_result_neg_pi(), std::numbers::pi), 44);
  EXPECT_GE(numMatchingMantissaBits(abs_result_e(), std::numbers::e), 44);
  EXPECT_GE(numMatchingMantissaBits(abs_result_neg_e(), std::numbers::e), 44);
}

TYPED_TEST(FltFltExecutorTests, CmpEq) {
  {
    // First, test negative cases where values differ
    auto pi_f64 = make_tensor<fltflt>({});
    auto pi_f32 = make_tensor<fltflt>({});
    auto zero = make_tensor<fltflt>({});
    (pi_f64 = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (pi_f32 = static_cast<fltflt>(std::numbers::pi_v<float>)).run(this->exec);
    (zero = static_cast<fltflt>(0.0)).run(this->exec);

    auto eq_result_pi_eq_0 = make_tensor<bool>({});
    auto eq_result_f32_f64 = make_tensor<bool>({});

    (eq_result_pi_eq_0 = matx::apply(FltFltCmpEq{}, pi_f64, zero)).run(this->exec);
    (eq_result_f32_f64 = matx::apply(FltFltCmpEq{}, pi_f32, pi_f64)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(eq_result_pi_eq_0(), false);
    EXPECT_EQ(eq_result_f32_f64(), false);

    // Now, test positive cases where values are the same
    auto eq_result_f32 = make_tensor<bool>({});
    auto eq_result_f64 = make_tensor<bool>({});
    (eq_result_f32 = matx::apply(FltFltCmpEq{}, pi_f32, pi_f32)).run(this->exec);
    (eq_result_f64 = matx::apply(FltFltCmpEq{}, pi_f64, pi_f64)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(eq_result_f32(), true);
    EXPECT_EQ(eq_result_f64(), true);

  }

  // Now, test cases where the hi component is the same, but the lo component is different.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto eq_ab = make_tensor<bool>({});
    auto eq_ba = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_lo_1{hi, 1.0e-7f};
    const fltflt ff_lo_2{hi, 2.0e-7f};

    (a = ff_lo_1).run(this->exec);
    (b = ff_lo_2).run(this->exec);
    (eq_ab = matx::apply(FltFltCmpEq{}, a, b)).run(this->exec);
    (eq_ba = matx::apply(FltFltCmpEq{}, b, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(eq_ab(), false);
    EXPECT_EQ(eq_ba(), false);

    // Exact match should compare equal.
    (b = ff_lo_1).run(this->exec);
    (eq_ab = matx::apply(FltFltCmpEq{}, a, b)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(eq_ab(), true);
  }

  // +0.0 and -0.0 compare equal in IEEE-754; ensure we follow float semantics.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto eq_ab = make_tensor<bool>({});

    const float hi = 1.0f;
    const fltflt ff_pos0{hi, 0.0f};
    const fltflt ff_neg0{hi, -0.0f};
    (a = ff_pos0).run(this->exec);
    (b = ff_neg0).run(this->exec);
    (eq_ab = matx::apply(FltFltCmpEq{}, a, b)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(eq_ab(), true);
  }

  // Mixed fltflt <-> float comparisons.
  {
    auto a = make_tensor<fltflt>({});
    auto f = make_tensor<float>({});
    auto eq_af = make_tensor<bool>({});
    auto eq_fa = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_exact_f{hi, 0.0f};
    const fltflt ff_inexact{hi, 1.0e-7f};

    // lo == 0 should compare equal to the float.
    (a = ff_exact_f).run(this->exec);
    (f = hi).run(this->exec);
    (eq_af = matx::apply(FltFltCmpEq{}, a, f)).run(this->exec);
    (eq_fa = matx::apply(FltFltCmpEq{}, f, a)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(eq_af(), true);
    EXPECT_EQ(eq_fa(), true);

    // lo != 0 should compare not-equal to the float, even if hi matches.
    (a = ff_inexact).run(this->exec);
    (eq_af = matx::apply(FltFltCmpEq{}, a, f)).run(this->exec);
    (eq_fa = matx::apply(FltFltCmpEq{}, f, a)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(eq_af(), false);
    EXPECT_EQ(eq_fa(), false);
  }
}

TYPED_TEST(FltFltExecutorTests, CmpNeq) {
  {
    // First, test cases where values differ
    auto pi_f64 = make_tensor<fltflt>({});
    auto pi_f32 = make_tensor<fltflt>({});
    auto zero = make_tensor<fltflt>({});
    (pi_f64 = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (pi_f32 = static_cast<fltflt>(std::numbers::pi_v<float>)).run(this->exec);
    (zero = static_cast<fltflt>(0.0)).run(this->exec);

    auto neq_result_pi_neq_0 = make_tensor<bool>({});
    auto neq_result_f32_f64 = make_tensor<bool>({});
    (neq_result_pi_neq_0 = matx::apply(FltFltCmpNeq{}, pi_f64, zero)).run(this->exec);
    (neq_result_f32_f64 = matx::apply(FltFltCmpNeq{}, pi_f32, pi_f64)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(neq_result_pi_neq_0(), true);
    EXPECT_EQ(neq_result_f32_f64(), true);

    // Now, test cases where values are the same
    auto neq_result_f32 = make_tensor<bool>({});
    auto neq_result_f64 = make_tensor<bool>({});
    (neq_result_f32 = matx::apply(FltFltCmpNeq{}, pi_f32, pi_f32)).run(this->exec);
    (neq_result_f64 = matx::apply(FltFltCmpNeq{}, pi_f64, pi_f64)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(neq_result_f32(), false);
    EXPECT_EQ(neq_result_f64(), false);
  }

  // Now, test cases where the hi component is the same, but the lo component is different.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto neq_ab = make_tensor<bool>({});
    auto neq_ba = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_lo_1{hi, 1.0e-7f};
    const fltflt ff_lo_2{hi, 2.0e-7f};

    (a = ff_lo_1).run(this->exec);
    (b = ff_lo_2).run(this->exec);
    (neq_ab = matx::apply(FltFltCmpNeq{}, a, b)).run(this->exec);
    (neq_ba = matx::apply(FltFltCmpNeq{}, b, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(neq_ab(), true);
    EXPECT_EQ(neq_ba(), true);

    // Exact match should compare not-equal as false.
    (b = ff_lo_1).run(this->exec);
    (neq_ab = matx::apply(FltFltCmpNeq{}, a, b)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(neq_ab(), false);
  }

  // +0.0 and -0.0 compare equal in IEEE-754; ensure != follows float semantics.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto neq_ab = make_tensor<bool>({});

    const float hi = 1.0f;
    const fltflt ff_pos0{hi, 0.0f};
    const fltflt ff_neg0{hi, -0.0f};
    (a = ff_pos0).run(this->exec);
    (b = ff_neg0).run(this->exec);
    (neq_ab = matx::apply(FltFltCmpNeq{}, a, b)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(neq_ab(), false);
  }

  // Mixed fltflt <-> float comparisons.
  {
    auto a = make_tensor<fltflt>({});
    auto f = make_tensor<float>({});
    auto neq_af = make_tensor<bool>({});
    auto neq_fa = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_exact_f{hi, 0.0f};
    const fltflt ff_inexact{hi, 1.0e-7f};

    // lo == 0 should compare not-equal as false.
    (a = ff_exact_f).run(this->exec);
    (f = hi).run(this->exec);
    (neq_af = matx::apply(FltFltCmpNeq{}, a, f)).run(this->exec);
    (neq_fa = matx::apply(FltFltCmpNeq{}, f, a)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(neq_af(), false);
    EXPECT_EQ(neq_fa(), false);

    // lo != 0 should compare not-equal as true, even if hi matches.
    (a = ff_inexact).run(this->exec);
    (neq_af = matx::apply(FltFltCmpNeq{}, a, f)).run(this->exec);
    (neq_fa = matx::apply(FltFltCmpNeq{}, f, a)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(neq_af(), true);
    EXPECT_EQ(neq_fa(), true);
  }
}

TYPED_TEST(FltFltExecutorTests, CmpLt) {
  // Basic cases that depend primarily on hi (and ensure strict-weak ordering).
  {
    auto pi = make_tensor<fltflt>({});
    auto e = make_tensor<fltflt>({});
    (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);

    auto lt_pi_e = make_tensor<bool>({});
    auto lt_e_pi = make_tensor<bool>({});
    auto lt_pi_pi = make_tensor<bool>({});
    (lt_pi_e = matx::apply(FltFltCmpLt{}, pi, e)).run(this->exec);
    (lt_e_pi = matx::apply(FltFltCmpLt{}, e, pi)).run(this->exec);
    (lt_pi_pi = matx::apply(FltFltCmpLt{}, pi, pi)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(lt_pi_e(), false);
    EXPECT_EQ(lt_e_pi(), true);
    EXPECT_EQ(lt_pi_pi(), false);
  }

  // Same hi, different lo should order by lo.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto lt_ab = make_tensor<bool>({});
    auto lt_ba = make_tensor<bool>({});
    auto lt_aa = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_lo_1{hi, 1.0e-7f};
    const fltflt ff_lo_2{hi, 2.0e-7f};
    (a = ff_lo_1).run(this->exec);
    (b = ff_lo_2).run(this->exec);
    (lt_ab = matx::apply(FltFltCmpLt{}, a, b)).run(this->exec);
    (lt_ba = matx::apply(FltFltCmpLt{}, b, a)).run(this->exec);
    (lt_aa = matx::apply(FltFltCmpLt{}, a, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(lt_ab(), true);
    EXPECT_EQ(lt_ba(), false);
    EXPECT_EQ(lt_aa(), false);
  }

  // +0.0 and -0.0 are equal; neither should be less-than the other.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto lt_ab = make_tensor<bool>({});
    auto lt_ba = make_tensor<bool>({});

    const float hi = 1.0f;
    const fltflt ff_pos0{hi, 0.0f};
    const fltflt ff_neg0{hi, -0.0f};
    (a = ff_pos0).run(this->exec);
    (b = ff_neg0).run(this->exec);
    (lt_ab = matx::apply(FltFltCmpLt{}, a, b)).run(this->exec);
    (lt_ba = matx::apply(FltFltCmpLt{}, b, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(lt_ab(), false);
    EXPECT_EQ(lt_ba(), false);
  }

  // Mixed fltflt <-> float comparisons. When hi matches, the sign of lo determines ordering.
  {
    auto ff = make_tensor<fltflt>({});
    auto f = make_tensor<float>({});
    auto lt_ff_f = make_tensor<bool>({});
    auto lt_f_ff = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    (f = hi).run(this->exec);

    // lo == 0: neither is less than the other.
    (ff = fltflt{hi, 0.0f}).run(this->exec);
    (lt_ff_f = matx::apply(FltFltCmpLt{}, ff, f)).run(this->exec);
    (lt_f_ff = matx::apply(FltFltCmpLt{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(lt_ff_f(), false);
    EXPECT_EQ(lt_f_ff(), false);

    // lo > 0: float(hi) is less than fltflt(hi, +eps)
    (ff = fltflt{hi, 1.0e-7f}).run(this->exec);
    (lt_ff_f = matx::apply(FltFltCmpLt{}, ff, f)).run(this->exec);
    (lt_f_ff = matx::apply(FltFltCmpLt{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(lt_ff_f(), false);
    EXPECT_EQ(lt_f_ff(), true);

    // lo < 0: fltflt(hi, -eps) is less than float(hi)
    (ff = fltflt{hi, -1.0e-7f}).run(this->exec);
    (lt_ff_f = matx::apply(FltFltCmpLt{}, ff, f)).run(this->exec);
    (lt_f_ff = matx::apply(FltFltCmpLt{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(lt_ff_f(), true);
    EXPECT_EQ(lt_f_ff(), false);
  }
}

TYPED_TEST(FltFltExecutorTests, CmpGt) {
  // Basic cases that depend primarily on hi (and ensure strict-weak ordering).
  {
    auto pi = make_tensor<fltflt>({});
    auto e = make_tensor<fltflt>({});
    (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);

    auto gt_pi_e = make_tensor<bool>({});
    auto gt_e_pi = make_tensor<bool>({});
    auto gt_pi_pi = make_tensor<bool>({});
    (gt_pi_e = matx::apply(FltFltCmpGt{}, pi, e)).run(this->exec);
    (gt_e_pi = matx::apply(FltFltCmpGt{}, e, pi)).run(this->exec);
    (gt_pi_pi = matx::apply(FltFltCmpGt{}, pi, pi)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(gt_pi_e(), true);
    EXPECT_EQ(gt_e_pi(), false);
    EXPECT_EQ(gt_pi_pi(), false);
  }

  // Same hi, different lo should order by lo.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto gt_ab = make_tensor<bool>({});
    auto gt_ba = make_tensor<bool>({});
    auto gt_aa = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_lo_1{hi, 1.0e-7f};
    const fltflt ff_lo_2{hi, 2.0e-7f};
    (a = ff_lo_1).run(this->exec);
    (b = ff_lo_2).run(this->exec);
    (gt_ab = matx::apply(FltFltCmpGt{}, a, b)).run(this->exec);
    (gt_ba = matx::apply(FltFltCmpGt{}, b, a)).run(this->exec);
    (gt_aa = matx::apply(FltFltCmpGt{}, a, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(gt_ab(), false);
    EXPECT_EQ(gt_ba(), true);
    EXPECT_EQ(gt_aa(), false);
  }

  // +0.0 and -0.0 are equal; neither should be greater-than the other.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto gt_ab = make_tensor<bool>({});
    auto gt_ba = make_tensor<bool>({});

    const float hi = 1.0f;
    const fltflt ff_pos0{hi, 0.0f};
    const fltflt ff_neg0{hi, -0.0f};
    (a = ff_pos0).run(this->exec);
    (b = ff_neg0).run(this->exec);
    (gt_ab = matx::apply(FltFltCmpGt{}, a, b)).run(this->exec);
    (gt_ba = matx::apply(FltFltCmpGt{}, b, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(gt_ab(), false);
    EXPECT_EQ(gt_ba(), false);
  }

  // Mixed fltflt <-> float comparisons. When hi matches, the sign of lo determines ordering.
  {
    auto ff = make_tensor<fltflt>({});
    auto f = make_tensor<float>({});
    auto gt_ff_f = make_tensor<bool>({});
    auto gt_f_ff = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    (f = hi).run(this->exec);

    // lo == 0: neither is greater than the other.
    (ff = fltflt{hi, 0.0f}).run(this->exec);
    (gt_ff_f = matx::apply(FltFltCmpGt{}, ff, f)).run(this->exec);
    (gt_f_ff = matx::apply(FltFltCmpGt{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(gt_ff_f(), false);
    EXPECT_EQ(gt_f_ff(), false);

    // lo > 0: fltflt(hi, +eps) is greater than float(hi)
    (ff = fltflt{hi, 1.0e-7f}).run(this->exec);
    (gt_ff_f = matx::apply(FltFltCmpGt{}, ff, f)).run(this->exec);
    (gt_f_ff = matx::apply(FltFltCmpGt{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(gt_ff_f(), true);
    EXPECT_EQ(gt_f_ff(), false);

    // lo < 0: float(hi) is greater than fltflt(hi, -eps)
    (ff = fltflt{hi, -1.0e-7f}).run(this->exec);
    (gt_ff_f = matx::apply(FltFltCmpGt{}, ff, f)).run(this->exec);
    (gt_f_ff = matx::apply(FltFltCmpGt{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(gt_ff_f(), false);
    EXPECT_EQ(gt_f_ff(), true);
  }
}

TYPED_TEST(FltFltExecutorTests, CmpLe) {
  // Basic cases that depend primarily on hi.
  {
    auto pi = make_tensor<fltflt>({});
    auto e = make_tensor<fltflt>({});
    (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);

    auto le_e_pi = make_tensor<bool>({});
    auto le_pi_e = make_tensor<bool>({});
    auto le_pi_pi = make_tensor<bool>({});
    (le_e_pi = matx::apply(FltFltCmpLe{}, e, pi)).run(this->exec);
    (le_pi_e = matx::apply(FltFltCmpLe{}, pi, e)).run(this->exec);
    (le_pi_pi = matx::apply(FltFltCmpLe{}, pi, pi)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(le_e_pi(), true);
    EXPECT_EQ(le_pi_e(), false);
    EXPECT_EQ(le_pi_pi(), true);
  }

  // Same hi, different lo should order by lo.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto le_ab = make_tensor<bool>({});
    auto le_ba = make_tensor<bool>({});
    auto le_aa = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_lo_1{hi, 1.0e-7f};
    const fltflt ff_lo_2{hi, 2.0e-7f};
    (a = ff_lo_1).run(this->exec);
    (b = ff_lo_2).run(this->exec);
    (le_ab = matx::apply(FltFltCmpLe{}, a, b)).run(this->exec);
    (le_ba = matx::apply(FltFltCmpLe{}, b, a)).run(this->exec);
    (le_aa = matx::apply(FltFltCmpLe{}, a, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(le_ab(), true);
    EXPECT_EQ(le_ba(), false);
    EXPECT_EQ(le_aa(), true);
  }

  // +0.0 and -0.0 are equal; both should be <= each other.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto le_ab = make_tensor<bool>({});
    auto le_ba = make_tensor<bool>({});

    const float hi = 1.0f;
    const fltflt ff_pos0{hi, 0.0f};
    const fltflt ff_neg0{hi, -0.0f};
    (a = ff_pos0).run(this->exec);
    (b = ff_neg0).run(this->exec);
    (le_ab = matx::apply(FltFltCmpLe{}, a, b)).run(this->exec);
    (le_ba = matx::apply(FltFltCmpLe{}, b, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(le_ab(), true);
    EXPECT_EQ(le_ba(), true);
  }

  // Mixed fltflt <-> float comparisons. When hi matches, the sign of lo determines ordering.
  {
    auto ff = make_tensor<fltflt>({});
    auto f = make_tensor<float>({});
    auto le_ff_f = make_tensor<bool>({});
    auto le_f_ff = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    (f = hi).run(this->exec);

    // lo == 0: both directions should be <= (equal).
    (ff = fltflt{hi, 0.0f}).run(this->exec);
    (le_ff_f = matx::apply(FltFltCmpLe{}, ff, f)).run(this->exec);
    (le_f_ff = matx::apply(FltFltCmpLe{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(le_ff_f(), true);
    EXPECT_EQ(le_f_ff(), true);

    // lo > 0: fltflt(hi, +eps) is greater than float(hi)
    (ff = fltflt{hi, 1.0e-7f}).run(this->exec);
    (le_ff_f = matx::apply(FltFltCmpLe{}, ff, f)).run(this->exec);
    (le_f_ff = matx::apply(FltFltCmpLe{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(le_ff_f(), false);
    EXPECT_EQ(le_f_ff(), true);

    // lo < 0: fltflt(hi, -eps) is less than float(hi)
    (ff = fltflt{hi, -1.0e-7f}).run(this->exec);
    (le_ff_f = matx::apply(FltFltCmpLe{}, ff, f)).run(this->exec);
    (le_f_ff = matx::apply(FltFltCmpLe{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(le_ff_f(), true);
    EXPECT_EQ(le_f_ff(), false);
  }
}

TYPED_TEST(FltFltExecutorTests, CmpGe) {
  // Basic cases that depend primarily on hi.
  {
    auto pi = make_tensor<fltflt>({});
    auto e = make_tensor<fltflt>({});
    (pi = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
    (e = static_cast<fltflt>(std::numbers::e)).run(this->exec);

    auto ge_pi_e = make_tensor<bool>({});
    auto ge_e_pi = make_tensor<bool>({});
    auto ge_pi_pi = make_tensor<bool>({});
    (ge_pi_e = matx::apply(FltFltCmpGe{}, pi, e)).run(this->exec);
    (ge_e_pi = matx::apply(FltFltCmpGe{}, e, pi)).run(this->exec);
    (ge_pi_pi = matx::apply(FltFltCmpGe{}, pi, pi)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(ge_pi_e(), true);
    EXPECT_EQ(ge_e_pi(), false);
    EXPECT_EQ(ge_pi_pi(), true);
  }

  // Same hi, different lo should order by lo.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto ge_ab = make_tensor<bool>({});
    auto ge_ba = make_tensor<bool>({});
    auto ge_aa = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    const fltflt ff_lo_1{hi, 1.0e-7f};
    const fltflt ff_lo_2{hi, 2.0e-7f};
    (a = ff_lo_1).run(this->exec);
    (b = ff_lo_2).run(this->exec);
    (ge_ab = matx::apply(FltFltCmpGe{}, a, b)).run(this->exec);
    (ge_ba = matx::apply(FltFltCmpGe{}, b, a)).run(this->exec);
    (ge_aa = matx::apply(FltFltCmpGe{}, a, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(ge_ab(), false);
    EXPECT_EQ(ge_ba(), true);
    EXPECT_EQ(ge_aa(), true);
  }

  // +0.0 and -0.0 are equal; both should be >= each other.
  {
    auto a = make_tensor<fltflt>({});
    auto b = make_tensor<fltflt>({});
    auto ge_ab = make_tensor<bool>({});
    auto ge_ba = make_tensor<bool>({});

    const float hi = 1.0f;
    const fltflt ff_pos0{hi, 0.0f};
    const fltflt ff_neg0{hi, -0.0f};
    (a = ff_pos0).run(this->exec);
    (b = ff_neg0).run(this->exec);
    (ge_ab = matx::apply(FltFltCmpGe{}, a, b)).run(this->exec);
    (ge_ba = matx::apply(FltFltCmpGe{}, b, a)).run(this->exec);
    this->exec.sync();

    EXPECT_EQ(ge_ab(), true);
    EXPECT_EQ(ge_ba(), true);
  }

  // Mixed fltflt <-> float comparisons. When hi matches, the sign of lo determines ordering.
  {
    auto ff = make_tensor<fltflt>({});
    auto f = make_tensor<float>({});
    auto ge_ff_f = make_tensor<bool>({});
    auto ge_f_ff = make_tensor<bool>({});

    const float hi = std::numbers::pi_v<float>;
    (f = hi).run(this->exec);

    // lo == 0: both directions should be >= (equal).
    (ff = fltflt{hi, 0.0f}).run(this->exec);
    (ge_ff_f = matx::apply(FltFltCmpGe{}, ff, f)).run(this->exec);
    (ge_f_ff = matx::apply(FltFltCmpGe{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(ge_ff_f(), true);
    EXPECT_EQ(ge_f_ff(), true);

    // lo > 0: fltflt(hi, +eps) is greater than float(hi)
    (ff = fltflt{hi, 1.0e-7f}).run(this->exec);
    (ge_ff_f = matx::apply(FltFltCmpGe{}, ff, f)).run(this->exec);
    (ge_f_ff = matx::apply(FltFltCmpGe{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(ge_ff_f(), true);
    EXPECT_EQ(ge_f_ff(), false);

    // lo < 0: fltflt(hi, -eps) is less than float(hi)
    (ff = fltflt{hi, -1.0e-7f}).run(this->exec);
    (ge_ff_f = matx::apply(FltFltCmpGe{}, ff, f)).run(this->exec);
    (ge_f_ff = matx::apply(FltFltCmpGe{}, f, ff)).run(this->exec);
    this->exec.sync();
    EXPECT_EQ(ge_ff_f(), false);
    EXPECT_EQ(ge_f_ff(), true);
  }
}
template <typename T>
static T normT(T x, T y, T z) {
  return std::sqrt(x * x + y * y + z * z);
}

TYPED_TEST(FltFltExecutorTests, ExampleQuadraticEquation) {
  auto a = make_tensor<fltflt>({});
  auto b = make_tensor<fltflt>({});
  auto c = make_tensor<fltflt>({});
  auto norm = make_tensor<fltflt>({});
  (a = static_cast<fltflt>(std::numbers::pi)).run(this->exec);
  (b = static_cast<fltflt>(std::numbers::e)).run(this->exec);
  (c = static_cast<fltflt>(std::numbers::sqrt2)).run(this->exec);
  (norm = sqrt(a * a + b * b + c * c)).run(this->exec);
  this->exec.sync();

  const double ref_f64 = normT(std::numbers::pi, std::numbers::e, std::numbers::sqrt2);
  const double ref_f32 = normT(std::numbers::pi_v<float>, std::numbers::e_v<float>, std::numbers::sqrt2_v<float>);

  EXPECT_GE(numMatchingMantissaBits(static_cast<double>(norm()), ref_f64), 44);
  EXPECT_LE(numMatchingMantissaBits(ref_f32, ref_f64), 24);
}
