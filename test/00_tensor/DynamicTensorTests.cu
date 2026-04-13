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
#include "test_types.h"
#include "gtest/gtest.h"

using namespace matx;

// All dynamic-tensor tests require JIT.
#ifdef MATX_EN_JIT

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class DynamicTensorTest : public ::testing::Test {
protected:
  CUDAJITExecutor exec{};

  void sync() { cudaDeviceSynchronize(); }
};

// ---------------------------------------------------------------------------
// Construction & metadata
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ConstructRank1) {
  auto dt = make_tensor<float>();
  make_tensor(dt, {16});
  ASSERT_EQ(dt.DynRank(), 1);
  ASSERT_EQ(dt.Size(0), 16);
  ASSERT_EQ(dt.TotalSize(), 16);
  ASSERT_NE(dt.Data(), nullptr);
}

TEST_F(DynamicTensorTest, ConstructRank2) {
  auto dt = make_tensor<float>();
  make_tensor(dt, {32, 64});
  ASSERT_EQ(dt.DynRank(), 2);
  ASSERT_EQ(dt.Size(0), 32);
  ASSERT_EQ(dt.Size(1), 64);
  ASSERT_EQ(dt.TotalSize(), 32 * 64);
}

TEST_F(DynamicTensorTest, ConstructRank3) {
  auto dt = make_tensor<double>();
  make_tensor(dt, {4, 8, 16});
  ASSERT_EQ(dt.DynRank(), 3);
  ASSERT_EQ(dt.Size(0), 4);
  ASSERT_EQ(dt.Size(1), 8);
  ASSERT_EQ(dt.Size(2), 16);
  ASSERT_EQ(dt.TotalSize(), 4 * 8 * 16);
}

TEST_F(DynamicTensorTest, ConstructRank4) {
  auto dt = make_tensor<float>();
  make_tensor(dt, {2, 3, 4, 5});
  ASSERT_EQ(dt.DynRank(), 4);
  ASSERT_EQ(dt.TotalSize(), 2 * 3 * 4 * 5);
}

TEST_F(DynamicTensorTest, RowMajorStrides) {
  auto dt = make_tensor<float>();
  make_tensor(dt, {6, 8, 4});
  // Row-major: strides should be {8*4, 4, 1}
  ASSERT_EQ(dt.Stride(0), 32);
  ASSERT_EQ(dt.Stride(1), 4);
  ASSERT_EQ(dt.Stride(2), 1);
}

TEST_F(DynamicTensorTest, WrapExistingPointer) {
  float *ptr;
  cudaMallocManaged(&ptr, 100 * sizeof(float));
  ptr[0] = 42.0f;

  auto dt = make_tensor<float>();
  make_tensor(dt, ptr, {10, 10});
  ASSERT_EQ(dt.DynRank(), 2);
  ASSERT_EQ(dt.Data(), ptr);
  ASSERT_EQ(dt.Data()[0], 42.0f);

  cudaFree(ptr);
}

// ---------------------------------------------------------------------------
// JIT string generation (should match static tensor_t output format)
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, JITClassNameFormat) {
  auto dt = make_tensor<float>();
  make_tensor(dt, {128, 64});
  std::string name = dt.get_jit_class_name();

  // Should start with JITTensorImpl_float_
  ASSERT_TRUE(name.find("JITTensorImpl_float_") == 0)
      << "JIT class name has unexpected prefix: " << name;

  // Should contain R2 (rank 2)
  ASSERT_TRUE(name.find("R2") != std::string::npos)
      << "JIT class name missing rank: " << name;

  // Should contain the sizes
  ASSERT_TRUE(name.find("SI_128_64") != std::string::npos)
      << "JIT class name missing sizes: " << name;
}

TEST_F(DynamicTensorTest, JITClassNameMatchesStaticTensor) {
  // A dynamic tensor and a static tensor with the same shape should
  // produce identical JIT class names.
  auto dyn = make_tensor<float>();
  make_tensor(dyn, {32, 16});
  auto stat = make_tensor<float>({32, 16});

  std::string dyn_name = dyn.get_jit_class_name();
  std::string stat_name = stat.get_jit_class_name();

  ASSERT_EQ(dyn_name, stat_name)
      << "Dynamic: " << dyn_name << "\nStatic:  " << stat_name;
}

// ---------------------------------------------------------------------------
// Type traits
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, TypeTraits) {
  using dt_t = detail::dynamic_tensor_t<float>;
  ASSERT_TRUE(is_dynamic_tensor_v<dt_t>);
  ASSERT_TRUE(is_dynamic_rank_op_v<dt_t>);
  ASSERT_TRUE((is_matx_op_c<dt_t>));

  // Static tensor should NOT be dynamic
  using st_t = tensor_t<float, 2>;
  ASSERT_FALSE(is_dynamic_tensor_v<st_t>);
  ASSERT_FALSE(is_dynamic_rank_op_v<st_t>);
}

// ---------------------------------------------------------------------------
// Expression tree construction
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, BinaryExpressionCompiles) {
  auto a = make_tensor<float>();
  make_tensor(a, {10, 20});
  auto b = make_tensor<float>();
  make_tensor(b, {10, 20});
  auto expr = a + b;
  // Rank() returns MATX_MAX_DYNAMIC_RANK (large positive) so all operator
  // code compiles.  DynRank() returns the actual runtime rank.
  ASSERT_EQ(expr.Rank(), MATX_MAX_DYNAMIC_RANK);
  ASSERT_EQ(expr.DynRank(), 2);
  // Size() still works because it queries children directly
  ASSERT_EQ(expr.Size(0), 10);
  ASSERT_EQ(expr.Size(1), 20);
}

TEST_F(DynamicTensorTest, SetExpressionCompiles) {
  auto a = make_tensor<float>();
  make_tensor(a, {10, 20});
  auto b = make_tensor<float>();
  make_tensor(b, {10, 20});
  auto expr = (a = b);
  ASSERT_EQ(expr.Rank(), MATX_MAX_DYNAMIC_RANK);
  ASSERT_EQ(expr.DynRank(), 2); // set delegates to output tensor
}

TEST_F(DynamicTensorTest, ExecMixedStaticDynamicWithJIT) {
  const index_t R = 16, C = 32;

  auto dyn = make_tensor<float>();
  make_tensor(dyn, {R, C});
  auto stat = make_tensor<float>({R, C});
  auto out_static = make_tensor<float>({R, C});

  for (index_t i = 0; i < R * C; i++) {
    dyn.Data()[i] = static_cast<float>(i);
    stat.Data()[i] = 3.0f;
  }

  (out_static = dyn + stat).run(exec);
  sync();

  for (index_t i = 0; i < R * C; i++) {
    ASSERT_FLOAT_EQ(out_static.Data()[i], static_cast<float>(i) + 3.0f)
        << "Mismatch at linear index " << i;
  }
}

TEST_F(DynamicTensorTest, MixedStaticDynamicRejectedByCudaExecutor) {
  const index_t R = 16, C = 32;

  auto dyn = make_tensor<float>();
  make_tensor(dyn, {R, C});
  auto stat = make_tensor<float>({R, C});
  auto out_static = make_tensor<float>({R, C});

  auto expr = (out_static = dyn + stat);
  cudaExecutor non_jit{};
  EXPECT_THROW({ non_jit.Exec(expr); }, matx::detail::matxException);
}

TEST_F(DynamicTensorTest, RCollapseRejectsDynamicRankLessThanDim) {
  auto dyn = make_tensor<float>();
  make_tensor(dyn, {16}); // rank 1

  EXPECT_THROW({ [[maybe_unused]] auto op = rcollapse<2>(dyn); },
               matx::detail::matxException);
}

TEST_F(DynamicTensorTest, LCollapseRejectsDynamicRankLessThanDim) {
  auto dyn = make_tensor<float>();
  make_tensor(dyn, {16}); // rank 1

  EXPECT_THROW({ [[maybe_unused]] auto op = lcollapse<2>(dyn); },
               matx::detail::matxException);
}

// ---------------------------------------------------------------------------
// JIT execution: rank 1
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecRank1_Add) {
  constexpr int N = 256;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});
  auto c = make_tensor<float>();
  make_tensor(c, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i);
    b.Data()[i] = 1.0f;
  }

  (c = a + b).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], static_cast<float>(i) + 1.0f)
        << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// JIT execution: rank 2
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecRank2_ScalarMul) {
  const index_t R = 32, C = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto out = make_tensor<float>();
  make_tensor(out, {R, C});

  for (index_t i = 0; i < R * C; i++) {
    a.Data()[i] = 2.0f;
  }

  (out = a * 3.0f).run(exec);
  sync();

  for (index_t i = 0; i < R * C; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 6.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// JIT execution: rank 3
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecRank3_Compound) {
  auto a = make_tensor<float>();
  make_tensor(a, {4, 8, 16});
  auto b = make_tensor<float>();
  make_tensor(b, {4, 8, 16});
  auto c = make_tensor<float>();
  make_tensor(c, {4, 8, 16});

  index_t total = a.TotalSize();
  for (index_t i = 0; i < total; i++) {
    a.Data()[i] = 1.0f;
    b.Data()[i] = 2.0f;
  }

  // c = a + b * 3  =>  1 + 2*3 = 7
  (c = a + b * 3.0f).run(exec);
  sync();

  for (index_t i = 0; i < total; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], 7.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// JIT execution: rank 4
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecRank4_Sub) {
  auto a = make_tensor<float>();
  make_tensor(a, {2, 4, 4, 8});
  auto b = make_tensor<float>();
  make_tensor(b, {2, 4, 4, 8});
  auto c = make_tensor<float>();
  make_tensor(c, {2, 4, 4, 8});

  index_t total = a.TotalSize();
  for (index_t i = 0; i < total; i++) {
    a.Data()[i] = 10.0f;
    b.Data()[i] = 3.0f;
  }

  (c = a - b).run(exec);
  sync();

  for (index_t i = 0; i < total; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], 7.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// JIT execution: double type
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecDouble) {
  constexpr int N = 128;
  auto a = make_tensor<double>();
  make_tensor(a, {N});
  auto b = make_tensor<double>();
  make_tensor(b, {N});
  auto c = make_tensor<double>();
  make_tensor(c, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 1.5;
    b.Data()[i] = 2.5;
  }

  (c = a + b).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_DOUBLE_EQ(c.Data()[i], 4.0) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Runtime rank selection
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, RuntimeRankSelection) {
  // Simulate choosing rank at runtime (e.g. from a config file)
  for (int rank = 1; rank <= 4; rank++) {
    std::vector<index_t> shape(rank, 8); // all dims = 8
    auto a = make_tensor<float>();
    make_tensor(a, shape);
    auto b = make_tensor<float>();
    make_tensor(b, shape);
    auto c = make_tensor<float>();
    make_tensor(c, shape);

    index_t total = a.TotalSize();
    for (index_t i = 0; i < total; i++) {
      a.Data()[i] = 5.0f;
      b.Data()[i] = 3.0f;
    }

    (c = a - b).run(exec);
    sync();

    for (index_t i = 0; i < total; i++) {
      ASSERT_FLOAT_EQ(c.Data()[i], 2.0f)
          << "Rank " << rank << " mismatch at flat index " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// In-place accumulation (a = a + b)
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecAccumulate) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 10.0f;
    b.Data()[i] = 5.0f;
  }

  (a = a + b).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(a.Data()[i], 15.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Division
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecDivision) {
  constexpr int N = 128;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});
  auto c = make_tensor<float>();
  make_tensor(c, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 10.0f;
    b.Data()[i] = 2.0f;
  }

  (c = a / b).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], 5.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Scalar division
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecScalarDiv) {
  const index_t R = 16, C = 32;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto out = make_tensor<float>();
  make_tensor(out, {R, C});

  for (index_t i = 0; i < R * C; i++) {
    a.Data()[i] = 12.0f;
  }

  (out = a / 4.0f).run(exec);
  sync();

  for (index_t i = 0; i < R * C; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 3.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// Unary negation
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecNegate) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 7.0f;
  }

  (out = -a).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], -7.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Unary sqrt
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecSqrt) {
  constexpr int N = 128;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 16.0f;
  }

  (out = sqrt(a)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 4.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Unary abs
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecAbs) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = (i % 2 == 0) ? -3.0f : 3.0f;
  }

  (out = abs(a)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 3.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Unary exp and log round-trip
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecExpLog) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto tmp = make_tensor<float>();
  make_tensor(tmp, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 2.0f;
  }

  // out = log(exp(a)) should give back a
  (tmp = exp(a)).run(exec);
  sync();
  (out = log(tmp)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_NEAR(out.Data()[i], 2.0f, 1e-5f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Unary sin/cos identity: sin^2 + cos^2 = 1
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecSinCosIdentity) {
  constexpr int N = 128;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto s = make_tensor<float>();
  make_tensor(s, {N});
  auto c = make_tensor<float>();
  make_tensor(c, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i) * 0.1f;
  }

  (s = sin(a)).run(exec);
  sync();
  (c = cos(a)).run(exec);
  sync();
  (out = s * s + c * c).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_NEAR(out.Data()[i], 1.0f, 1e-5f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Floor and ceil
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecFloorCeil) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto fl = make_tensor<float>();
  make_tensor(fl, {N});
  auto ce = make_tensor<float>();
  make_tensor(ce, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 2.7f;
  }

  (fl = floor(a)).run(exec);
  sync();
  (ce = ceil(a)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(fl.Data()[i], 2.0f) << "Floor mismatch at index " << i;
    ASSERT_FLOAT_EQ(ce.Data()[i], 3.0f) << "Ceil mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Binary pow
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecPow) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 3.0f;
    b.Data()[i] = 2.0f;
  }

  (out = pow(a, b)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 9.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Binary max and min
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecMaxMin) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});
  auto mx = make_tensor<float>();
  make_tensor(mx, {N});
  auto mn = make_tensor<float>();
  make_tensor(mn, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i);
    b.Data()[i] = 32.0f;
  }

  (mx = max(a, b)).run(exec);
  sync();
  (mn = min(a, b)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    float expected_max = (static_cast<float>(i) > 32.0f) ? static_cast<float>(i) : 32.0f;
    float expected_min = (static_cast<float>(i) < 32.0f) ? static_cast<float>(i) : 32.0f;
    ASSERT_FLOAT_EQ(mx.Data()[i], expected_max) << "Max mismatch at index " << i;
    ASSERT_FLOAT_EQ(mn.Data()[i], expected_min) << "Min mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Deeply chained expression
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecDeepChain) {
  constexpr int N = 128;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 4.0f;
    b.Data()[i] = 1.0f;
  }

  // out = sqrt(a) + (b * 3 - 1) / 2  =>  2 + (3 - 1)/2 = 2 + 1 = 3
  (out = sqrt(a) + (b * 3.0f - 1.0f) / 2.0f).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 3.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Rank-2 expression with multiple unary/binary ops
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecRank2_CompoundUnaryBinary) {
  const index_t R = 16, C = 32;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto b = make_tensor<float>();
  make_tensor(b, {R, C});
  auto out = make_tensor<float>();
  make_tensor(out, {R, C});

  for (index_t i = 0; i < R * C; i++) {
    a.Data()[i] = 9.0f;
    b.Data()[i] = -5.0f;
  }

  // out = sqrt(a) + abs(b)  =>  3 + 5 = 8
  (out = sqrt(a) + abs(b)).run(exec);
  sync();

  for (index_t i = 0; i < R * C; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 8.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// Integer type: int32_t arithmetic
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecInt32) {
  constexpr int N = 128;
  auto a = make_tensor<int32_t>();
  make_tensor(a, {N});
  auto b = make_tensor<int32_t>();
  make_tensor(b, {N});
  auto c = make_tensor<int32_t>();
  make_tensor(c, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = i;
    b.Data()[i] = 10;
  }

  (c = a + b).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c.Data()[i], i + 10) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Integer modulo
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecIntMod) {
  constexpr int N = 64;
  auto a = make_tensor<int32_t>();
  make_tensor(a, {N});
  auto b = make_tensor<int32_t>();
  make_tensor(b, {N});
  auto c = make_tensor<int32_t>();
  make_tensor(c, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = i;
    b.Data()[i] = 7;
  }

  (c = a % b).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c.Data()[i], i % 7) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Multiple successive assignments to the same output tensor
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecMultipleAssignments) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});
  auto c = make_tensor<float>();
  make_tensor(c, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 1.0f;
    b.Data()[i] = 2.0f;
  }

  // First: c = a + b  => 3
  (c = a + b).run(exec);
  sync();
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], 3.0f);
  }

  // Second: c = a * b  => 2
  (c = a * b).run(exec);
  sync();
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], 2.0f);
  }

  // Third: c = a - b  => -1
  (c = a - b).run(exec);
  sync();
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], -1.0f);
  }
}

// ---------------------------------------------------------------------------
// Large rank-2 tensor (stress test dimensions)
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecLargeRank2) {
  const index_t R = 256, C = 512;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto b = make_tensor<float>();
  make_tensor(b, {R, C});
  auto c = make_tensor<float>();
  make_tensor(c, {R, C});

  index_t total = R * C;
  for (index_t i = 0; i < total; i++) {
    a.Data()[i] = 1.0f;
    b.Data()[i] = 2.0f;
  }

  (c = a + b).run(exec);
  sync();

  for (index_t i = 0; i < total; i++) {
    ASSERT_FLOAT_EQ(c.Data()[i], 3.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// Three-operand expression: a + b + c
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecThreeOperandAdd) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto b = make_tensor<float>();
  make_tensor(b, {N});
  auto c = make_tensor<float>();
  make_tensor(c, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = 1.0f;
    b.Data()[i] = 2.0f;
    c.Data()[i] = 3.0f;
  }

  (out = a + b + c).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 6.0f) << "Mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Rank 3: unary op on multi-dimensional tensor
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecRank3_Sqrt) {
  auto a = make_tensor<float>();
  make_tensor(a, {4, 8, 16});
  auto out = make_tensor<float>();
  make_tensor(out, {4, 8, 16});

  index_t total = a.TotalSize();
  for (index_t i = 0; i < total; i++) {
    a.Data()[i] = 25.0f;
  }

  (out = sqrt(a)).run(exec);
  sync();

  for (index_t i = 0; i < total; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 5.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// Rank 4: negation + addition
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecRank4_NegateAdd) {
  auto a = make_tensor<float>();
  make_tensor(a, {2, 3, 4, 5});
  auto b = make_tensor<float>();
  make_tensor(b, {2, 3, 4, 5});
  auto out = make_tensor<float>();
  make_tensor(out, {2, 3, 4, 5});

  index_t total = a.TotalSize();
  for (index_t i = 0; i < total; i++) {
    a.Data()[i] = 3.0f;
    b.Data()[i] = 10.0f;
  }

  // out = -a + b  =>  -3 + 10 = 7
  (out = -a + b).run(exec);
  sync();

  for (index_t i = 0; i < total; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 7.0f) << "Mismatch at flat index " << i;
  }
}

// ---------------------------------------------------------------------------
// fftshift1D: even-length 1D signal
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecFFTShift1D) {
  // For an even-length array, fftshift swaps the two halves.
  // Input:  [0, 1, 2, 3, 4, 5, 6, 7]
  // Output: [4, 5, 6, 7, 0, 1, 2, 3]
  constexpr int N = 8;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i);
  }

  (out = fftshift1D(a)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    float expected = static_cast<float>((i + N / 2) % N);
    ASSERT_FLOAT_EQ(out.Data()[i], expected)
        << "fftshift1D mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// ifftshift1D: round-trip with fftshift
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecIFFTShift1D_Roundtrip) {
  constexpr int N = 8;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto shifted = make_tensor<float>();
  make_tensor(shifted, {N});
  auto roundtrip = make_tensor<float>();
  make_tensor(roundtrip, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i);
  }

  // fftshift then ifftshift should give back the original
  (shifted = fftshift1D(a)).run(exec);
  sync();
  (roundtrip = ifftshift1D(shifted)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(roundtrip.Data()[i], static_cast<float>(i))
        << "ifftshift1D roundtrip mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// fftshift1D on rank-2 tensor (shifts last dimension)
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecFFTShift1D_Rank2) {
  const index_t R = 4, C = 8;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto out = make_tensor<float>();
  make_tensor(out, {R, C});

  for (index_t r = 0; r < R; r++) {
    for (index_t c = 0; c < C; c++) {
      a.Data()[r * C + c] = static_cast<float>(c);
    }
  }

  (out = fftshift1D(a)).run(exec);
  sync();

  for (index_t r = 0; r < R; r++) {
    for (index_t c = 0; c < C; c++) {
      float expected = static_cast<float>((c + C / 2) % C);
      ASSERT_FLOAT_EQ(out.Data()[r * C + c], expected)
          << "fftshift1D rank-2 mismatch at [" << r << "," << c << "]";
    }
  }
}

// ---------------------------------------------------------------------------
// reverse<0> on rank-1 tensor
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecReverse1D) {
  constexpr int N = 16;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i);
  }

  (out = reverse<0>(a)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], static_cast<float>(N - 1 - i))
        << "reverse<0> mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// reverse<1> on rank-2 tensor (reverse columns)
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecReverse_Rank2_Dim1) {
  const index_t R = 4, C = 8;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto out = make_tensor<float>();
  make_tensor(out, {R, C});

  for (index_t r = 0; r < R; r++) {
    for (index_t c = 0; c < C; c++) {
      a.Data()[r * C + c] = static_cast<float>(c);
    }
  }

  (out = reverse<1>(a)).run(exec);
  sync();

  for (index_t r = 0; r < R; r++) {
    for (index_t c = 0; c < C; c++) {
      float expected = static_cast<float>(C - 1 - c);
      ASSERT_FLOAT_EQ(out.Data()[r * C + c], expected)
          << "reverse<1> mismatch at [" << r << "," << c << "]";
    }
  }
}

// ---------------------------------------------------------------------------
// shift<0> on rank-1 tensor
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecShift1D) {
  constexpr int N = 8;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  // Input: [0, 1, 2, 3, 4, 5, 6, 7]
  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i);
  }

  // shift<0>(a, 3) negates the shift internally: out[i] = a[(-3 + i) mod N]
  // out[0]=a[5], out[1]=a[6], out[2]=a[7], out[3]=a[0], out[4]=a[1], ...
  (out = shift<0>(a, 3)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    int src = ((i - 3) % N + N) % N;
    float expected = static_cast<float>(src);
    ASSERT_FLOAT_EQ(out.Data()[i], expected)
        << "shift<0> mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// as_double (cast) from float to double
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecCastFloatToDouble) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<double>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i) + 0.5f;
  }

  (out = as_double(a)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_DOUBLE_EQ(out.Data()[i], static_cast<double>(static_cast<float>(i) + 0.5f))
        << "as_double mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// as_int32 cast from float, truncating
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecCastFloatToInt) {
  constexpr int N = 64;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<int32_t>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i) + 0.7f;
  }

  (out = as_int32(a)).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_EQ(out.Data()[i], i)
        << "as_int32 mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Chained: reverse + fftshift on rank-1
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecChainedReverseFftshift) {
  constexpr int N = 8;
  auto a = make_tensor<float>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = static_cast<float>(i);
  }

  // reverse then fftshift
  (out = fftshift1D(reverse<0>(a))).run(exec);
  sync();

  // reverse<0>: [7, 6, 5, 4, 3, 2, 1, 0]
  // fftshift1D: swap halves -> [3, 2, 1, 0, 7, 6, 5, 4]
  float expected[] = {3, 2, 1, 0, 7, 6, 5, 4};
  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], expected[i])
        << "Chained reverse+fftshift mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// Cast + arithmetic: as_float(int_tensor) * 2.5
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecCastAndArithmetic) {
  constexpr int N = 64;
  auto a = make_tensor<int32_t>();
  make_tensor(a, {N});
  auto out = make_tensor<float>();
  make_tensor(out, {N});

  for (int i = 0; i < N; i++) {
    a.Data()[i] = i;
  }

  (out = as_float(a) * 2.5f).run(exec);
  sync();

  for (int i = 0; i < N; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], static_cast<float>(i) * 2.5f)
        << "cast+arithmetic mismatch at index " << i;
  }
}

// ---------------------------------------------------------------------------
// fftshift2D on rank-2 tensor
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecFFTShift2D) {
  const index_t R = 4, C = 4;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto out = make_tensor<float>();
  make_tensor(out, {R, C});

  // Fill with sequential values
  for (index_t i = 0; i < R * C; i++) {
    a.Data()[i] = static_cast<float>(i);
  }

  (out = fftshift2D(a)).run(exec);
  sync();

  // fftshift2D swaps quadrants: each dim shifted by (Size+1)/2
  for (index_t r = 0; r < R; r++) {
    for (index_t c = 0; c < C; c++) {
      index_t sr = (r + (R + 1) / 2) % R;
      index_t sc = (c + (C + 1) / 2) % C;
      float expected = static_cast<float>(sr * C + sc);
      ASSERT_FLOAT_EQ(out.Data()[r * C + c], expected)
          << "fftshift2D mismatch at [" << r << "," << c << "]";
    }
  }
}

// ---------------------------------------------------------------------------
// Reverse + arithmetic on rank-2
// ---------------------------------------------------------------------------

TEST_F(DynamicTensorTest, ExecReverseArithmeticRank2) {
  const index_t R = 8, C = 16;
  auto a = make_tensor<float>();
  make_tensor(a, {R, C});
  auto b = make_tensor<float>();
  make_tensor(b, {R, C});
  auto out = make_tensor<float>();
  make_tensor(out, {R, C});

  for (index_t i = 0; i < R * C; i++) {
    a.Data()[i] = 1.0f;
    b.Data()[i] = 2.0f;
  }

  // out = reverse<0>(a) + b  (since a is all 1s, reverse is still 1s)
  (out = reverse<0>(a) + b).run(exec);
  sync();

  for (index_t i = 0; i < R * C; i++) {
    ASSERT_FLOAT_EQ(out.Data()[i], 3.0f) << "Mismatch at flat index " << i;
  }
}

#endif // MATX_EN_JIT
