#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Pad)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  {
    // example-begin pad-test-1
    const int N = 5;
    const int before = 2;
    const int after = 3;

    auto t1 = make_tensor<TestType>({N});
    auto t1_padded = make_tensor<TestType>({N+before+after});

    for (int i = 0; i < N; i++) {
      t1(i) = TestType(i+1);
    }

    // Pad tensor t1 along dimension 0 using value 42. This will included two padding
    // elements before the original data and three padding elements after the original data.
    (t1_padded = pad(t1, 0, {before, after}, TestType{42})).run(exec);

    exec.sync();
    // t1_padded is {42, 42, 1, 2, 3, 4, 5, 42, 42, 42}
    // example-end pad-test-1

    // Check padding values
    for (int i = 0; i < before; i++) {
      ASSERT_EQ(t1_padded(i), TestType{static_cast<TestType>(42)}); // before padding
    }
    for (int i = before; i < N+before; i++) {
      ASSERT_EQ(t1_padded(i), TestType{static_cast<TestType>(i-before+1)}); // original data
    }
    for (int i = N+before; i < N+before+after; i++) {
      ASSERT_EQ(t1_padded(i), TestType{static_cast<TestType>(42)}); // after padding
    }
  }

  {
    // example-begin pad-test-2
    const int N = 5;
    const int before = 2;
    const int after = 3;

    auto t1 = make_tensor<TestType>({N});
    auto t1_padded = make_tensor<TestType>({N+before+after});

    for (int i = 0; i < N; i++) {
      t1(i) = TestType(i+1);
    }

    // Pad tensor t1 along dimension 0 using edge padding. This will included two padding
    // elements before the original data and three padding elements after the original data.
    (t1_padded = pad(t1, 0, {before, after}, TestType{42}, matx::MATX_PAD_MODE_EDGE)).run(exec);

    exec.sync();
    // t1_padded is {1, 1, 1, 2, 3, 4, 5, 5, 5, 5}
    // example-end pad-test-2

    // Check padding values
    for (int i = 0; i < before; i++) {
      ASSERT_EQ(t1_padded(i), TestType{static_cast<TestType>(1)}); // before padding
    }
    for (int i = before; i < N+before; i++) {
      ASSERT_EQ(t1_padded(i), TestType{static_cast<TestType>(i-before+1)}); // original data
    }
    for (int i = N+before; i < N+before+after; i++) {
      ASSERT_EQ(t1_padded(i), TestType{static_cast<TestType>(N)}); // after padding
    }
  }

  {
    // Test 2D constant padding along dimension 1
    const int M = 3;
    const int N = 4;
    const int before = 1;
    const int after = 2;
    auto t2d = make_tensor<TestType>({M, N});
    auto t2d_padded = make_tensor<TestType>({M, N+before+after});

    // Fill with values [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    for (int i = 0; i <M; i++) {
      for (int j = 0; j < N; j++) {
        t2d(i, j) = TestType(i * N + j);
      }
    }

    // Pad along dimension 1 with {before, after} padding (before, after), using value 42
    (t2d_padded = pad(t2d, 1, {before, after}, TestType{42})).run(exec);
    exec.sync();

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < before; j++) {
        ASSERT_EQ(t2d_padded(i, j), TestType{static_cast<TestType>(42)}); // before padding
      }
      for (int j = before; j < N+before; j++) {
        ASSERT_EQ(t2d_padded(i, j), TestType{static_cast<TestType>(i*N+j-before)}); // original data
      }
      for (int j = N+before; j < N+before+after; j++) {
        ASSERT_EQ(t2d_padded(i, j), TestType{static_cast<TestType>(42)}); // after padding
      }
    }
  }

  {
    // Test 2D edge padding along dimension 1
    const int M = 3;
    const int N = 4;
    const int before = 1;
    const int after = 2;
    auto t2d = make_tensor<TestType>({M, N});
    auto t2d_padded = make_tensor<TestType>({M, N+before+after});

    // Fill with values [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    for (int i = 0; i <M; i++) {
      for (int j = 0; j < N; j++) {
        t2d(i, j) = TestType(i * N + j);
      }
    }

    // Pad along dimension 1 with {before, after} padding (before, after), using edge padding
    (t2d_padded = pad(t2d, 1, {before, after}, TestType{0}, matx::MATX_PAD_MODE_EDGE)).run(exec);
    exec.sync();

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < before; j++) {
        ASSERT_EQ(t2d_padded(i, j), TestType{static_cast<TestType>(i*N+0)}); // before padding
      }
      for (int j = before; j < N+before; j++) {
        ASSERT_EQ(t2d_padded(i, j), TestType{static_cast<TestType>(i*N+j-before)}); // original data
      }
      for (int j = N+before; j < N+before+after; j++) {
        ASSERT_EQ(t2d_padded(i, j), TestType{static_cast<TestType>(i*N+N-1)}); // after padding
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, PadAPIVariations)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  const int N = 3;
  auto t1d = make_tensor<TestType>({N});
  for (int i = 0; i < N; i++) {
    t1d(i) = TestType(i+1);
  }

  // tuple API with {,} syntax
  auto t1_tuple = make_tensor<TestType>({N+2+2}); 
  (t1_tuple = pad(t1d, 0, {2, 2}, TestType{0})).run(exec);
  exec.sync();

  for (int i = 0; i < 2; i++) {
    ASSERT_EQ(t1_tuple(i), TestType{static_cast<TestType>(0)}); // before padding
  }
  for (int i = 2; i < N+2; i++) {
    ASSERT_EQ(t1_tuple(i), TestType{static_cast<TestType>(i-2+1)}); // original data
  }
  for (int i = N+2; i < N+2+2; i++) {
    ASSERT_EQ(t1_tuple(i), TestType{static_cast<TestType>(0)}); // after padding
  }

  // tuple API with std::array
  auto t1_array = make_tensor<TestType>({N+1+2}); 
  std::array<index_t, 2> pad_sizes = {1, 2};
  (t1_array = pad(t1d, 0, pad_sizes, TestType{0})).run(exec);
  exec.sync();

  for (int i = 0; i < 1; i++) {
    ASSERT_EQ(t1_array(i), TestType{static_cast<TestType>(0)}); // before padding
  }
  for (int i = 1; i < N+1; i++) {
    ASSERT_EQ(t1_array(i), TestType{static_cast<TestType>(i-1+1)}); // original data
  }
  for (int i = N+1; i < N+1+2; i++) {
    ASSERT_EQ(t1_array(i), TestType{static_cast<TestType>(0)}); // after padding
  }

  // Asymmetric padding
  auto t1_asym = make_tensor<TestType>({N+3+2}); 
  (t1_asym = pad(t1d, 0, {3, 2}, TestType{0})).run(exec);
  exec.sync();

  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(t1_asym(i), TestType{static_cast<TestType>(0)}); // before padding
  }
  for (int i = 3; i < N+3; i++) {
    ASSERT_EQ(t1_asym(i), TestType{static_cast<TestType>(i-3+1)}); // original data
  }
  for (int i = N+3; i < N+3+2; i++) {
    ASSERT_EQ(t1_asym(i), TestType{static_cast<TestType>(0)}); // after padding
  }

  // Padding using C-style array
  auto t1_c_array = make_tensor<TestType>({N+1+1});
  const index_t c_pad_sizes[] = {1, 1};
  (t1_c_array = pad(t1d, 0, c_pad_sizes, TestType{0})).run(exec);
  exec.sync();

  for (int i = 0; i < 1; i++) {
    ASSERT_EQ(t1_c_array(i), TestType{static_cast<TestType>(0)}); // before padding
  }
  for (int i = 1; i < N+1; i++) {
    ASSERT_EQ(t1_c_array(i), TestType{static_cast<TestType>(i-1+1)}); // original data
  }
  for (int i = N+1; i < N+1+1; i++) {
    ASSERT_EQ(t1_c_array(i), TestType{static_cast<TestType>(0)}); // after padding
  }

  // Zero-length padding
  auto t1_zero = make_tensor<TestType>({N});
  (t1_zero = pad(t1d, 0, {0, 0}, TestType{0})).run(exec);
  exec.sync();
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(t1_zero(i), TestType{static_cast<TestType>(i+1)}); // original data
  }

  MATX_EXIT_HANDLER();
}

// Pads a rank-3 tensor along each axis in both constant and edge modes
TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, PadAllAxes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  const index_t in_shape[3] = {2, 3, 4};
  const index_t before = 1;
  const index_t after = 2;
  const TestType pad_value = TestType(99);

  auto in = make_tensor<TestType>({in_shape[0], in_shape[1], in_shape[2]});
  for (index_t i = 0; i < in_shape[0]; i++)
    for (index_t j = 0; j < in_shape[1]; j++)
      for (index_t k = 0; k < in_shape[2]; k++)
        in(i, j, k) = TestType(static_cast<float>(i * 16 + j * 4 + k));

  for (int axis = 0; axis < 3; axis++) {
    for (PadMode mode : {MATX_PAD_MODE_CONSTANT, MATX_PAD_MODE_EDGE}) {
      cuda::std::array<index_t, 3> shape{in_shape[0], in_shape[1], in_shape[2]};
      shape[axis] += before + after;
      auto out = make_tensor<TestType>(shape);

      (out = pad(in, axis, {before, after}, pad_value, mode)).run(exec);
      exec.sync();

      for (index_t i = 0; i < shape[0]; i++) {
        for (index_t j = 0; j < shape[1]; j++) {
          for (index_t k = 0; k < shape[2]; k++) {
            index_t idx[3] = {i, j, k};
            const index_t p = idx[axis] - before;
            TestType expected;
            if (p >= 0 && p < in_shape[axis]) {
              idx[axis] = p;
              expected = in(idx[0], idx[1], idx[2]);
            } else if (mode == MATX_PAD_MODE_CONSTANT) {
              expected = pad_value;
            } else {
              idx[axis] = (p < 0) ? 0 : in_shape[axis] - 1;
              expected = in(idx[0], idx[1], idx[2]);
            }
            ASSERT_EQ(out(i, j, k), expected) << "axis=" << axis << " mode=" << static_cast<int>(mode);
          }
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// pad rejects an axis outside [0, Rank()) in every build mode
TEST(OperatorValidationTests, PadInvalidAxis)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto t = make_tensor<TestType>({2, 3, 4});
  const TestType v = TestType(0);

  EXPECT_THROW(pad(t, 3, {1, 2}, v), matx::detail::matxException);
  EXPECT_THROW(pad(t, -1, {1, 2}, v), matx::detail::matxException);
  EXPECT_NO_THROW(pad(t, 0, {1, 2}, v));
  EXPECT_NO_THROW(pad(t, 2, {1, 2}, v));

  // Negative pad sizes are rejected in every build mode
  EXPECT_THROW(pad(t, 0, {-1, 2}, v), matx::detail::matxException);
  EXPECT_THROW(pad(t, 0, {1, -2}, v), matx::detail::matxException);
  EXPECT_NO_THROW(pad(t, 0, {0, 0}, v));

  MATX_EXIT_HANDLER();
}
