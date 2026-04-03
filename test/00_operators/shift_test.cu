#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, ShiftOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  index_t count0 = 100;
  index_t count1 = 201;
  tensor_t<TestType, 2> t2({count0, count1});
  tensor_t<TestType, 2> t2s({count0, count1});
  tensor_t<TestType, 2> t2s2({count0, count1});
  auto t0 = make_tensor<int>({});
  t0() = -5;

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TestType>>(i * count1 + j);
    }
  }

  {
    // example-begin shift-test-1
    // Shift the first dimension of "t2" by -5 so the 5th element of "t2" is the first element of "t2s"
    (t2s = shift<0>(t2, -5)).run(exec);
    // example-end shift-test-1
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2((i + 5) % count0, j)));
      }
    }
  }
  
  {
    (t2s = shift<0>(t2, t0)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2((i + 5) % count0, j)));
      }
    }
  }

  {
    (t2s = shift<1>(t2, -5)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, (j + 5) % count1)));
      }
    }
  }

  {
    (t2s = shift<1,0>(t2, -5, -6)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j), t2((i + 6) % count0, (j + 5) % count1)));
      }
    }
  }

  {
    // example-begin fftshift2D-test-1
    (t2s = fftshift2D(t2)).run(exec);
    // example-end fftshift2D-test-1
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j), t2((i + (count0 + 1) / 2) % count0,
                          (j + (count1 + 1) / 2) % count1)));
      }
    }
  }

  {
    // example-begin ifftshift2D-test-1
    (t2s = ifftshift2D(t2)).run(exec);
    // example-end ifftshift2D-test-1
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j),
            t2((i + (count0) / 2) % count0, (j + (count1) / 2) % count1)));
      }
    }
  }

  // Right shifts
  {
    (t2s = shift<0>(t2, 5)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        index_t idim = i < 5 ? (t2.Size(0) - 5 + i) : (i - 5);
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(idim, j)));
      }
    }
  }

  {
    (t2s = shift<1>(t2, 5)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        index_t jdim = j < 5 ? (t2.Size(1) - 5 + j) : (j - 5);
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, jdim)));
      }
    }
  }

  // Large shifts
  {
    (t2s = shift<0>(t2, -t2.Size(0) * 4)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, j)));
      }
    }
  }

  {
    // Shift 4 times the size back, minus one. This should be equivalent to
    // simply shifting by -1
    (t2s = shift<0>(t2, -t2.Size(0) * 4 - 1)).run(exec);
    (t2s2 = shift<0>(t2, -1)).run(exec);
    exec.sync();

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2s2(i, j)));
      }
    }
  }

  // Rank-1 shift operator: shift dim 0 with per-column shifts
  {
    index_t rows = 10;
    index_t cols = 5;
    tensor_t<TestType, 2> t2r1({rows, cols});
    tensor_t<TestType, 2> t2r1s({rows, cols});
    auto shifts_col = make_tensor<int>({cols});

    // Initialize input
    for (index_t i = 0; i < rows; i++) {
      for (index_t j = 0; j < cols; j++) {
        t2r1(i, j) = static_cast<detail::value_promote_t<TestType>>(i * cols + j);
      }
    }

    // Each column gets a different shift amount
    for (index_t j = 0; j < cols; j++) {
      shifts_col(j) = static_cast<int>(j + 1);  // shifts: 1, 2, 3, 4, 5
    }

    (t2r1s = shift<0>(t2r1, shifts_col)).run(exec);
    exec.sync();

    for (index_t i = 0; i < rows; i++) {
      for (index_t j = 0; j < cols; j++) {
        index_t shift_amount = static_cast<index_t>(j + 1);
        index_t src_i = i < shift_amount ? (rows - shift_amount + i) : (i - shift_amount);
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2r1s(i, j), t2r1(src_i, j)));
      }
    }
  }

  // Rank-1 shift operator: shift dim 1 with per-row shifts
  {
    index_t rows = 10;
    index_t cols = 5;
    tensor_t<TestType, 2> t2r1({rows, cols});
    tensor_t<TestType, 2> t2r1s({rows, cols});
    auto shifts_row = make_tensor<int>({rows});

    for (index_t i = 0; i < rows; i++) {
      for (index_t j = 0; j < cols; j++) {
        t2r1(i, j) = static_cast<detail::value_promote_t<TestType>>(i * cols + j);
      }
    }

    // Each row gets a different shift amount
    for (index_t i = 0; i < rows; i++) {
      shifts_row(i) = static_cast<int>(i % 3);  // shifts: 0, 1, 2, 0, 1, 2, ...
    }

    (t2r1s = shift<1>(t2r1, shifts_row)).run(exec);
    exec.sync();

    for (index_t i = 0; i < rows; i++) {
      for (index_t j = 0; j < cols; j++) {
        index_t shift_amount = static_cast<index_t>(i % 3);
        index_t src_j = j < shift_amount ? (cols - shift_amount + j) : (j - shift_amount);
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2r1s(i, j), t2r1(i, src_j)));
      }
    }
  }

  // Rank-1 shift operator: negative shifts
  {
    index_t rows = 10;
    index_t cols = 5;
    tensor_t<TestType, 2> t2r1({rows, cols});
    tensor_t<TestType, 2> t2r1s({rows, cols});
    auto shifts_col = make_tensor<int>({cols});

    for (index_t i = 0; i < rows; i++) {
      for (index_t j = 0; j < cols; j++) {
        t2r1(i, j) = static_cast<detail::value_promote_t<TestType>>(i * cols + j);
      }
    }

    // Negative shifts (shift left)
    for (index_t j = 0; j < cols; j++) {
      shifts_col(j) = -static_cast<int>(j + 1);
    }

    (t2r1s = shift<0>(t2r1, shifts_col)).run(exec);
    exec.sync();

    for (index_t i = 0; i < rows; i++) {
      for (index_t j = 0; j < cols; j++) {
        index_t shift_amount = static_cast<index_t>(j + 1);
        ASSERT_TRUE(
            MatXUtils::MatXTypeCompare(t2r1s(i, j), t2r1((i + shift_amount) % rows, j)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}