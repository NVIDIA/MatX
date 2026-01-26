#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

// No JIT since we use custom types that can't be stringified
TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecsWithoutJIT, ZipVecOp)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test 1D vector types
    using vec1_type = typename matx::detail::VecTypeSelector<TestType, 1>::type;
    auto t = make_tensor<vec1_type>({1});
    (t = zipvec(static_cast<TestType>(2)*ones<TestType>({1}))).run(exec);
    exec.sync();
    ASSERT_EQ(t(0).x, static_cast<TestType>(2));
  }

  if constexpr (std::is_same_v<TestType, float>)
  { // 2D example for documentation
    // example-begin zipvec-test-1
    auto v = linspace<float>(0.25f, 1.0f, 4);
    auto duplicate_rows = clone<2>(v, {4, matxKeepDim});
    auto duplicate_cols = clone<2>(v, {matxKeepDim, 4});
    // Form a 2D grid of coordinates by replicating v along the rows and columns
    // and zipping the two 2D tensors together into a 2D vector tensor. This creates
    // coordinates in the form:
    //   [ (0.25, 0.25) (0.5, 0.25) (0.75, 0.25) (1.0, 0.25) ]
    //   [ (0.25, 0.5)  (0.5, 0.5)  (0.75, 0.5)  (1.0, 0.5)  ]
    //   [ (0.25, 0.75) (0.5, 0.75) (0.75, 0.75) (1.0, 0.75) ]
    //   [ (0.25, 1.0)  (0.5, 1.0)  (0.75, 1.0)  (1.0, 1.0)  ]
    auto coords = make_tensor<float2>({4, 4});
    (coords = zipvec(duplicate_rows, duplicate_cols)).run(exec);
    exec.sync();

    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        ASSERT_EQ(coords(r,c).x, static_cast<TestType>((c+1)*0.25f));
        ASSERT_EQ(coords(r,c).y, static_cast<TestType>((r+1)*0.25f));
      }
    }
    // example-end zipvec-test-1
  }

  { // Test 2D vector types
    // Similar to the float2 example above, but we use integer values to make
    // them representable for all numeric types.
    using vec2_type = typename matx::detail::VecTypeSelector<TestType, 2>::type;
    auto v = linspace<TestType>(1, 3, 3);
    auto duplicate_rows = clone<2>(v, {3, matxKeepDim});
    auto duplicate_cols = clone<2>(v, {matxKeepDim, 3});
    auto coords = make_tensor<vec2_type>({3, 3});
    (coords = zipvec(duplicate_rows, duplicate_cols)).run(exec);
    exec.sync();

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        ASSERT_EQ(coords(i,j).x, static_cast<TestType>(j+1));
        ASSERT_EQ(coords(i,j).y, static_cast<TestType>(i+1));
      }
    }
  }

  { // Test 3D vector types
    using vec3_type = typename matx::detail::VecTypeSelector<TestType, 3>::type;
    auto v = linspace<TestType>(1, 3, 3);
    auto duplicate_rows = clone<2>(v, {3, matxKeepDim});
    auto duplicate_cols = clone<2>(v, {matxKeepDim, 3});
    auto coords = make_tensor<vec3_type>({3, 3});
    (coords = zipvec(duplicate_rows, ones<TestType>({3, 3}), duplicate_cols)).run(exec);
    exec.sync();

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        ASSERT_EQ(coords(i,j).x, static_cast<TestType>(j+1));
        ASSERT_EQ(coords(i,j).y, static_cast<TestType>(1));
        ASSERT_EQ(coords(i,j).z, static_cast<TestType>(i+1));
      }
    }
  }

  { // Test 4D vector types
    using vec4_type = typename matx::detail::VecTypeSelector<TestType, 4>::type;
    auto v = linspace<TestType>(1, 3, 3);
    auto duplicate_rows = clone<2>(v, {3, matxKeepDim});
    auto duplicate_cols = clone<2>(v, {matxKeepDim, 3});
    auto coords = make_tensor<vec4_type>({3, 3});
    auto twos = static_cast<TestType>(2)*ones<TestType>({3, 3});
    (coords = zipvec(duplicate_rows, ones<TestType>({3, 3}), duplicate_cols, twos)).run(exec);
    exec.sync();

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        ASSERT_EQ(coords(i,j).x, static_cast<TestType>(j+1));
        ASSERT_EQ(coords(i,j).y, static_cast<TestType>(1));
        ASSERT_EQ(coords(i,j).z, static_cast<TestType>(i+1));
        ASSERT_EQ(coords(i,j).w, static_cast<TestType>(2));
      }
    }
  }

  { // Test that non-narrowing conversions are supported
    {
      // short can be converted to float without narrowing
      auto t = ones<float>({1});
      auto t2 = ones<short>({1});
      auto t3 = make_tensor<float2>({1});
      (t3 = zipvec(t, t2)).run(exec);
      exec.sync();
      ASSERT_EQ(t3(0).x, static_cast<float>(1));
      ASSERT_EQ(t3(0).y, static_cast<float>(1));  
    }
    {
      // float can be converted to double without narrowing
      auto t = ones<double>({1});
      auto t2 = ones<float>({1});
      auto t3 = make_tensor<double2>({1});
      (t3 = zipvec(t, t2)).run(exec);
      exec.sync();
      ASSERT_EQ(t3(0).x, static_cast<double>(1));
      ASSERT_EQ(t3(0).y, static_cast<double>(1));  
    }
    {
      // int can be converted to double without narrowing
      auto t = ones<double>({1});
      auto t2 = ones<int>({1});
      auto t3 = make_tensor<double2>({1});
      (t3 = zipvec(t, t2)).run(exec);
      exec.sync();
      ASSERT_EQ(t3(0).x, static_cast<double>(1));
      ASSERT_EQ(t3(0).y, static_cast<double>(1));  
    }
    // Note that some narrowing conversions like int -> float are currently allowed.
    // This is because std::common_type_t<int, float> is float, and the zipvec operator
    // uses std::common_type for its value_type.
  }

  MATX_EXIT_HANDLER();
} 