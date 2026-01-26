#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

// Define functors to avoid extended lambda restrictions in private member functions
struct SquareFunctor {
  template<typename T>
  __host__ __device__ auto operator()(T x) const { return x * x; }
};

struct AddFunctor {
  template<typename T>
  __host__ __device__ auto operator()(T x, T y) const { return x + y; }
};

struct CombineFunctor {
  template<typename T>
  __host__ __device__ auto operator()(T x, T y, T z) const { return x + y * z; }
};

struct MultiplyFunctor {
  template<typename T>
  __host__ __device__ auto operator()(T x, T y) const { return x * y; }
};

template<typename T>
struct AddOneFunctor {
  __host__ __device__ auto operator()(T x) const { return x + T(1); }
};

template<typename T>
struct DoubleItFunctor {
  __host__ __device__ auto operator()(T x) const { return x * T(2); }
};

template<typename T>
struct CustomOpFunctor {
  __host__ __device__ auto operator()(T x, T y) const { return x * x + y; }
};

template<typename T>
struct IncrementFunctor {
  __host__ __device__ auto operator()(T x) const { return x + T(10); }
};

template<typename T>
struct CustomMathFunctor {
  __host__ __device__ auto operator()(T x) const { return x * x + T(2) * x + T(1); }
};

TYPED_TEST(OperatorTestsNumericAllExecs, ApplyOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test single input operator with simple lambda
    // example-begin apply-test-1
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      t_in(i) = static_cast<detail::value_promote_t<TestType>>(i);
    }

    // Apply a lambda that squares each element
    (t_out = matx::apply(SquareFunctor{}, t_in)).run(exec);
    // example-end apply-test-1
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      auto expected = t_in(i) * t_in(i);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
    }
  }

  { // Test two input operators
    // example-begin apply-test-2
    auto t_in1 = make_tensor<TestType>({10});
    auto t_in2 = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      t_in1(i) = static_cast<detail::value_promote_t<TestType>>(i);
      t_in2(i) = static_cast<detail::value_promote_t<TestType>>(i + 1);
    }

    // Apply a lambda that adds two inputs
    (t_out = matx::apply(AddFunctor{}, t_in1, t_in2)).run(exec);
    // example-end apply-test-2
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      auto expected = t_in1(i) + t_in2(i);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
    }
  }

  { // Test three input operators
    // example-begin apply-test-3
    auto t_in1 = make_tensor<TestType>({10});
    auto t_in2 = make_tensor<TestType>({10});
    auto t_in3 = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      t_in1(i) = static_cast<detail::value_promote_t<TestType>>(i);
      t_in2(i) = static_cast<detail::value_promote_t<TestType>>(i + 1);
      t_in3(i) = static_cast<detail::value_promote_t<TestType>>(i + 2);
    }

    // Apply a lambda that combines three inputs: x + y * z
    (t_out = matx::apply(CombineFunctor{}, t_in1, t_in2, t_in3)).run(exec);
    // example-end apply-test-3
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      auto expected = t_in1(i) + t_in2(i) * t_in3(i);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
    }
  }

  { // Test with 2D tensors
    auto t_in1 = make_tensor<TestType>({5, 8});
    auto t_in2 = make_tensor<TestType>({5, 8});
    auto t_out = make_tensor<TestType>({5, 8});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      for (index_t j = 0; j < t_in1.Size(1); j++) {
        t_in1(i, j) = static_cast<detail::value_promote_t<TestType>>(i * 10 + j);
        t_in2(i, j) = static_cast<detail::value_promote_t<TestType>>(i + j);
      }
    }

    // Apply a lambda that multiplies two inputs
    (t_out = matx::apply(MultiplyFunctor{}, t_in1, t_in2)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      for (index_t j = 0; j < t_in1.Size(1); j++) {
        auto expected = t_in1(i, j) * t_in2(i, j);
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), expected));
      }
    }
  }

  // Skip this test for half-precision complex types due to limited arithmetic support
  if constexpr (!is_complex_half_v<TestType>) {
    { // Test with nested operators
      auto t_in = make_tensor<TestType>({10});
      auto t_out = make_tensor<TestType>({10});

      for (index_t i = 0; i < t_in.Size(0); i++) {
        t_in(i) = static_cast<detail::value_promote_t<TestType>>(i + 1);
      }

      // Apply to a nested operator expression
      (t_out = matx::apply(AddFunctor{}, t_in * TestType(2), t_in + TestType(1))).run(exec);
      exec.sync();

      for (index_t i = 0; i < t_in.Size(0); i++) {
        auto expected = (t_in(i) * TestType(2)) + (t_in(i) + TestType(1));
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
      }
    }
  }

  { // Test chaining apply operators
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      t_in(i) = static_cast<detail::value_promote_t<TestType>>(i);
    }

    // Chain multiple apply operators
    (t_out = matx::apply(DoubleItFunctor<TestType>{}, matx::apply(AddOneFunctor<TestType>{}, t_in))).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      auto expected = (t_in(i) + TestType(1)) * TestType(2);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
    }
  }

  { // Test with 3D tensors
    auto t_in1 = make_tensor<TestType>({3, 4, 5});
    auto t_in2 = make_tensor<TestType>({3, 4, 5});
    auto t_out = make_tensor<TestType>({3, 4, 5});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      for (index_t j = 0; j < t_in1.Size(1); j++) {
        for (index_t k = 0; k < t_in1.Size(2); k++) {
          t_in1(i, j, k) = static_cast<detail::value_promote_t<TestType>>(i + j + k);
          t_in2(i, j, k) = static_cast<detail::value_promote_t<TestType>>(i * j * k + 1);
        }
      }
    }

    // Apply a lambda that performs a custom operation
    (t_out = matx::apply(CustomOpFunctor<TestType>{}, t_in1, t_in2)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      for (index_t j = 0; j < t_in1.Size(1); j++) {
        for (index_t k = 0; k < t_in1.Size(2); k++) {
          auto expected = t_in1(i, j, k) * t_in1(i, j, k) + t_in2(i, j, k);
          ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j, k), expected));
        }
      }
    }
  }

  { // Test with scalar (0D tensor)
    auto t_in = make_tensor<TestType>({});
    auto t_out = make_tensor<TestType>({});

    t_in() = static_cast<detail::value_promote_t<TestType>>(5);

    // Apply a lambda to scalar
    (t_out = matx::apply(IncrementFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    auto expected = t_in() + TestType(10);
    ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(), expected));
  }

  MATX_EXIT_HANDLER();
}

// Test with complex types
TYPED_TEST(OperatorTestsComplexTypesAllExecs, ApplyOpComplex)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test with complex multiplication
    auto t_in1 = make_tensor<TestType>({10});
    auto t_in2 = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      t_in1(i) = TestType(static_cast<typename TestType::value_type>(i), 
                          static_cast<typename TestType::value_type>(i + 1));
      t_in2(i) = TestType(static_cast<typename TestType::value_type>(2), 
                          static_cast<typename TestType::value_type>(3));
    }

    // Apply a lambda that multiplies complex numbers
    (t_out = matx::apply(MultiplyFunctor{}, t_in1, t_in2)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      auto expected = t_in1(i) * t_in2(i);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
    }
  }

  MATX_EXIT_HANDLER();
}

// Test with floating point types for more complex operations
TYPED_TEST(OperatorTestsFloatAllExecs, ApplyOpFloat)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test with mathematical operations
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      if constexpr (is_complex_v<TestType>) {
        t_in(i) = static_cast<TestType>(static_cast<typename TestType::value_type>(i + 1));
      } else {
        t_in(i) = static_cast<TestType>(i + 1);
      }
    }

    // Apply a lambda using math functions
    (t_out = matx::apply(CustomMathFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      auto x = t_in(i);
      auto expected = x * x + TestType(2) * x + TestType(1);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected, 0.01));
    }
  }

  MATX_EXIT_HANDLER();
}

