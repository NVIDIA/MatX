#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

// Define functors for apply_idx tests

// Simple functor that accesses the current element using indices
template<typename T>
struct AccessCurrentFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 1> idx, const Op& op) const {
    return op(idx[0]);
  }
};

// Functor that accesses current element in 2D
template<typename T>
struct AccessCurrent2DFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
    return op(idx[0], idx[1]);
  }
};

// 3-point moving average stencil
template<typename T>
struct Stencil3Functor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 1> idx, const Op& op) const {
    auto i = idx[0];
    // Handle boundaries
    if (i == 0) return op(i);
    if (i == op.Size(0) - 1) return op(i);
    // 3-point average
    return (op(i-1) + op(i) + op(i+1)) / T(3);
  }
};

// Functor that combines two operators using indices
template<typename T>
struct CombineWithIndexFunctor {
  template<typename Op1, typename Op2>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 1> idx, const Op1& op1, const Op2& op2) const {
    auto i = idx[0];
    using value_t = typename detail::value_promote_t<T>;
    return op1(i) * T(static_cast<value_t>(i)) + op2(i);
  }
};

// Functor for 2D stencil (simple neighbor sum)
template<typename T>
struct Stencil2DFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
    auto i = idx[0];
    auto j = idx[1];
    
    // Handle boundaries by returning current value
    if (i == 0 || i == op.Size(0) - 1 || j == 0 || j == op.Size(1) - 1) {
      return op(i, j);
    }
    
    // Sum of 4 neighbors
    return (op(i-1, j) + op(i+1, j) + op(i, j-1) + op(i, j+1)) / T(4);
  }
};

// Functor that accesses multiple operators at current indices
template<typename T>
struct MultiOpAccessFunctor {
  template<typename Op1, typename Op2, typename Op3>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 1> idx, 
                                      const Op1& op1, const Op2& op2, const Op3& op3) const {
    auto i = idx[0];
    return op1(i) + op2(i) * op3(i);
  }
};

// Functor for scalar (0D tensor)
template<typename T>
struct ScalarAccessFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 0> /* idx */, const Op& op) const {
    return op() * T(2);
  }
};

// Functor that uses index values in calculation
template<typename T>
struct IndexWeightedFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
    auto i = idx[0];
    auto j = idx[1];
    using value_t = typename detail::value_promote_t<T>;
    return op(i, j) * T(static_cast<value_t>(i + j + 1));
  }
};

// Functor for 3D tensor access
template<typename T>
struct Access3DFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 3> idx, const Op& op) const {
    return op(idx[0], idx[1], idx[2]);
  }
};

// Functor that accesses diagonal in 2D
template<typename T>
struct DiagonalAccessFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
    auto i = idx[0];
    auto j = idx[1];
    // Access diagonal element (wrap around if needed)
    auto diag_idx = (i + j) % op.Size(0);
    return op(diag_idx, diag_idx % op.Size(1));
  }
};

TYPED_TEST(OperatorTestsNumericAllExecs, ApplyIdxOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test single input operator - simple element access
    // example-begin apply-idx-test-1
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      t_in(i) = static_cast<detail::value_promote_t<TestType>>(i + 1);
    }

    // Apply a functor that accesses the current element using indices
    (t_out = matx::apply_idx(AccessCurrentFunctor<TestType>{}, t_in)).run(exec);
    // example-end apply-idx-test-1
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), t_in(i)));
    }
  }

  { // Test with 3-point stencil
    // example-begin apply-idx-test-2
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      t_in(i) = static_cast<detail::value_promote_t<TestType>>(i);
    }

    // Apply a 3-point moving average stencil
    (t_out = matx::apply_idx(Stencil3Functor<TestType>{}, t_in)).run(exec);
    // example-end apply-idx-test-2
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      TestType expected;
      if (i == 0 || i == t_in.Size(0) - 1) {
        expected = t_in(i);
      } else {
        expected = (t_in(i-1) + t_in(i) + t_in(i+1)) / TestType(3);
      }
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected, 0.01));
    }
  }

  { // Test with two input operators
    // example-begin apply-idx-test-3
    auto t_in1 = make_tensor<TestType>({10});
    auto t_in2 = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      t_in1(i) = static_cast<detail::value_promote_t<TestType>>(i + 1);
      t_in2(i) = static_cast<detail::value_promote_t<TestType>>(i * 2);
    }

    // Apply a functor that combines two operators with index weighting
    (t_out = matx::apply_idx(CombineWithIndexFunctor<TestType>{}, t_in1, t_in2)).run(exec);
    // example-end apply-idx-test-3
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      auto expected = t_in1(i) * TestType(static_cast<detail::value_promote_t<TestType>>(i)) + t_in2(i);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
    }
  }

  { // Test with 2D tensors
    auto t_in = make_tensor<TestType>({5, 8});
    auto t_out = make_tensor<TestType>({5, 8});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        t_in(i, j) = static_cast<detail::value_promote_t<TestType>>(i * 10 + j);
      }
    }

    // Apply a functor that accesses current element in 2D
    (t_out = matx::apply_idx(AccessCurrent2DFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), t_in(i, j)));
      }
    }
  }

  { // Test with 2D stencil
    auto t_in = make_tensor<TestType>({5, 8});
    auto t_out = make_tensor<TestType>({5, 8});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        t_in(i, j) = static_cast<detail::value_promote_t<TestType>>(i + j);
      }
    }

    // Apply a 2D stencil (average of 4 neighbors)
    (t_out = matx::apply_idx(Stencil2DFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        TestType expected;
        if (i == 0 || i == t_in.Size(0) - 1 || j == 0 || j == t_in.Size(1) - 1) {
          expected = t_in(i, j);
        } else {
          expected = (t_in(i-1, j) + t_in(i+1, j) + t_in(i, j-1) + t_in(i, j+1)) / TestType(4);
        }
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), expected, 0.01));
      }
    }
  }

  { // Test with three input operators
    auto t_in1 = make_tensor<TestType>({10});
    auto t_in2 = make_tensor<TestType>({10});
    auto t_in3 = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      t_in1(i) = static_cast<detail::value_promote_t<TestType>>(i);
      t_in2(i) = static_cast<detail::value_promote_t<TestType>>(i + 1);
      t_in3(i) = static_cast<detail::value_promote_t<TestType>>(i + 2);
    }

    // Apply a functor that combines three operators: op1 + op2 * op3
    (t_out = matx::apply_idx(MultiOpAccessFunctor<TestType>{}, t_in1, t_in2, t_in3)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      auto expected = t_in1(i) + t_in2(i) * t_in3(i);
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected));
    }
  }

  { // Test with 3D tensors
    auto t_in = make_tensor<TestType>({3, 4, 5});
    auto t_out = make_tensor<TestType>({3, 4, 5});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        for (index_t k = 0; k < t_in.Size(2); k++) {
          t_in(i, j, k) = static_cast<detail::value_promote_t<TestType>>(i + j + k);
        }
      }
    }

    // Apply a functor that accesses elements in 3D
    (t_out = matx::apply_idx(Access3DFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        for (index_t k = 0; k < t_in.Size(2); k++) {
          ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j, k), t_in(i, j, k)));
        }
      }
    }
  }

  { // Test with scalar (0D tensor)
    auto t_in = make_tensor<TestType>({});
    auto t_out = make_tensor<TestType>({});

    t_in() = static_cast<detail::value_promote_t<TestType>>(5);

    // Apply a functor to scalar
    (t_out = matx::apply_idx(ScalarAccessFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    auto expected = t_in() * TestType(2);
    ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(), expected));
  }

  { // Test with index-weighted operation
    auto t_in = make_tensor<TestType>({3, 4});
    auto t_out = make_tensor<TestType>({3, 4});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        t_in(i, j) = static_cast<detail::value_promote_t<TestType>>(2);
      }
    }

    // Apply a functor that weights by index values
    (t_out = matx::apply_idx(IndexWeightedFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        auto expected = t_in(i, j) * TestType(static_cast<detail::value_promote_t<TestType>>(i + j + 1));
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), expected));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// Test with complex types
TYPED_TEST(OperatorTestsComplexTypesAllExecs, ApplyIdxOpComplex)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test with complex numbers
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      t_in(i) = TestType(static_cast<typename TestType::value_type>(i), 
                         static_cast<typename TestType::value_type>(i + 1));
    }

    // Apply a functor that accesses current element
    (t_out = matx::apply_idx(AccessCurrentFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), t_in(i)));
    }
  }

  MATX_EXIT_HANDLER();
}

// Test with floating point types
TYPED_TEST(OperatorTestsFloatAllExecs, ApplyIdxOpFloat)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test diagonal access in 2D
    auto t_in = make_tensor<TestType>({5, 5});
    auto t_out = make_tensor<TestType>({5, 5});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        if constexpr (is_complex_v<TestType>) {
          t_in(i, j) = static_cast<TestType>(static_cast<typename TestType::value_type>(i * 10 + j));
        } else {
          t_in(i, j) = static_cast<TestType>(i * 10 + j);
        }
      }
    }

    // Apply a functor that accesses diagonal elements
    (t_out = matx::apply_idx(DiagonalAccessFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        auto diag_idx = (i + j) % t_in.Size(0);
        auto expected = t_in(diag_idx, diag_idx % t_in.Size(1));
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), expected, 0.01));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// Functors using cuda::std::apply to convert indices to parameter pack

// Helper struct for applying indices
template<typename Op>
struct ApplyIndices1D {
  const Op& op;
  __host__ __device__ ApplyIndices1D(const Op& o) : op(o) {}
  __host__ __device__ auto operator()(index_t i) const {
    return op(i);
  }
};

template<typename Op>
struct ApplyIndices2D {
  const Op& op;
  __host__ __device__ ApplyIndices2D(const Op& o) : op(o) {}
  __host__ __device__ auto operator()(index_t i, index_t j) const {
    return op(i, j);
  }
};

template<typename Op>
struct ApplyIndices3D {
  const Op& op;
  __host__ __device__ ApplyIndices3D(const Op& o) : op(o) {}
  __host__ __device__ auto operator()(index_t i, index_t j, index_t k) const {
    return op(i, j, k);
  }
};

// 1D element access using cuda::std::apply
template<typename T>
struct ApplyIndex1DFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 1> idx, const Op& op) const {
    // Use cuda::std::apply to convert array to parameter pack
    return cuda::std::apply(ApplyIndices1D<Op>(op), idx);
  }
};

// 2D element access using cuda::std::apply
template<typename T>
struct ApplyIndex2DFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
    return cuda::std::apply(ApplyIndices2D<Op>(op), idx);
  }
};

// 3D element access using cuda::std::apply
template<typename T>
struct ApplyIndex3DFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 3> idx, const Op& op) const {
    return cuda::std::apply(ApplyIndices3D<Op>(op), idx);
  }
};

// Transpose using cuda::std::apply with reversed indices
template<typename T>
struct TransposeApplyFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
    // Reverse the indices for transposition
    cuda::std::array<index_t, 2> transposed_idx = {idx[1], idx[0]};
    return cuda::std::apply(ApplyIndices2D<Op>(op), transposed_idx);
  }
};

// Multiple operators with cuda::std::apply
template<typename T>
struct MultiOpApplyFunctor {
  template<typename Op1, typename Op2>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, 
                                      const Op1& op1, const Op2& op2) const {
    auto val1 = cuda::std::apply(ApplyIndices2D<Op1>(op1), idx);
    auto val2 = cuda::std::apply(ApplyIndices2D<Op2>(op2), idx);
    return val1 + val2;
  }
};

// Stencil using cuda::std::apply with modified indices
template<typename T>
struct StencilApplyFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 1> idx, const Op& op) const {
    auto i = idx[0];
    
    if (i == 0 || i == op.Size(0) - 1) {
      return cuda::std::apply(ApplyIndices1D<Op>(op), idx);
    }
    
    // Access neighbors using modified index arrays
    cuda::std::array<index_t, 1> left_idx = {i - 1};
    cuda::std::array<index_t, 1> right_idx = {i + 1};
    
    auto left = cuda::std::apply(ApplyIndices1D<Op>(op), left_idx);
    auto center = cuda::std::apply(ApplyIndices1D<Op>(op), idx);
    auto right = cuda::std::apply(ApplyIndices1D<Op>(op), right_idx);
    
    return (left + center + right) / T(3);
  }
};

// 2D neighbor access using cuda::std::apply
template<typename T>
struct Neighbor2DApplyFunctor {
  template<typename Op>
  __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
    auto i = idx[0];
    auto j = idx[1];
    
    if (i == 0 || i == op.Size(0) - 1 || j == 0 || j == op.Size(1) - 1) {
      return cuda::std::apply(ApplyIndices2D<Op>(op), idx);
    }
    
    // Access 4 neighbors
    cuda::std::array<index_t, 2> top = {i - 1, j};
    cuda::std::array<index_t, 2> bottom = {i + 1, j};
    cuda::std::array<index_t, 2> left = {i, j - 1};
    cuda::std::array<index_t, 2> right = {i, j + 1};
    
    auto sum = cuda::std::apply(ApplyIndices2D<Op>(op), top) +
               cuda::std::apply(ApplyIndices2D<Op>(op), bottom) +
               cuda::std::apply(ApplyIndices2D<Op>(op), left) +
               cuda::std::apply(ApplyIndices2D<Op>(op), right);
    
    return sum / T(4);
  }
};

// Tests using cuda::std::apply for index unpacking
TYPED_TEST(OperatorTestsNumericAllExecs, ApplyIdxWithStdApply)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  { // Test 1D element access using cuda::std::apply
    // example-begin apply-idx-test-4
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      t_in(i) = static_cast<detail::value_promote_t<TestType>>(i + 1);
    }

    // Use cuda::std::apply to unpack indices
    (t_out = matx::apply_idx(ApplyIndex1DFunctor<TestType>{}, t_in)).run(exec);
    // example-end apply-idx-test-4
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), t_in(i)));
    }
  }

  { // Test 2D element access using cuda::std::apply
    auto t_in = make_tensor<TestType>({5, 8});
    auto t_out = make_tensor<TestType>({5, 8});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        t_in(i, j) = static_cast<detail::value_promote_t<TestType>>(i * 10 + j);
      }
    }

    (t_out = matx::apply_idx(ApplyIndex2DFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), t_in(i, j)));
      }
    }
  }

  { // Test 3D element access using cuda::std::apply
    auto t_in = make_tensor<TestType>({3, 4, 5});
    auto t_out = make_tensor<TestType>({3, 4, 5});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        for (index_t k = 0; k < t_in.Size(2); k++) {
          t_in(i, j, k) = static_cast<detail::value_promote_t<TestType>>(i * 100 + j * 10 + k);
        }
      }
    }

    (t_out = matx::apply_idx(ApplyIndex3DFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        for (index_t k = 0; k < t_in.Size(2); k++) {
          ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j, k), t_in(i, j, k)));
        }
      }
    }
  }

  { // Test transpose using cuda::std::apply with modified indices
    auto t_in = make_tensor<TestType>({5, 8});
    auto t_out = make_tensor<TestType>({5, 8});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        t_in(i, j) = static_cast<detail::value_promote_t<TestType>>(i * 100 + j);
      }
    }

    // Note: Output shape matches input, but we access transposed indices
    // For a proper transpose, output would be (8, 5), but this demonstrates the technique
    (t_out = matx::apply_idx(TransposeApplyFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        // We're accessing [j, i] instead of [i, j]
        if (j < t_in.Size(0) && i < t_in.Size(1)) {
          auto expected = t_in(j, i);
          ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), expected));
        }
      }
    }
  }

  { // Test multiple operators with cuda::std::apply
    auto t_in1 = make_tensor<TestType>({5, 8});
    auto t_in2 = make_tensor<TestType>({5, 8});
    auto t_out = make_tensor<TestType>({5, 8});

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      for (index_t j = 0; j < t_in1.Size(1); j++) {
        t_in1(i, j) = static_cast<detail::value_promote_t<TestType>>(i + j);
        t_in2(i, j) = static_cast<detail::value_promote_t<TestType>>(i * j);
      }
    }

    (t_out = matx::apply_idx(MultiOpApplyFunctor<TestType>{}, t_in1, t_in2)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in1.Size(0); i++) {
      for (index_t j = 0; j < t_in1.Size(1); j++) {
        auto expected = t_in1(i, j) + t_in2(i, j);
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), expected));
      }
    }
  }

  { // Test stencil using cuda::std::apply with modified indices
    auto t_in = make_tensor<TestType>({10});
    auto t_out = make_tensor<TestType>({10});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      t_in(i) = static_cast<detail::value_promote_t<TestType>>(i);
    }

    (t_out = matx::apply_idx(StencilApplyFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      TestType expected;
      if (i == 0 || i == t_in.Size(0) - 1) {
        expected = t_in(i);
      } else {
        expected = (t_in(i-1) + t_in(i) + t_in(i+1)) / TestType(3);
      }
      ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i), expected, 0.01));
    }
  }

  { // Test 2D neighbor access using cuda::std::apply
    auto t_in = make_tensor<TestType>({5, 8});
    auto t_out = make_tensor<TestType>({5, 8});

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        t_in(i, j) = static_cast<detail::value_promote_t<TestType>>(i + j);
      }
    }

    (t_out = matx::apply_idx(Neighbor2DApplyFunctor<TestType>{}, t_in)).run(exec);
    exec.sync();

    for (index_t i = 0; i < t_in.Size(0); i++) {
      for (index_t j = 0; j < t_in.Size(1); j++) {
        TestType expected;
        if (i == 0 || i == t_in.Size(0) - 1 || j == 0 || j == t_in.Size(1) - 1) {
          expected = t_in(i, j);
        } else {
          expected = (t_in(i-1, j) + t_in(i+1, j) + t_in(i, j-1) + t_in(i, j+1)) / TestType(4);
        }
        ASSERT_TRUE(MatXUtils::MatXTypeCompare(t_out(i, j), expected, 0.01));
      }
    }
  }

  MATX_EXIT_HANDLER();
}


