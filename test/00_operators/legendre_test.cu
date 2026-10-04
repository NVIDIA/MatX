#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

template<class TypeParam>
TypeParam legendre_check(int n, int m, TypeParam x) {
  if (m > n ) return 0;

  TypeParam a = detail::scalar_internal_sqrt(TypeParam(1)-x*x);
  // first we will move move along diagonal

  // initialize registers
  TypeParam d1 = 1, d0;

  for(int i=0; i < m; i++) {
    // advance diagonal (shift)
    d0 = d1;
    // compute next term using recurrence relationship
    d1 = -TypeParam(2*i+1)*a*d0;
  }

  // next we will move to the right till we get to the correct entry

  // initialize registers
  TypeParam p0, p1 = 0, p2 = d1;

  for(int l=m; l<n; l++) {
    // advance one step (shift)
    p0 = p1;
    p1 = p2;

    // Compute next term using recurrence relationship
    p2 = (TypeParam(2*l+1) * x * p1 - TypeParam(l+m)*p0)/(TypeParam(l-m+1));
  }

  return p2;
}

// No JIT until constexpr half is fixed
TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Legendre)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  index_t size = 11;
  int order = 5;
  
  { // vector for n and m
    // example-begin legendre-test-1
    auto n = range<0, 1, int>({order}, 0, 1);
    auto m = range<0, 1, int>({order}, 0, 1);
    auto x = as_type<TestType>(linspace(TestType(0), TestType(1), size));

    auto out = make_tensor<TestType>({order, order, size});

    (out = legendre(n, m, x)).run(exec);
    // example-end legendre-test-1

    exec.sync();

    for(int j = 0; j < order; j++) {
      for(int p = 0; p < order; p++) {
        for(int i = 0 ; i < size; i++) {
          if constexpr (is_matx_half_v<TestType>) {
            ASSERT_NEAR(out(p,j,i), legendre_check(p, j, x(i)),50.0);
          }
          else {
            ASSERT_NEAR(out(p,j,i), legendre_check(p, j, x(i)),.0001);
          }
        }
      }
    }
  }
 
  { // constant for n
    auto m = range<0, 1, int>({order}, 0, 1);
    auto x = as_type<TestType>(linspace(TestType(0), TestType(1), size));

    auto out = make_tensor<TestType>({order, size});

    (out = lcollapse<2>(legendre(order, m, x))).run(exec);

    exec.sync();

    for(int i = 0 ; i < size; i++) {
      for(int p = 0; p < order; p++) {
        if constexpr (is_matx_half_v<TestType>) {
          ASSERT_NEAR(out(p,i), legendre_check(order, p, x(i)),50.0);
        }
        else {
          ASSERT_NEAR(out(p,i), legendre_check(order, p, x(i)),.0001);
        }        
      }
    }
  }

  { // taking a constant for m and n;
    auto x = as_type<TestType>(linspace(TestType(0), TestType(1), size));

    auto out = make_tensor<TestType>({size});

    (out = lcollapse<3>(legendre(order, order,  x))).run(exec);

    exec.sync();

    for(int i = 0 ; i < size; i++) {
      if constexpr (is_matx_half_v<TestType>) {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),50.0);
      }
      else {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),.0001);
      }        
    }
  }
  
  { // taking a rank0 tensor for m and constant for n
    auto x = as_type<TestType>(linspace(TestType(0), TestType(1), size));
    auto m = make_tensor<int>({});
    auto out = make_tensor<TestType>({size});
    m() = order;

    (out = lcollapse<3>(legendre(order, m,  x))).run(exec);

    exec.sync();

    for(int i = 0 ; i < size; i++) {
      if constexpr (is_matx_half_v<TestType>) {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),50.0);
      }
      else {
        ASSERT_NEAR(out(i), legendre_check(order, order, x(i)),.0001);
      }
    }
  }
  MATX_EXIT_HANDLER();
}

// Places the n and m dimensions at every ordered pair of output axes
TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, LegendreAxes)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (is_cuda_jit_executor_v<ExecType>) {
    // JIT legendre faults on a rank-2 input for every axis placement, including the default
    GTEST_SKIP() << "JIT legendre does not support a rank-2 input";
  }

  ExecType exec{};

  const int order = 3;
  const index_t x0_size = 2;
  const index_t x1_size = 4;
  const double tol = is_matx_half_v<TestType> ? 50.0 : .0001;

  auto n = range<0, 1, int>({order}, 0, 1);
  auto m = range<0, 1, int>({order}, 0, 1);
  auto x = make_tensor<TestType>({x0_size, x1_size});
  for (index_t i = 0; i < x0_size; i++) {
    for (index_t j = 0; j < x1_size; j++) {
      x(i, j) = TestType(static_cast<float>(i * x1_size + j) / static_cast<float>(x0_size * x1_size));
    }
  }

  for (int an = 0; an < 4; an++) {
    for (int am = 0; am < 4; am++) {
      if (an == am) continue;

      cuda::std::array<index_t, 4> shape;
      const index_t x_sizes[2] = {x0_size, x1_size};
      for (int d = 0, xd = 0; d < 4; d++) {
        shape[d] = (d == an || d == am) ? order : x_sizes[xd++];
      }
      auto out = make_tensor<TestType>(shape);

      (out = legendre(n, m, x, cuda::std::array<int, 2>{an, am})).run(exec);
      exec.sync();

      for (index_t i0 = 0; i0 < shape[0]; i0++) {
        for (index_t i1 = 0; i1 < shape[1]; i1++) {
          for (index_t i2 = 0; i2 < shape[2]; i2++) {
            for (index_t i3 = 0; i3 < shape[3]; i3++) {
              const index_t idx[4] = {i0, i1, i2, i3};
              index_t xi[2];
              for (int d = 0, xd = 0; d < 4; d++) {
                if (d != an && d != am) xi[xd++] = idx[d];
              }
              ASSERT_NEAR(out(i0, i1, i2, i3),
                          legendre_check(static_cast<int>(idx[an]), static_cast<int>(idx[am]), x(xi[0], xi[1])), tol)
                  << "axis={" << an << "," << am << "}";
            }
          }
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}
// legendre rejects repeated or out-of-range axes in every build mode
TEST(OperatorValidationTests, LegendreInvalidAxes)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto n = range<0, 1, int>({3}, 0, 1);
  auto m = range<0, 1, int>({3}, 0, 1);
  auto x = make_tensor<TestType>({5});  // output rank 3

  EXPECT_THROW(legendre(n, m, x, cuda::std::array<int, 2>{0, 0}), matx::detail::matxException);
  EXPECT_THROW(legendre(n, m, x, cuda::std::array<int, 2>{0, 3}), matx::detail::matxException);
  EXPECT_THROW(legendre(n, m, x, cuda::std::array<int, 2>{-1, 1}), matx::detail::matxException);
  EXPECT_NO_THROW(legendre(n, m, x, cuda::std::array<int, 2>{1, 0}));
  EXPECT_NO_THROW(legendre(n, m, x, cuda::std::array<int, 2>{0, 2}));

  MATX_EXIT_HANDLER();
}
