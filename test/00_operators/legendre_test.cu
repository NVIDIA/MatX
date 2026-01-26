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