#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsNumericAllExecs, RemapRankZero)
{
  MATX_ENTER_HANDLER();
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  const int N = 16;

  // 1D source tensor cases
  {
    auto from = make_tensor<int>({N});
    (from = range<0>({N}, 0, 1)).run(exec);
    exec.sync();
    auto ind = make_tensor<int>({});
    auto r = remap<0>(from, ind);
    auto to = make_tensor<int>({1});

    ind() = N/2;
    (to = r).run(exec);
    exec.sync();

    ASSERT_EQ(to(0), N/2);

    ind() = N/4;
    (to = r).run(exec);
    exec.sync();

    ASSERT_EQ(to(0), N/4);
  }

  // 2D source tensor cases
  {
    auto from = make_tensor<int>({N,N});
    (from = ones()).run(exec);
    exec.sync();

    auto i0 = make_tensor<int>({});
    auto i1 = make_tensor<int>({});
    auto r0 = remap<0>(from, i0);
    auto r1 = remap<1>(from, i0);

    auto to0 = make_tensor<int>({1,N});
    auto to1 = make_tensor<int>({N,1});

    i0() = N/2;
    from(N/2,0) = 2;
    from(0,N/2) = 3;
    (to0 = r0).run(exec);
    (to1 = r1).run(exec);
    exec.sync();

    ASSERT_EQ(to0(0,0), 2);
    ASSERT_EQ(to0(0,1), 1);
    ASSERT_EQ(to1(0,0), 3);
    ASSERT_EQ(to1(1,0), 1);

    i0() = N/3;
    i1() = 2*(N/3);
    from(N/3, 2*(N/3)) = 11;

    // Select a single entry from the 2D input tensor
    auto entry = make_tensor<int>({1,1});
    (entry = remap<0,1>(from, i0, i1)).run(exec);
    exec.sync();

    ASSERT_EQ(entry(0,0), 11);
  }

  MATX_EXIT_HANDLER();
}