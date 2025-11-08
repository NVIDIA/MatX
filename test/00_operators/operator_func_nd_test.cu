#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, NDOperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   

  auto a = make_tensor<TestType>({1,2,3,4,5});
  auto b = make_tensor<TestType>({1,2,3,4,5});
  (a = ones<TestType>(a.Shape())).run(exec);
  exec.sync();
  (b = ones<TestType>(b.Shape())).run(exec);
  exec.sync();
  (a = a + b).run(exec);

  {
    if constexpr (is_cuda_non_jit_executor<ExecType>) {
      auto t0 = make_tensor<TestType>({});
      (t0 = sum(a)).run(exec);
      exec.sync();
      ASSERT_EQ(t0(), static_cast<TestType>(2 * a.TotalSize()));
    }
 }
  MATX_EXIT_HANDLER();
}

