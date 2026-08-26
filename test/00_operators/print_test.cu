#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsFloatAllExecs, Print)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t1 = make_tensor<TestType>({3});
  auto r1 = ones<TestType>(t1.Shape());
  print(r1);

  auto t3 = matx::make_tensor<TestType>({3, 2, 20});
  print(matx::ones(t3.Shape()), 1, 0, 2);

  auto t5 = matx::make_tensor<TestType>({2, 2, 2, 2, 2});
  auto r5 = matx::ones(t5.Shape());
  print(r5, 1, 0, 0, 0, 2);

  set_print_format_type(MATX_PRINT_FORMAT_MLAB);
  print(r5, 1, 0, 0, 0, 2);

  set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
  print(r5, 1, 0, 0, 0, 2);

  set_print_format_type(MATX_PRINT_FORMAT_DEFAULT);

  MATX_EXIT_HANDLER();
}

// Compile coverage for the two shapes of operator PrintData() has to
// classify memory for: a view whose data pointer is offset into its allocation,
// and the tensor_impl_t base class, which carries no storage to take a base
// pointer from and so must fall back to Data(). The second case is a build
// regression guard.
//
// This asserts no printed output. Both spellings print correctly
// either way: an unresolved pointer falls through to cuPointerGetAttributes,
// which classifies offset pointers fine wherever the driver knows the
// allocation. The fix removes the dependency on that fallback.
TYPED_TEST(OperatorTestsFloatAllExecs, PrintOffsetViewAndTensorImpl)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto t = make_tensor<TestType>({5, 10, 20});
  (t = zeros<TestType>(t.Shape())).run(exec);
  exec.sync();

  // A slice starting partway into the allocation, so Data() is not the base pointer
  auto s = t.Slice({1, 2, 3}, {4, 8, 15});
  ASSERT_NE(static_cast<const void *>(s.Data()),
            static_cast<const void *>(t.GetStorage().data()));
  print(s, 1, 1, 2);

  // print() is also instantiated on the tensor_impl_t base class
  const detail::tensor_impl_t<TestType, 3> &ti = t;
  print(ti, 1, 1, 2);

  MATX_EXIT_HANDLER();
}
