#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

// Testing 4 basic arithmetic operations with complex numbers and non-complex
TYPED_TEST(OperatorTestsComplexTypesAllExecs, ComplexTypeCompatibility)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  using scalar_type = typename TestType::value_type;

  ExecType exec{};   
  index_t count = 10;

  tensor_t<float, 1> fview({count});
  tensor_t<TestType, 1> dview({count});

  // Multiply by scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = dview * as_type<scalar_type>(fview)).run(exec);
  exec.sync();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(i * i));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(i * i));
  }

  // Divide by scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = i == 0 ? static_cast<float>(1) : static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = dview / as_type<scalar_type>(fview)).run(exec);
  exec.sync();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              i == 0 ? static_cast<detail::value_promote_t<TestType>>(0)
                     : static_cast<detail::value_promote_t<TestType>>(1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              i == 0 ? static_cast<detail::value_promote_t<TestType>>(0)
                     : static_cast<detail::value_promote_t<TestType>>(1));
  }

  // Add scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }
  

  (dview = dview + as_type<scalar_type>(fview)).run(exec);
  exec.sync();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(i + i));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(i));
  }

  // Subtract scalar from complex
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i + 1);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = dview - as_type<scalar_type>(fview)).run(exec);
  exec.sync();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(-1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(i));
  }

  // Subtract complex from scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i + 1);
    dview(i) = {static_cast<detail::value_promote_t<TestType>>(i),
                static_cast<detail::value_promote_t<TestType>>(i)};
  }

  (dview = as_type<scalar_type>(fview) - dview).run(exec);
  exec.sync();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).real()),
              static_cast<detail::value_promote_t<TestType>>(1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TestType>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TestType>>(-i));
  }

  MATX_EXIT_HANDLER();
}