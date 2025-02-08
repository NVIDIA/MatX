#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TEST(OperatorTests, Cast)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 4;
  auto t = make_tensor<int8_t>({count0});
  auto t2 = make_tensor<int8_t>({count0});
  auto to = make_tensor<float>({count0});

  cudaExecutor exec{};

  t.SetVals({126, 126, 126, 126});
  t2.SetVals({126, 126, 126, 126});
  
  // example-begin as_type-test-1
  (to = as_type<int8_t>(t + t2)).run(exec);
  // example-end as_type-test-1
  exec.sync();

  for (int i = 0; i < t.Size(0); i++) {
    ASSERT_EQ(to(i), -4); // -4 from 126 + 126 wrap-around
  }

  // example-begin as_int8-test-1
  (to = as_int8(t + t2)).run(exec);
  // example-end as_int8-test-1
  exec.sync();
  
  for (int i = 0; i < t.Size(0); i++) {
    ASSERT_EQ(to(i), -4); // -4 from 126 + 126 wrap-around
  }  

  // example-begin as_complex_float-test-1
  auto c32 = make_tensor<cuda::std::complex<float>>({});
  auto s64 = make_tensor<double>({});
  s64.SetVals({5.0});
  (c32 = as_complex_float(s64)).run();
  // c32() will be (5.0f, 0.0f)
  // example-end as_complex_float-test-1

  // example-begin as_complex_double-test-1
  auto c64 = make_tensor<cuda::std::complex<double>>({});
  auto s32 = make_tensor<float>({});
  s32.SetVals({3.0f});
  (c64 = as_complex_double(s32)).run();
  // c64() will be (3.0, 0.0)
  // example-end as_complex_double-test-1

  cudaStreamSynchronize(0);

  ASSERT_EQ(c32().real(), 5.0f);
  ASSERT_EQ(c32().imag(), 0.0f);
  ASSERT_EQ(c64().real(), 3.0);
  ASSERT_EQ(c64().imag(), 0.0);

  (c32 = as_complex_float(s32)).run();
  (c64 = as_complex_double(s64)).run();
  cudaStreamSynchronize(0);

  ASSERT_EQ(c32().real(), 3.0f);
  ASSERT_EQ(c32().imag(), 0.0f);
  ASSERT_EQ(c64().real(), 5.0);
  ASSERT_EQ(c64().imag(), 0.0);

  MATX_EXIT_HANDLER();
}