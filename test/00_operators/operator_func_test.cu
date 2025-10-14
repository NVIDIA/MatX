#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();  
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  TestType d = c;
  TestType z = 0;
  tiv0() = c;

  auto tov00 = make_tensor<TestType>({});

  // example-begin IFELSE-test-1
  IFELSE(tiv0 == d, tov0 = z, tov0 = d).run(exec);
  // example-end IFELSE-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), z));

  IFELSE(tiv0 == d, tov0 = tiv0, tov0 = d).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), tiv0()));

  IFELSE(tiv0 != d, tov0 = d, tov0 = z).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), z));

  (tov0 = c, tov00 = c).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov00(), c));  

  MATX_EXIT_HANDLER();
} 


TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, OperatorFuncsR2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<typename detail::complex_from_scalar_t<TestType>>({});
  // example-begin expj-test-1
  // TestType is float, double, bf16, etc.
  tiv0() = static_cast<TestType>(M_PI/2.0);
  (tov0 = expj(tiv0)).run(exec);
  // tov0 is complex with value 0 + 1j
  // example-end expj-test-1

  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::complex_from_scalar_t<TestType>(0.0, 1.0)));

  tiv0() = static_cast<TestType>(-1.0 * M_PI);
  (tov0 = expj(tiv0)).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::complex_from_scalar_t<TestType>(-1.0, 0.0)));

  tiv0() = 0;
  (tov0 = expj(tiv0)).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::complex_from_scalar_t<TestType>(1.0, 0.0)));

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  (tov0 = expj(tiv0)).run(exec);
  exec.sync();

  EXPECT_TRUE(MatXUtils::MatXTypeCompare(
      tov0(),
      typename detail::complex_from_scalar_t<TestType>(detail::scalar_internal_cos(tiv0()), detail::scalar_internal_sin(tiv0()))));  
  MATX_EXIT_HANDLER();      
}

TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};    
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin log10-test-1
  (tov0 = log10(tiv0)).run(exec);
  // example-end log10-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_log10(c)));

  // example-begin log-test-1
  (tov0 = log(tiv0)).run(exec);
  // example-end log-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_log(c)));

  // example-begin log2-test-1
  (tov0 = log2(tiv0)).run(exec);
  // example-end log2-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_log2(c)));

  // example-begin floor-test-1
  (tov0 = floor(tiv0)).run(exec);
  // example-end floor-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_floor(c)));

  // example-begin ceil-test-1
  (tov0 = ceil(tiv0)).run(exec);
  // example-end ceil-test-1  
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_ceil(c)));

  // example-begin round-test-1
  (tov0 = round(tiv0)).run(exec);
  // example-end round-test-1  
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_round(c)));

  // example-begin sqrt-test-1
  (tov0 = sqrt(tiv0)).run(exec);
  // example-end sqrt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_sqrt(c)));      

  // example-begin rsqrt-test-1
  (tov0 = rsqrt(tiv0)).run(exec);
  // example-end rsqrt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_rsqrt(c)));   

  MATX_EXIT_HANDLER();
}

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

  auto t0 = make_tensor<TestType>({});
  (t0 = sum(a)).run(exec);
  exec.sync();
  ASSERT_EQ(t0(), static_cast<TestType>(2 * a.TotalSize()));
  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsNumericNonComplexAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};     
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  TestType d = c + 1;

  // example-begin max-el-test-1
  (tov0 = max(tiv0, d)).run(exec);
  // example-end max-el-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), std::max(c, d)));

  // example-begin min-el-test-1
  (tov0 = min(tiv0, d)).run(exec);
  // example-end min-el-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), std::min(c, d)));

  // These operators convert type T into bool
  auto tob = make_tensor<bool>({});

  // example-begin lt-test-1
  (tob = tiv0 < d).run(exec);
  // example-end lt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c < d));

  // example-begin gt-test-1
  (tob = tiv0 > d).run(exec);
  // example-end gt-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c > d));

  // example-begin lte-test-1
  (tob = tiv0 <= d).run(exec);
  // example-end lte-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c <= d));

  // example-begin gte-test-1
  (tob = tiv0 >= d).run(exec);
  // example-end gte-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c >= d));

  // example-begin eq-test-1
  (tob = tiv0 == d).run(exec);
  // example-end eq-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c == d));

  // example-begin neq-test-1
  (tob = tiv0 != d).run(exec);
  // example-end neq-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c != d));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, OperatorFuncDivComplex)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};  
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});
  typename TestType::value_type s = 5.0;

  TestType c = GenerateData<TestType>();  
  tiv0() = c;

  (tov0 = s / tiv0).run(exec);
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), s / tiv0()));

  MATX_EXIT_HANDLER();  
}


TYPED_TEST(OperatorTestsNumericAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};    
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin add-test-1
  (tov0 = tiv0 + tiv0).run(exec);
  // example-end add-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c + c));

  // example-begin sub-test-1
  (tov0 = tiv0 - tiv0).run(exec);
  // example-end sub-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c - c));

  // example-begin mul-test-1
  (tov0 = tiv0 * tiv0).run(exec);
  // example-end mul-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c * c));

  // example-begin div-test-1
  (tov0 = tiv0 / tiv0).run(exec);
  // example-end div-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c / c));

  // example-begin neg-test-1
  (tov0 = -tiv0).run(exec);
  // example-end neg-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), -c));

  // example-begin IF-test-1
  IF(tiv0 == tiv0, tov0 = c).run(exec);
  // example-end IF-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));

  TestType p = 2.0f;
  // example-begin pow-test-1
  (tov0 = as_type<TestType>(pow(tiv0, p))).run(exec);
  // example-end pow-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_pow(c, p)));

  TestType three = 3.0f;

  (tov0 = tiv0 * tiv0 * (tiv0 + tiv0) / tiv0 + three).run(exec);
  exec.sync();

  TestType res;
  res = c * c * (c + c) / c + three;
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), res, 0.07));


  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsIntegralAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};    
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;
  TestType mod = 2;

  // example-begin mod-test-1
  (tov0 = tiv0 % mod).run(exec);
  // example-end mod-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c % mod));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsBooleanAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  TestType d = false;
  tiv0() = c;

  // example-begin land-test-1
  (tov0 = tiv0 && d).run(exec);
  // example-end land-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c && d));

  // example-begin lor-test-1
  (tov0 = tiv0 || d).run(exec);
  // example-end lor-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c || d));

  // example-begin lnot-test-1
  (tov0 = !tiv0).run(exec);
  // example-end lnot-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), !c));

  // example-begin xor-test-1
  (tov0 = tiv0 ^ d).run(exec);
  // example-end xor-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c ^ d));

  // example-begin or-test-1
  (tov0 = tiv0 | d).run(exec);
  // example-end or-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c | d));

  // example-begin and-test-1
  (tov0 = tiv0 & d).run(exec);
  // example-end and-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c & d));    

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplexTypesAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   

  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin exp-test-1
  (tov0 = exp(tiv0)).run(exec);
  // example-end exp-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_exp(c)));

  // example-begin conj-test-1
  (tov0 = conj(tiv0)).run(exec);
  // example-end conj-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::scalar_internal_conj(c)));

  // abs takes a complex and output a floating point value
  auto tdd0 = make_tensor<typename TestType::value_type>({});

  // example-begin abs-test-1
  (tdd0 = abs(tiv0)).run(exec);
  // example-end abs-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tdd0(), detail::scalar_internal_abs(c)));

  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsQuaternionTypesAllExecs, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   

  auto tiv0 = make_tensor<TestType>({});
  auto tov0 = make_tensor<TestType>({});

  TestType c = GenerateData<TestType>();
  tiv0() = c;

  // example-begin exp-test-1
  (tov0 = exp(tiv0)).run(exec);
  // example-end exp-test-1
  exec.sync();
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_exp(c)));

  MATX_EXIT_HANDLER();
}
