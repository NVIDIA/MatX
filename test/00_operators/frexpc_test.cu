#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TYPED_TEST(OperatorTestsComplexNonHalfTypesAllExecs, Frexpc)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  // example-begin frexpc-test-1
  // Input data
  auto tiv0  = make_tensor<TestType>({10});

  // Output fractional/integer parts
  auto tofrac_real = make_tensor<typename TestType::value_type>({10});
  auto tofrac_imag = make_tensor<typename TestType::value_type>({10});
  auto toint_real  = make_tensor<int>({10});
  auto toint_imag  = make_tensor<int>({10});

  // Create operators representing fractional and integer
  (tiv0 = random<TestType>(tiv0.Shape(), NORMAL)).run(exec);
  const auto [ofrac_real, oint_real, ofrac_imag, oint_imag] = frexpc(tiv0);
  
  ( tofrac_real = ofrac_real, 
    toint_real = oint_real,
    tofrac_imag = ofrac_imag, 
    toint_imag = oint_imag).run(exec);
  // example-end frexpc-test-1

  exec.sync();
  int texp_real, texp_imag;  
  for (int i = 0; i < tiv0.Size(0); i++) {
    if constexpr (std::is_same_v<TypeParam, cuda::std::complex<float>>) {
      float tfrac_real = cuda::std::frexpf(tiv0(i).real(), &texp_real);
      float tfrac_imag = cuda::std::frexpf(tiv0(i).imag(), &texp_imag);
      ASSERT_EQ(tfrac_real, tofrac_real(i)); 
      ASSERT_EQ(texp_real,  toint_real(i)); 
      ASSERT_EQ(tfrac_imag, tofrac_imag(i)); 
      ASSERT_EQ(texp_imag,  toint_imag(i));       
    }
    else {
      double tfrac_real = cuda::std::frexp(tiv0(i).real(), &texp_real);
      double tfrac_imag = cuda::std::frexp(tiv0(i).imag(), &texp_imag);
      ASSERT_EQ(tfrac_real, tofrac_real(i)); 
      ASSERT_EQ(texp_real,  toint_real(i)); 
      ASSERT_EQ(tfrac_imag, tofrac_imag(i)); 
      ASSERT_EQ(texp_imag,  toint_imag(i));      
    }
  }

  MATX_EXIT_HANDLER();
} 