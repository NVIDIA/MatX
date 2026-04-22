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
  for (index_t i = 0; i < tiv0.Size(0); i++) {
    using InnerType = typename TestType::value_type;
    const double xr = (static_cast<double>(i) - 5.0) * 0.25;
    const double xi = (static_cast<double>(i) - 3.0) * 0.5;
    tiv0(i) = TestType(static_cast<InnerType>(xr),
                       static_cast<InnerType>(xi));
  }
  exec.sync();
  const auto [ofrac_real, oint_real, ofrac_imag, oint_imag] = frexpc(tiv0);
  
  (tofrac_real = ofrac_real).run(exec);
  (toint_real = oint_real).run(exec);
  (tofrac_imag = ofrac_imag).run(exec);
  (toint_imag = oint_imag).run(exec);
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
