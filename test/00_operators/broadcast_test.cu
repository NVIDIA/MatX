#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsNumericAllExecs, Broadcast)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{}; 

  {
    auto t0 = make_tensor<TestType>({});
    tensor_t<TestType, 4> t4i({10, 20, 30, 40});
    tensor_t<TestType, 4> t4o({10, 20, 30, 40});
    (t4o = t0).run(exec);
    exec.sync();

    t0() = (TestType)2.0f;
    for (index_t i = 0; i < t4i.Size(0); i++) {
      for (index_t j = 0; j < t4i.Size(1); j++) {
        for (index_t k = 0; k < t4i.Size(2); k++) {
          for (index_t l = 0; l < t4i.Size(3); l++) {
            t4i(i, j, k, l) =
                static_cast<detail::value_promote_t<TestType>>(i + j + k + l);
          }
        }
      }
    }

    (t4o = t4i * t0).run(exec);
    exec.sync();
  
    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TestType>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (TestType)t4i(i, j, k, l) * (TestType)t0());
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t0());
            }
          }
        }
      }
    }
    (t4o = t0 * t4i).run(exec);
    exec.sync();

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TestType>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (TestType)t0() * (TestType)t4i(i, j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t0() * t4i(i, j, k, l));
            }
          }
        }
      }
    }

  }
  MATX_EXIT_HANDLER();
}