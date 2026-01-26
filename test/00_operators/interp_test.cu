#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;


//TYPED_TEST(OperatorTestsFloatNonComplexNonHalfAllExecs, Interp)
TEST(InterpTests, Interp)
{
  MATX_ENTER_HANDLER();
  using TestType = float;
  using ExecType = cudaExecutor;

  if constexpr (!is_cuda_executor_v<ExecType>) {
    GTEST_SKIP();
  }

  using inner_type = typename inner_op_type_t<TestType>::type;
  ExecType exec{};

  // example-begin interp-test-1
  auto x = make_tensor<TestType>({5});
  x.SetVals({0.0, 1.0, 3.0, 3.5, 4.0});

  auto v = make_tensor<TestType>(x.Shape());
  v.SetVals({0.0, 2.0, 1.0, 3.0, 4.0});

  auto xq = make_tensor<TestType>({6});
  xq.SetVals({-1.0, 0.0, 0.25, 1.0, 1.5, 5.0});

  auto vq_linear = make_tensor<TestType>({xq.Size(0)});
  vq_linear.SetVals({0.0, 0.0, 0.5, 2.0, 1.75, 4.0});

  auto vq_nearest = make_tensor<TestType>({xq.Size(0)});
  vq_nearest.SetVals({0.0, 0.0, 0.0, 2.0, 2.0, 4.0});

  auto vq_next = make_tensor<TestType>({xq.Size(0)});
  vq_next.SetVals({0.0, 0.0, 2.0, 2.0, 1.0, 4.0});

  auto vq_prev = make_tensor<TestType>({xq.Size(0)});
  vq_prev.SetVals({0.0, 0.0, 0.0, 2.0, 2.0, 4.0});

  auto vq_spline = make_tensor<TestType>({xq.Size(0)});
  vq_spline.SetVals({-10.7391, 0.0, 1.1121, 2.0, 1.3804, -8.1739});



  auto out_linear = make_tensor<TestType>({xq.Size(0)});
  (out_linear = interp1(x, v, xq, InterpMethod::LINEAR)).run(exec);
  // example-end interp-test-1
  exec.sync();

  for (index_t i = 0; i < xq.Size(0); i++) {
    ASSERT_EQ(out_linear(i), vq_linear(i));
  }

  // example-begin interp-test-2
  auto out_nearest = make_tensor<TestType>({xq.Size(0)});
  (out_nearest = interp1(x, v, xq, InterpMethod::NEAREST)).run(exec);
  // example-end interp-test-2
  exec.sync();

  for (index_t i = 0; i < xq.Size(0); i++) {
    ASSERT_EQ(out_nearest(i), vq_nearest(i));
  }

  auto out_next = make_tensor<TestType>(xq.Shape());
  (out_next = interp1(x, v, xq, InterpMethod::NEXT)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq.Size(0); i++) {
    ASSERT_EQ(out_next(i), vq_next(i));
  }

  auto out_prev = make_tensor<TestType>(xq.Shape());
  (out_prev = interp1(x, v, xq, InterpMethod::PREV)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq.Size(0); i++) {
    ASSERT_EQ(out_prev(i), vq_prev(i));
  }

  auto out_spline = make_tensor<TestType>(xq.Shape());
  (out_spline = interp1(x, v, xq, InterpMethod::SPLINE)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq.Size(0); i++) {
    ASSERT_NEAR(out_spline(i), vq_spline(i), 1e-4);
  }


  auto x2 = make_tensor<TestType>({2, 5});
  auto v3 = make_tensor<TestType>({3, 2, 5});
  auto xq4 = make_tensor<TestType>({4, 3, 2, 6});

  (x2 = x).run(exec);
  (v3 = v).run(exec);
  (xq4 = xq).run(exec);


  auto out_linear4 = make_tensor<TestType>(xq4.Shape());
  (out_linear4 = interp1(x, v3, xq4, InterpMethod::LINEAR)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq4.Size(0); i++) {
    for (index_t j = 0; j < xq4.Size(1); j++) {
      for (index_t k = 0; k < xq4.Size(2); k++) {
        for (index_t l = 0; l < xq4.Size(3); l++) {
          ASSERT_EQ(out_linear4(i, j, k, l), vq_linear(l));
        }
      }
    }
  }
  

  auto out_nearest4 = make_tensor<TestType>(xq4.Shape());
  (out_nearest4 = interp1(x, v3, xq4, InterpMethod::NEAREST)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq4.Size(0); i++) {
    for (index_t j = 0; j < xq4.Size(1); j++) {
      for (index_t k = 0; k < xq4.Size(2); k++) {
        for (index_t l = 0; l < xq4.Size(3); l++) {
          ASSERT_EQ(out_nearest4(i, j, k, l), vq_nearest(l));
        }
      }
    }
  }

  auto out_next4 = make_tensor<TestType>(xq4.Shape());
  (out_next4 = interp1(x, v3, xq4, InterpMethod::NEXT)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq4.Size(0); i++) {
    for (index_t j = 0; j < xq4.Size(1); j++) {
      for (index_t k = 0; k < xq4.Size(2); k++) {
        for (index_t l = 0; l < xq4.Size(3); l++) {
          ASSERT_EQ(out_next4(i, j, k, l), vq_next(l));
        }
      }
    }
  }

  auto out_prev4 = make_tensor<TestType>(xq4.Shape());
  (out_prev4 = interp1(x, v3, xq4, InterpMethod::PREV)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq4.Size(0); i++) {
    for (index_t j = 0; j < xq4.Size(1); j++) {
      for (index_t k = 0; k < xq4.Size(2); k++) {
        for (index_t l = 0; l < xq4.Size(3); l++) {
          ASSERT_EQ(out_prev4(i, j, k, l), vq_prev(l));
        }
      }
    }
  }

  auto out_spline4 = make_tensor<TestType>(xq4.Shape());
  (out_spline4 = interp1(x, v3, xq4, InterpMethod::SPLINE)).run(exec);
  exec.sync();

  for (index_t i = 0; i < xq4.Size(0); i++) {
    for (index_t j = 0; j < xq4.Size(1); j++) {
      for (index_t k = 0; k < xq4.Size(2); k++) {
        for (index_t l = 0; l < xq4.Size(3); l++) {
          ASSERT_NEAR(out_spline4(i, j, k, l), vq_spline(l), 1e-4);
        }
      }
    }
  }


  auto px2 = make_tensor<TestType>({5, 2});
  auto pv3 = make_tensor<TestType>({3, 5, 2});
  auto pxq4 = make_tensor<TestType>({4, 3, 6, 2});

  (px2 = permute(x2, {1, 0})).run(exec);
  (pv3 = permute(v3, {0, 2, 1})).run(exec);
  (pxq4 = permute(xq4, {0, 1, 3, 2})).run(exec);



  auto out_perm_linear4 = make_tensor<TestType>(pxq4.Shape());
  (out_perm_linear4 = interp1(px2, pv3, pxq4, {2}, InterpMethod::LINEAR)).run(exec);
  exec.sync();

  for (index_t i = 0; i < pxq4.Size(0); i++) {
    for (index_t j = 0; j < pxq4.Size(1); j++) {
      for (index_t k = 0; k < pxq4.Size(2); k++) {
        for (index_t l = 0; l < pxq4.Size(3); l++) {
          ASSERT_EQ(out_perm_linear4(i, j, k, l), vq_linear(k));
        }
      }
    }
  }
  

  auto out_perm_nearest4 = make_tensor<TestType>(pxq4.Shape());
  (out_perm_nearest4 = interp1(px2, pv3, pxq4, {2}, InterpMethod::NEAREST)).run(exec);
  exec.sync();

  for (index_t i = 0; i < pxq4.Size(0); i++) {
    for (index_t j = 0; j < pxq4.Size(1); j++) {
      for (index_t k = 0; k < pxq4.Size(2); k++) {
        for (index_t l = 0; l < pxq4.Size(3); l++) {
          ASSERT_EQ(out_perm_nearest4(i, j, k, l), vq_nearest(k));
        }
      }
    }
  }

  auto out_perm_next4 = make_tensor<TestType>(pxq4.Shape());
  (out_perm_next4 = interp1(px2, pv3, pxq4, {2}, InterpMethod::NEXT)).run(exec);
  exec.sync();

  for (index_t i = 0; i < pxq4.Size(0); i++) {
    for (index_t j = 0; j < pxq4.Size(1); j++) {
      for (index_t k = 0; k < pxq4.Size(2); k++) {
        for (index_t l = 0; l < pxq4.Size(3); l++) {
          ASSERT_EQ(out_perm_next4(i, j, k, l), vq_next(k));
        }
      }
    }
  }

  auto out_perm_prev4 = make_tensor<TestType>(pxq4.Shape());
  (out_perm_prev4 = interp1(px2, pv3, pxq4, {2}, InterpMethod::PREV)).run(exec);
  exec.sync();

  for (index_t i = 0; i < pxq4.Size(0); i++) {
    for (index_t j = 0; j < pxq4.Size(1); j++) {
      for (index_t k = 0; k < pxq4.Size(2); k++) {
        for (index_t l = 0; l < pxq4.Size(3); l++) {
          ASSERT_EQ(out_perm_prev4(i, j, k, l), vq_prev(k));
        }
      }
    }
  }

  auto out_perm_spline4 = make_tensor<TestType>(pxq4.Shape());
  (out_perm_spline4 = interp1(px2, pv3, pxq4, {2}, InterpMethod::SPLINE)).run(exec);
  exec.sync();

  for (index_t i = 0; i < pxq4.Size(0); i++) {
    for (index_t j = 0; j < pxq4.Size(1); j++) {
      for (index_t k = 0; k < pxq4.Size(2); k++) {
        for (index_t l = 0; l < pxq4.Size(3); l++) {
          ASSERT_NEAR(out_perm_spline4(i, j, k, l), vq_spline(k), 1e-4);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}
