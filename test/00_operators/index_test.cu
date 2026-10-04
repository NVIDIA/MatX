#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

// index(d) returns the output index along dimension d, for every dimension
TYPED_TEST(OperatorTestsFloatNonComplexAllExecs, Index)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  const cuda::std::array<index_t, 4> shape{2, 3, 4, 5};
  auto out_i = make_tensor<index_t>(shape);
  auto out_t = make_tensor<TestType>(shape);

  // JIT code for CastOp and binary operators cannot wrap rank-less operators such as index()
  constexpr bool check_cast = !is_cuda_jit_executor_v<ExecType>;

  for (int d = 0; d < 4; d++) {
    (out_i = index(d)).run(exec);
    if constexpr (check_cast) {
      (out_t = as_type<TestType>(index(d)) + TestType(1)).run(exec);
    }
    exec.sync();

    for (index_t i0 = 0; i0 < shape[0]; i0++) {
      for (index_t i1 = 0; i1 < shape[1]; i1++) {
        for (index_t i2 = 0; i2 < shape[2]; i2++) {
          for (index_t i3 = 0; i3 < shape[3]; i3++) {
            const index_t idx[4] = {i0, i1, i2, i3};
            ASSERT_EQ(out_i(i0, i1, i2, i3), idx[d]) << "dim=" << d;
            if constexpr (check_cast) {
              ASSERT_EQ(out_t(i0, i1, i2, i3), TestType(static_cast<float>(idx[d] + 1))) << "dim=" << d;
            }
          }
        }
      }
    }
  }

  // Masking with a comparison on index(), as used to zero a lower triangle
  if constexpr (check_cast) {
    auto tri = make_tensor<TestType>({4, 5});
    (tri = TestType(1)).run(exec);
    (IF(index(1) < index(0), tri = TestType(0))).run(exec);
    exec.sync();
    for (index_t i = 0; i < 4; i++) {
      for (index_t j = 0; j < 5; j++) {
        ASSERT_EQ(tri(i, j), TestType(j < i ? 0 : 1));
      }
    }
  }

  MATX_EXIT_HANDLER();
}
