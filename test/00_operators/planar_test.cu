#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include <vector>

using namespace matx;
using namespace matx::test;


TYPED_TEST(OperatorTestsComplexTypesAllExecs, PlanarTransform)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};   
  index_t m = 10;
  index_t k = 20;
  tensor_t<TestType, 2> t2({m, k});
  tensor_t<typename TestType::value_type, 2> t2p({m * 2, k});
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      TestType tmp = {(float)i, (float)-j};
      t2(i, j) = tmp;
    }
  }

  (t2p = planar(t2)).run(exec);
  exec.sync();

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j).real(), t2p(i, j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2(i, j).imag(), t2p(i + t2.Size(0), j))) << i << " " << j << "\n";
    }
  }
  MATX_EXIT_HANDLER();
}

namespace {
template <typename PlanarType>
void ValidatePlanarTensorOperatorLayout()
{
  constexpr index_t m = 3;
  constexpr index_t k = 5;
  constexpr index_t mk = m * k;
  using ScalarType = typename PlanarType::value_type;

  tensor_t<PlanarType, 2> t{{m, k}};
  std::vector<ScalarType> raw(static_cast<size_t>(mk * 2));

  for (index_t i = 0; i < mk; i++) {
    raw[static_cast<size_t>(i)] = ScalarType{static_cast<float>(i + 1)};
    raw[static_cast<size_t>(i + mk)] = ScalarType{static_cast<float>(-(i + 1))};
  }

  ASSERT_EQ(cudaMemcpy(t.Data(), raw.data(), raw.size() * sizeof(ScalarType),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      const index_t idx = i * k + j;
      const auto v = t(i, j);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(v.real(), raw[static_cast<size_t>(idx)]));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(v.imag(), raw[static_cast<size_t>(idx + mk)]));
    }
  }
}
} // namespace

TEST(PlanarHalfComplexTypes, TensorOperatorUsesPlanarStorageForFp16AndBf16)
{
  MATX_ENTER_HANDLER();
  ValidatePlanarTensorOperatorLayout<matxFp16ComplexPlanar>();
  ValidatePlanarTensorOperatorLayout<matxBf16ComplexPlanar>();
  MATX_EXIT_HANDLER();
}
