#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

#include <type_traits>
#include <vector>

using namespace matx;
using namespace matx::test;

template <typename TensorType>
class FilterTestsFloatNonComplexNonHalfCUDA : public ::testing::Test {};

TYPED_TEST_SUITE(FilterTestsFloatNonComplexNonHalfCUDA,
                 MatXFloatNonComplexNonHalfTypesCUDAExec);

template <typename TestType>
void RunIIRRegressionCase(cudaExecutor &exec, index_t batches, index_t num_samples)
{
  const cuda::std::array<TestType, 2> h_rec{
      static_cast<TestType>(0.4), static_cast<TestType>(-0.1)};
  const cuda::std::array<TestType, 2> h_nonrec{
      static_cast<TestType>(2.0), static_cast<TestType>(1.0)};

  auto in = make_tensor<TestType>({batches, num_samples});
  auto out = make_tensor<TestType>({batches, num_samples});
  std::vector<double> expected(static_cast<size_t>(batches * num_samples), 0.0);

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < num_samples; i++) {
      const double x = static_cast<double>((((i + 11 * b) % 23) - 11)) / 7.0;
      in(b, i) = static_cast<TestType>(x);

      double y = 0.0;
      for (index_t nr = 0; nr < static_cast<index_t>(h_nonrec.size()); nr++) {
        if (i >= nr) {
          y += static_cast<double>(h_nonrec[static_cast<size_t>(nr)]) *
               static_cast<double>(in(b, i - nr));
        }
      }
      for (index_t r = 0; r < static_cast<index_t>(h_rec.size()); r++) {
        if (i >= (r + 1)) {
          y += static_cast<double>(h_rec[static_cast<size_t>(r)]) *
               expected[static_cast<size_t>(b * num_samples + i - r - 1)];
        }
      }

      expected[static_cast<size_t>(b * num_samples + i)] = y;
    }
  }

  (out = filter(in, h_rec, h_nonrec)).run(exec);
  exec.sync();

  const double tol = std::is_same_v<TestType, float> ? 1.0e-2 : 1.0e-8;
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < num_samples; i++) {
      ASSERT_NEAR(static_cast<double>(out(b, i)),
                  expected[static_cast<size_t>(b * num_samples + i)], tol)
          << "batch=" << b << " sample=" << i;
    }
  }
}

TYPED_TEST(FilterTestsFloatNonComplexNonHalfCUDA, IIRNonMultipleOfChunkSize)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  static_assert(std::is_same_v<ExecType, cudaExecutor>);

  ExecType exec{};
  constexpr index_t kRecursiveChunkSize = 1024 * 8;

  RunIIRRegressionCase<TestType>(exec, 3, kRecursiveChunkSize + 37);

  MATX_EXIT_HANDLER();
}
