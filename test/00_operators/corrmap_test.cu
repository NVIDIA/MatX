#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

#include <cmath>
#include <random>
#include <vector>

using namespace matx;
using namespace matx::test;

namespace {

// CPU reference for 2-D corrmap on a single (H, W) image.
//
// Two-pass implementation in all modes: pass 1 collects the in-bounds
// samples (and the means for ZNCC); pass 2 accumulates the products on the
// (already-centered, for ZNCC) sample values. The MAGNITUDE and ZNCC cases
// share the same final normalization step. This avoids the single-pass
// centered identity's cancellation, so the operator's single-pass form is
// being validated against an independent algorithm rather than against
// itself.
template <typename T>
T corrmap_ref_2d(const T *A, const T *B,
                 int H, int W,
                 int row, int col,
                 int win_r, int win_c,
                 CorrMapNormalize mode)
{
  using inner_t = typename inner_op_type_t<T>::type;
  constexpr bool is_cplx = is_complex_v<T>;

  // Pass 1: gather in-bounds samples.
  const int row_start = row - (win_r / 2);
  const int col_start = col - (win_c / 2);
  std::vector<T> a_samples;
  std::vector<T> b_samples;
  a_samples.reserve(static_cast<size_t>(win_r * win_c));
  b_samples.reserve(static_cast<size_t>(win_r * win_c));
  for (int dr = 0; dr < win_r; dr++) {
    const int r = row_start + dr;
    if (r < 0 || r >= H) continue;
    for (int dc = 0; dc < win_c; dc++) {
      const int c = col_start + dc;
      if (c < 0 || c >= W) continue;
      a_samples.push_back(A[r * W + c]);
      b_samples.push_back(B[r * W + c]);
    }
  }

  const int n_valid = static_cast<int>(a_samples.size());
  if (n_valid == 0) return T{};

  // For ZNCC, also compute the window-local means in pass 1.
  T ma{}, mb{};
  if (mode == CorrMapNormalize::ZNCC) {
    for (int i = 0; i < n_valid; i++) {
      ma += a_samples[i];
      mb += b_samples[i];
    }
    const inner_t n = static_cast<inner_t>(n_valid);
    if constexpr (is_cplx) {
      ma = T{ma.real() / n, ma.imag() / n};
      mb = T{mb.real() / n, mb.imag() / n};
    } else {
      ma /= n;
      mb /= n;
    }
  }

  // Pass 2: accumulate on (possibly centered) values.
  T sum_ab{};
  inner_t sum_aa{0};
  inner_t sum_bb{0};
  for (int i = 0; i < n_valid; i++) {
    T a = a_samples[i];
    T b = b_samples[i];
    if (mode == CorrMapNormalize::ZNCC) {
      if constexpr (is_cplx) {
        a = T{a.real() - ma.real(), a.imag() - ma.imag()};
        b = T{b.real() - mb.real(), b.imag() - mb.imag()};
      } else {
        a -= ma;
        b -= mb;
      }
    }
    if constexpr (is_cplx) {
      const inner_t ar = a.real();
      const inner_t ai = a.imag();
      const inner_t br = b.real();
      const inner_t bi = b.imag();
      // a * conj(b)
      sum_ab += T{ar * br + ai * bi, ai * br - ar * bi};
      sum_aa += ar * ar + ai * ai;
      sum_bb += br * br + bi * bi;
    } else {
      sum_ab += a * b;
      sum_aa += a * a;
      sum_bb += b * b;
    }
  }

  if (mode == CorrMapNormalize::NONE) {
    return sum_ab;
  }

  // MAGNITUDE and ZNCC share the final normalize step (ZNCC's centering
  // happened in pass 2; from here their math is identical).
  const inner_t denom = std::sqrt(sum_aa * sum_bb);
  if (denom == inner_t{0}) return T{};
  if constexpr (is_cplx) {
    return T{sum_ab.real() / denom, sum_ab.imag() / denom};
  } else {
    return sum_ab / denom;
  }
}

// CPU reference for 1-D corrmap on a single length-N signal.
template <typename T>
T corrmap_ref_1d(const T *A, const T *B,
                 int N,
                 int idx,
                 int win,
                 CorrMapNormalize mode)
{
  // Treat as 2-D with H=1, W=N, win_r=1.
  return corrmap_ref_2d<T>(A, B, /*H=*/1, /*W=*/N, /*row=*/0, /*col=*/idx,
                            /*win_r=*/1, /*win_c=*/win, mode);
}

// Fill a tensor of any rank with deterministic uniform random samples in
// [-1, 1) using std::mt19937 on the host. We use mt19937 rather than MatX's
// random() operator because the current MatX host executor generates one
// element at a time via curandGenerateUniform host calls, which makes
// the tests take longer to run.
//
// The tensor is treated as a contiguous row-major linear buffer (the
// default layout produced by make_tensor) — we populate flat once and
// std::copy it into tensor.Data().
template <typename T, typename Tensor>
void fill_random(Tensor &tensor, std::vector<T> &flat, uint64_t seed)
{
  using inner_t = typename inner_op_type_t<T>::type;
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  const index_t total = tensor.TotalSize();
  flat.resize(static_cast<size_t>(total));

  for (index_t i = 0; i < total; i++) {
    if constexpr (is_complex_v<T>) {
      flat[static_cast<size_t>(i)] =
        T{static_cast<inner_t>(dist(rng)), static_cast<inner_t>(dist(rng))};
    } else {
      flat[static_cast<size_t>(i)] = static_cast<T>(dist(rng));
    }
  }

  std::copy(flat.begin(), flat.end(), tensor.Data());
}

// Magnitude of a scalar difference. We compute through double to avoid
// cuda::std::abs ambiguity for half complex (matxFp16Complex / matxBf16Complex
// where multiple abs overloads are visible).
template <typename T>
double abs_diff(T x, T y)
{
  const T d = x - y;
  if constexpr (is_complex_v<T>) {
    const double r = static_cast<double>(d.real());
    const double i = static_cast<double>(d.imag());
    return std::sqrt(r * r + i * i);
  } else {
    return std::abs(static_cast<double>(d));
  }
}

// Helper: invoke a generic lambda once per normalization mode (compile-time).
template <typename F>
void for_each_mode(F &&fn) {
  fn.template operator()<CorrMapNormalize::NONE>();
  fn.template operator()<CorrMapNormalize::MAGNITUDE>();
  fn.template operator()<CorrMapNormalize::ZNCC>();
}

// Per-precision tolerances. The operator uses Welford's online algorithm
// for ZNCC (stable single-pass) and simple accumulators for NONE/MAGNITUDE
// (no cancellation risk). The constants returned by the helpers below are
// empirical upper bounds: each one is set just above the worst observed
// error across all tests at that precision. As a hard floor, every
// constant stays above 2*ulp(1.0) so we never assert below representable
// resolution:
//
//   precision  ulp(1.0)     2*ulp(1.0)   (lower bound only)
//   bf16       2^-7  ~ 7.8e-3  ~ 1.56e-2
//   fp16       2^-10 ~ 9.8e-4  ~ 1.95e-3
//   float      2^-23 ~ 1.2e-7  ~ 2.38e-7
//   double     2^-52 ~ 2.2e-16 ~ 4.44e-16
//
// For NONE mode the expected value can grow with window size (sum of N
// products) so ulp(expected) >= ulp(N). The matches-reference tolerances
// include enough headroom to cover this, which is why the bf16/fp16 NONE
// constants sit well above the 2*ulp(1.0) floor.

// Tolerance for matches-reference style tests (NONE/MAGNITUDE/ZNCC against
// an independent two-pass reference).
template <typename InnerT>
constexpr double tolerance_for()
{
  if constexpr (std::is_same_v<InnerT, matxBf16>) {
    return 3.0e-2;
  } else if constexpr (std::is_same_v<InnerT, matxFp16>) {
    return 5.0e-3;
  } else if constexpr (std::is_same_v<InnerT, float>) {
    return 5.0e-5;
  } else {
    return 1.0e-12; // double
  }
}

// Tolerance for MAGNITUDE-only tests (self-coherence, batched 1-D
// MAGNITUDE). Closer to the floor because there's no centering step.
template <typename InnerT>
constexpr double tolerance_magnitude_for()
{
  if constexpr (std::is_same_v<InnerT, matxBf16>) {
    return 2.0e-2;
  } else if constexpr (std::is_same_v<InnerT, matxFp16>) {
    return 2.0e-3;
  } else if constexpr (std::is_same_v<InnerT, float>) {
    return 1.0e-6;
  } else {
    return 1.0e-14; // double
  }
}

// Tolerance for the ZNCC affine-invariance test (B = alpha*A + beta).
template <typename InnerT>
constexpr double tolerance_zncc_affine_for()
{
  if constexpr (std::is_same_v<InnerT, matxBf16>) {
    return 5.0e-2;
  } else if constexpr (std::is_same_v<InnerT, matxFp16>) {
    return 5.0e-3;
  } else if constexpr (std::is_same_v<InnerT, float>) {
    return 5.0e-6;
  } else {
    return 1.0e-12; // double
  }
}

} // namespace

// 2-D corrmap matches CPU reference across odd/even windows and all modes,
// on float, double, complex<float>, complex<double>.
TYPED_TEST(OperatorTestsFloatAllExecs, CorrMap2DMatchesReference)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  ExecType exec{};

  constexpr int H = 12;
  constexpr int W = 14;
  // {1,1} is a degenerate case: NONE is pointwise A*conj(B); MAGNITUDE
  // produces unit-magnitude (phase-only) values; ZNCC is identically 0
  // (centered-value sums vanish for a single sample).
  const std::vector<std::pair<int,int>> windows = {{1,1}, {3,3}, {5,5}, {4,6}, {7,3}};

  auto A = make_tensor<TestType>({H, W});
  auto B = make_tensor<TestType>({H, W});
  auto Y = make_tensor<TestType>({H, W});

  std::vector<TestType> flat_a, flat_b;
  fill_random(A, flat_a, /*seed=*/0x5eed);
  fill_random(B, flat_b, /*seed=*/0x5eed + 1);

  const double tol = tolerance_for<typename inner_op_type_t<TestType>::type>();

  for (auto [wr, wc] : windows) {
    for_each_mode([&]<CorrMapNormalize Mode>() {
      (Y = corrmap<Mode>(A, B, cuda::std::array<index_t, 2>{wr, wc})).run(exec);
      exec.sync();

      for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
          TestType expected = corrmap_ref_2d<TestType>(
              flat_a.data(), flat_b.data(), H, W, r, c, wr, wc, Mode);
          ASSERT_LT(abs_diff(Y(r, c), expected), tol)
              << "mismatch at (" << r << "," << c << ") win=" << wr << "x" << wc
              << " mode=" << static_cast<int>(Mode);
        }
      }
    });
  }

  MATX_EXIT_HANDLER();
}

// 1-D corrmap matches CPU reference. Exercises the scalar-window overload.
TYPED_TEST(OperatorTestsFloatAllExecs, CorrMap1DMatchesReference)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_t = typename inner_op_type_t<TestType>::type;
  ExecType exec{};

  constexpr int N = 32;
  // w=1 is a degenerate case: NONE is pointwise A*conj(B); MAGNITUDE
  // produces unit-magnitude (phase-only) values; ZNCC is identically 0
  // (centered-value sums vanish for a single sample).
  const std::vector<int> windows = {1, 3, 5, 4, 8};

  auto A = make_tensor<TestType>({N});
  auto B = make_tensor<TestType>({N});
  auto Y = make_tensor<TestType>({N});

  std::vector<TestType> flat_a, flat_b;
  fill_random(A, flat_a, /*seed=*/0xFEEDFACE);
  fill_random(B, flat_b, /*seed=*/0xFEEDFACE + 1);

  const double tol = tolerance_for<inner_t>();
  for (auto w : windows) {
    for_each_mode([&]<CorrMapNormalize Mode>() {
      (Y = corrmap<Mode>(A, B, static_cast<index_t>(w))).run(exec);
      exec.sync();
      for (int i = 0; i < N; i++) {
        TestType expected = corrmap_ref_1d<TestType>(
            flat_a.data(), flat_b.data(), N, i, w, Mode);
        ASSERT_LT(abs_diff(Y(i), expected), tol)
            << "1d mismatch at " << i << " w=" << w
            << " mode=" << static_cast<int>(Mode);
      }
    });
  }

  MATX_EXIT_HANDLER();
}

// Self-coherence: A == B with MAGNITUDE mode → |result| == 1 everywhere the
// window has any in-bounds samples. Complex self-coherence has zero phase.
TYPED_TEST(OperatorTestsFloatAllExecs, CorrMapSelfCoherenceIsOne)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_t = typename inner_op_type_t<TestType>::type;
  ExecType exec{};

  constexpr int H = 8;
  constexpr int W = 8;
  constexpr int win = 3;

  auto A    = make_tensor<TestType>({H, W});
  auto Y    = make_tensor<TestType>({H, W});
  auto Ymag = make_tensor<inner_t>({H, W});

  std::vector<TestType> flat_a;
  fill_random(A, flat_a, /*seed=*/0xC0FFEE);

  // example-begin corrmap-2d-magnitude
  // 2-D windowed coherence between two complex images (here A==A for the
  // self-coherence sanity check). |Y(r,c)| is the SAR/InSAR coherence map.
  (Y = corrmap<CorrMapNormalize::MAGNITUDE>(
        A, A, cuda::std::array<index_t, 2>{win, win})).run(exec);
  // example-end corrmap-2d-magnitude
  // Use MatX's abs() to extract the coherence magnitude as a real tensor.
  (Ymag = abs(Y)).run(exec);
  exec.sync();

  const double tol = tolerance_magnitude_for<inner_t>();
  for (int r = 0; r < H; r++) {
    for (int c = 0; c < W; c++) {
      ASSERT_NEAR(static_cast<double>(Ymag(r, c)), 1.0, tol)
          << "self-coherence != 1 at (" << r << "," << c << ")";
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(static_cast<double>(Y(r, c).imag()), 0.0, tol)
            << "self-coherence phase != 0 at (" << r << "," << c << ")";
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// ZNCC is invariant to positive affine transforms of either input.
TYPED_TEST(OperatorTestsFloatAllExecs, CorrMapZnccAffineInvariant)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_t  = typename inner_op_type_t<TestType>::type;
  ExecType exec{};

  constexpr int H = 10;
  constexpr int W = 10;
  constexpr int win = 5;
  // Real-valued alpha > 0 keeps the expected ZNCC at exactly 1 (+0i for
  // complex inputs).
  constexpr TestType alpha = static_cast<TestType>(3);
  constexpr TestType beta  = static_cast<TestType>(7);
  const TestType expected  = static_cast<TestType>(1);

  auto A = make_tensor<TestType>({H, W});
  auto B = make_tensor<TestType>({H, W});
  auto Y = make_tensor<TestType>({H, W});

  std::vector<TestType> flat_a;
  fill_random(A, flat_a, /*seed=*/0x1234);
  // Compute B = alpha*A + beta elementwise on host. We avoid the tensor
  // expression form because we currently hit a template deduction issue
  // with matxHalfComplex's unconstrained scalar operator* templates.
  for (int r = 0; r < H; r++) {
    for (int c = 0; c < W; c++) {
      B(r, c) = A(r, c) * alpha + beta;
    }
  }

  // example-begin corrmap-2d-zncc
  // Per-pixel ZNCC (mean-subtracted normalized cross-correlation): classic
  // NCC for image processing, pattern matching, etc. Real result in [-1, 1].
  (Y = corrmap<CorrMapNormalize::ZNCC>(
        A, B, cuda::std::array<index_t, 2>{win, win})).run(exec);
  // example-end corrmap-2d-zncc
  exec.sync();

  const double tol = tolerance_zncc_affine_for<inner_t>();
  // Interior pixels see the full window — ZNCC must be 1 within tol.
  const int border = win / 2;
  for (int r = border; r < H - border; r++) {
    for (int c = border; c < W - border; c++) {
      ASSERT_LT(abs_diff(Y(r, c), expected), tol)
          << "zncc != 1 at (" << r << "," << c << ")";
    }
  }

  MATX_EXIT_HANDLER();
}

// Batched rank-3 inputs with 2-D window: leading dim is batched independently.
TYPED_TEST(OperatorTestsFloatAllExecs, CorrMapBatched2D)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_t  = typename inner_op_type_t<TestType>::type;
  ExecType exec{};

  constexpr int B_ = 3;
  constexpr int H = 7;
  constexpr int W = 9;
  constexpr int win = 3;

  auto A = make_tensor<TestType>({B_, H, W});
  auto B = make_tensor<TestType>({B_, H, W});
  auto Y = make_tensor<TestType>({B_, H, W});

  std::vector<TestType> flat_a, flat_b;
  fill_random(A, flat_a, /*seed=*/0xABCD);
  fill_random(B, flat_b, /*seed=*/0xABCD + 1);

  (Y = corrmap<CorrMapNormalize::ZNCC>(
        A, B, cuda::std::array<index_t, 2>{win, win})).run(exec);
  exec.sync();

  const double tol = tolerance_for<inner_t>();
  for (int b = 0; b < B_; b++) {
    for (int r = 0; r < H; r++) {
      for (int c = 0; c < W; c++) {
        const size_t batch_offset = static_cast<size_t>(b) * H * W;
        TestType expected = corrmap_ref_2d<TestType>(
            flat_a.data() + batch_offset, flat_b.data() + batch_offset,
            H, W, r, c, win, win, CorrMapNormalize::ZNCC);
        ASSERT_LT(abs_diff(Y(b, r, c), expected), tol)
            << "batched mismatch at b=" << b << " (" << r << "," << c << ")";
      }
    }
  }

  MATX_EXIT_HANDLER();
}

// Executor-only fixture: TypeParam is just the executor type. The test
// below hard-codes its own input type pairs, so there is no data-type
// axis to parameterize over.
template <typename ExecType>
class CorrMapExecutorOnly : public ::testing::Test {};

using CorrMapExecutorTypes = TupleToTypes<ExecutorTypesAll>::type;
TYPED_TEST_SUITE(CorrMapExecutorOnly, CorrMapExecutorTypes);

// Mixed input types: output type and value follow common_type semantics.
//   float    + double           -> double
//   complex<float> + complex<double> -> complex<double>
//   float    + complex<double>  -> complex<double>
//   complex<float> + double     -> complex<double>
TYPED_TEST(CorrMapExecutorOnly, MixedTypes)
{
  MATX_ENTER_HANDLER();
  using ExecType = TypeParam;
  ExecType exec{};

  constexpr int N = 8;

  // Case 1: float + double -> double
  {
    auto A = make_tensor<float>({N});
    auto B = make_tensor<double>({N});
    for (int i = 0; i < N; i++) {
      A(i) = static_cast<float>(0.5 * i);
      B(i) = static_cast<double>(0.25 * i + 0.1);
    }
    auto expr = corrmap<CorrMapNormalize::ZNCC>(A, B, static_cast<index_t>(3));
    using ExprT = decltype(expr);
    using OutT = typename ExprT::value_type;
    static_assert(cuda::std::is_same_v<OutT, double>,
                  "float + double should yield double");

    auto Y = make_tensor<OutT>({N});
    (Y = expr).run(exec);
    exec.sync();
    // Sanity: result is finite and in [-1, 1] for ZNCC.
    for (int i = 0; i < N; i++) {
      ASSERT_GE(Y(i), -1.0 - 1e-10);
      ASSERT_LE(Y(i),  1.0 + 1e-10);
    }
  }

  // Case 2: complex<float> + complex<double> -> complex<double>
  {
    auto A = make_tensor<cuda::std::complex<float>>({N});
    auto B = make_tensor<cuda::std::complex<double>>({N});
    for (int i = 0; i < N; i++) {
      A(i) = {static_cast<float>(0.5 * i), static_cast<float>(0.3)};
      B(i) = {0.25 * i + 0.1, -0.2};
    }
    auto expr = corrmap<CorrMapNormalize::MAGNITUDE>(A, B, static_cast<index_t>(3));
    using ExprT = decltype(expr);
    using OutT = typename ExprT::value_type;
    static_assert(cuda::std::is_same_v<OutT, cuda::std::complex<double>>,
                  "complex<float> + complex<double> should yield complex<double>");

    auto Y = make_tensor<OutT>({N});
    (Y = expr).run(exec);
    exec.sync();
    for (int i = 0; i < N; i++) {
      const double mag = std::sqrt(Y(i).real() * Y(i).real() +
                                   Y(i).imag() * Y(i).imag());
      ASSERT_LE(mag, 1.0 + 1e-12);
    }
  }

  // Case 3: float + complex<double> -> complex<double>
  {
    auto A = make_tensor<float>({N});
    auto B = make_tensor<cuda::std::complex<double>>({N});
    for (int i = 0; i < N; i++) {
      A(i) = static_cast<float>(0.5 * i);
      B(i) = {0.25 * i + 0.1, -0.2};
    }
    auto expr = corrmap<CorrMapNormalize::MAGNITUDE>(A, B, static_cast<index_t>(3));
    using ExprT = decltype(expr);
    using OutT = typename ExprT::value_type;
    static_assert(cuda::std::is_same_v<OutT, cuda::std::complex<double>>,
                  "float + complex<double> should yield complex<double>");

    auto Y = make_tensor<OutT>({N});
    (Y = expr).run(exec);
    exec.sync();
    for (int i = 0; i < N; i++) {
      const double mag = std::sqrt(Y(i).real() * Y(i).real() +
                                   Y(i).imag() * Y(i).imag());
      ASSERT_LE(mag, 1.0 + 1e-12);
    }
  }

  // Case 4: complex<float> + double -> complex<double>
  // (the complex-ness and the precision are decided independently)
  {
    auto A = make_tensor<cuda::std::complex<float>>({N});
    auto B = make_tensor<double>({N});
    for (int i = 0; i < N; i++) {
      A(i) = {static_cast<float>(0.5 * i), static_cast<float>(0.3)};
      B(i) = 0.25 * i + 0.1;
    }
    auto expr = corrmap<CorrMapNormalize::MAGNITUDE>(A, B, static_cast<index_t>(3));
    using ExprT = decltype(expr);
    using OutT = typename ExprT::value_type;
    static_assert(cuda::std::is_same_v<OutT, cuda::std::complex<double>>,
                  "complex<float> + double should yield complex<double>");

    auto Y = make_tensor<OutT>({N});
    (Y = expr).run(exec);
    exec.sync();
    for (int i = 0; i < N; i++) {
      const double mag = std::sqrt(Y(i).real() * Y(i).real() +
                                   Y(i).imag() * Y(i).imag());
      ASSERT_LE(mag, 1.0 + 1e-12);
    }
  }

  MATX_EXIT_HANDLER();
}

// Batched rank-2 inputs with 1-D window: each row is an independent signal.
TYPED_TEST(OperatorTestsFloatAllExecs, CorrMapBatched1D)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  using inner_t  = typename inner_op_type_t<TestType>::type;
  ExecType exec{};

  constexpr int B_ = 4;
  constexpr int N = 24;
  constexpr int win = 5;

  auto A = make_tensor<TestType>({B_, N});
  auto B = make_tensor<TestType>({B_, N});
  auto Y = make_tensor<TestType>({B_, N});

  std::vector<TestType> flat_a, flat_b;
  fill_random(A, flat_a, /*seed=*/0x9999);
  fill_random(B, flat_b, /*seed=*/0x9999 + 1);

  // example-begin corrmap-1d-magnitude
  // 1-D sliding-window correlation on a batch of length-N signals. The
  // scalar window argument selects the 1-D overload; the leading dim is
  // batched independently.
  (Y = corrmap<CorrMapNormalize::MAGNITUDE>(A, B, win)).run(exec);
  // example-end corrmap-1d-magnitude
  exec.sync();

  const double tol = tolerance_magnitude_for<inner_t>();
  for (int b = 0; b < B_; b++) {
    for (int i = 0; i < N; i++) {
      const size_t batch_offset = static_cast<size_t>(b) * N;
      TestType expected = corrmap_ref_1d<TestType>(
          flat_a.data() + batch_offset, flat_b.data() + batch_offset,
          N, i, win, CorrMapNormalize::MAGNITUDE);
      ASSERT_LT(abs_diff(Y(b, i), expected), tol)
          << "batched 1d mismatch at b=" << b << " i=" << i;
    }
  }

  MATX_EXIT_HANDLER();
}
