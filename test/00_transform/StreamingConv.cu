////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

// Test for the Conv1DStream object (matx/streaming/conv1d_stream.h). Streaming
// a signal in chunks (feed() ... flush()) and concatenating the produced outputs
// must equal a single one-shot conv1d over the whole input in the configured
// mode (FULL / SAME / VALID). Covers odd and even filter lengths (the SAME
// start offset differs by parity), the degenerate L==1 filter, chunk > N, and
// total streams shorter than the SAME/VALID startup skip.

#include "matx.h"
#include "matx/streaming/conv1d_stream.h"

#include "gtest/gtest.h"
#include "prerun_tester.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace matx;
using namespace matx::test;

namespace {

index_t one_shot_len(index_t N, index_t L, matxConvCorrMode_t mode)
{
  switch (mode) {
    case MATX_C_MODE_FULL: return N + L - 1;
    case MATX_C_MODE_SAME: return std::max(N, L); // conv1d SAME: size of larger input
    default:               return std::max(N, L) - std::min(N, L) + 1; // VALID
  }
}

// Streams the signal using the cyclic chunk-size schedule `chunks` (a
// single-entry schedule is a fixed chunk size).
template <typename Exec>
bool run_case(Exec &exec, index_t N, index_t L, matxConvCorrMode_t mode,
              const std::vector<index_t> &chunks)
{
  using T = float;
  std::string sched;
  for (auto c : chunks) { sched += std::to_string(c) + ","; }
  auto x = make_tensor<T>({N});
  auto h = make_tensor<T>({L});
  for (index_t i = 0; i < N; i++) {
    x(i) = std::sin(0.11f * static_cast<float>(i)) + 0.003f * static_cast<float>(i);
  }
  for (index_t k = 0; k < L; k++) {
    h(k) = std::cos(0.2f * static_cast<float>(k)) / static_cast<float>(L);
  }

  // Reference. For N >= L this is the one-shot conv1d in `mode` directly. For
  // N < L the streaming contract is the input-aligned result (roles fixed:
  // SAME = FULL[s, s+N)), whereas the one-shot swaps operand roles and returns
  // a different slice of the same full convolution -- so validate the streaming
  // contract against that slice of the (role-symmetric) FULL result.
  index_t M = one_shot_len(N, L, mode);
  auto ref = make_tensor<T>({std::max(M, N + L - 1)});
  index_t ref_off = 0;
  if (mode == MATX_C_MODE_SAME && N < L) {
    M = N;
    ref_off = (L % 2 == 1) ? (L - 1) / 2 : L / 2 - 1; // SAME start within FULL
    (slice(ref, {0}, {N + L - 1}) = conv1d(x, h, MATX_C_MODE_FULL)).run(exec);
  } else {
    (slice(ref, {0}, {M}) = conv1d(x, h, mode)).run(exec);
  }

  const index_t max_chunk = *std::max_element(chunks.begin(), chunks.end());
  auto stream_obj = make_conv1d_stream<T>(h, {.mode = mode}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(max_chunk)});
  auto acc = make_tensor<T>({M});

  index_t off = 0;
  size_t ci = 0;
  for (index_t g = 0; g < N; ) {
    const index_t nl = std::min(chunks[ci++ % chunks.size()], N - g);
    auto in_chunk = slice(x, {g}, {g + nl});
    g += nl;
    const index_t cnt = stream_obj.feed(in_chunk, frame);
    if (cnt > 0) {
      (slice(acc, {off}, {off + cnt}) = slice(frame, {0}, {cnt})).run(exec);
    }
    off += cnt;
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) {
    (slice(acc, {off}, {off + tcnt}) = slice(frame, {0}, {tcnt})).run(exec);
  }
  off += tcnt;
  exec.sync();

  EXPECT_EQ(off, M) << "N=" << N << " L=" << L << " mode=" << mode
                    << " chunks=" << sched;
  if (off != M) return false;

  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < M; i++) {
    max_abs = std::max(max_abs, std::fabs(ref(ref_off + i)));
    max_err = std::max(max_err, std::fabs(acc(i) - ref(ref_off + i)));
  }
  const bool ok = max_err < 1e-4f * (1.0f + max_abs);
  EXPECT_TRUE(ok) << "N=" << N << " L=" << L << " mode=" << mode
                  << " chunks=" << sched << " max_err=" << max_err
                  << " max_abs=" << max_abs;
  return ok;
}

// Fixed-chunk convenience wrapper.
template <typename Exec>
bool run_case(Exec &exec, index_t N, index_t L, matxConvCorrMode_t mode,
              index_t chunk)
{
  return run_case(exec, N, L, mode, std::vector<index_t>{chunk});
}

} // namespace

TEST(StreamingConv, MatchesOneShotAllModes)
{
  cudaExecutor exec{};
  const index_t N = 4096;
  for (matxConvCorrMode_t mode : {MATX_C_MODE_FULL, MATX_C_MODE_SAME,
                                  MATX_C_MODE_VALID}) {
    for (index_t L : {index_t(33), index_t(32), index_t(1)}) { // odd, even, degenerate
      for (index_t chunk : {index_t(1), index_t(7), index_t(64), index_t(500),
                            index_t(1000), index_t(5000)}) { // incl. chunk > N
        run_case(exec, N, L, mode, chunk);
      }
    }
  }
}

// Total stream shorter than (or comparable to) the SAME startup skip: the
// leading skip must carry over into flush() so the emitted count and values
// still match the input-aligned contract. (One-shot VALID for N < L swaps the
// operand roles and emits L-N+1 outputs -- a different role-dependent slice of
// the same commutative full convolution; the streaming VALID contract emits
// max(0, N-L+1) == 0 outputs, verified separately below.)
TEST(StreamingConv, ShortStreamEdges)
{
  cudaExecutor exec{};
  for (matxConvCorrMode_t mode : {MATX_C_MODE_FULL, MATX_C_MODE_SAME}) {
    for (index_t N : {index_t(1), index_t(2), index_t(5), index_t(16)}) {
      for (index_t L : {index_t(33), index_t(32), index_t(5)}) {
        for (index_t chunk : {index_t(1), index_t(3), index_t(16)}) {
          run_case(exec, N, L, mode, chunk);
        }
      }
    }
  }
}

// Streaming VALID on a stream shorter than the filter: no output window is
// ever fully immersed, so feed() and flush() must emit exactly zero outputs.
TEST(StreamingConv, ShortStreamValidEmitsNothing)
{
  cudaExecutor exec{};
  using T = float;
  for (index_t N : {index_t(1), index_t(5), index_t(31)}) {
    const index_t L = 32; // N < L for all cases above
    auto h = make_tensor<T>({L});
    for (index_t k = 0; k < L; k++) {
      h(k) = 1.0f / static_cast<float>(k + 1);
    }
    auto x = make_tensor<T>({N});
    for (index_t i = 0; i < N; i++) {
      x(i) = static_cast<float>(i + 1);
    }
    auto stream_obj = make_conv1d_stream<T>(h, {.mode = MATX_C_MODE_VALID}, exec);
    auto frame = make_tensor<T>({stream_obj.max_output(N)});
    const index_t produced = stream_obj.feed(x, frame);
    const index_t tail = stream_obj.flush(frame);
    exec.sync();
    EXPECT_EQ(produced, 0) << "N=" << N;
    EXPECT_EQ(tail, 0) << "N=" << N;
  }
}

// Chunk sizes that vary across feeds: the retain/skip bookkeeping must be
// correct for every transition between chunk sizes, not just for a repeated
// fixed size. Schedules mix tiny/large chunks and walk through consecutive
// sizes; all must reproduce the one-shot exactly.
TEST(StreamingConv, VaryingChunkSchedule)
{
  cudaExecutor exec{};
  const index_t N = 4096;
  const std::vector<std::vector<index_t>> schedules = {
      {1, 7, 64, 3, 1000, 2, 33},
      {1, 2, 3, 4, 5, 6, 7},
      {500, 1, 500, 2},
  };
  for (matxConvCorrMode_t mode : {MATX_C_MODE_FULL, MATX_C_MODE_SAME,
                                  MATX_C_MODE_VALID}) {
    for (index_t L : {index_t(33), index_t(32)}) {
      for (const auto &sched : schedules) {
        run_case(exec, N, L, mode, sched);
      }
    }
  }
}

// The object always uses direct (time-domain) convolution, whose kernel
// supports at most 1024 filter taps. The constructor must reject longer
// filters in all builds (the operator-level guard is debug-only).
TEST(StreamingConv, RejectsFilterBeyondDirectLimit)
{
  cudaExecutor exec{};
  using T = float;
  const index_t Lmax = detail::CONV1D_MAX_MIN_DIMENSION_DIRECT;
  auto h_ok = make_tensor<T>({Lmax});
  auto h_long = make_tensor<T>({Lmax + 1});
  (h_ok = zeros<T>({Lmax})).run(exec);
  (h_long = zeros<T>({Lmax + 1})).run(exec);
  exec.sync();
  EXPECT_NO_THROW(make_conv1d_stream<T>(h_ok, {.mode = MATX_C_MODE_SAME}, exec));
  EXPECT_THROW(make_conv1d_stream<T>(h_long, {.mode = MATX_C_MODE_SAME}, exec),
               detail::matxException);
}

// A flush() that throws (undersized output buffer) must not consume the
// stream: retrying with a large-enough buffer emits the full tail.
TEST(StreamingConv, FlushRetryAfterFailure)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 100, L = 33;
  auto h = make_tensor<T>({L});
  auto x = make_tensor<T>({N});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  for (index_t i = 0; i < N; i++) { x(i) = static_cast<float>(i + 1); }
  auto stream_obj = make_conv1d_stream<T>(h, {.mode = MATX_C_MODE_FULL}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(N)});
  auto small = make_tensor<T>({L - 2}); // smaller than the L-1 flush outputs
  EXPECT_EQ(stream_obj.feed(x, frame), N);
  EXPECT_THROW(stream_obj.flush(small), detail::matxException);
  // The failed flush did not end the stream; a retry emits the full tail.
  EXPECT_EQ(stream_obj.flush(frame), L - 1);
  EXPECT_EQ(stream_obj.flush(frame), 0);
  exec.sync();
}

// flush() ends the stream: a second flush() emits nothing, feed() throws
// until reset(), and reset() restores the object for a fresh stream that
// reproduces the same counts.
TEST(StreamingConv, FlushEndsStream)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 100, L = 33;
  auto h = make_tensor<T>({L});
  auto x = make_tensor<T>({N});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  for (index_t i = 0; i < N; i++) { x(i) = static_cast<float>(i + 1); }
  auto stream_obj = make_conv1d_stream<T>(h, {.mode = MATX_C_MODE_FULL}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(N)});
  EXPECT_EQ(stream_obj.feed(x, frame), N);
  EXPECT_EQ(stream_obj.flush(frame), L - 1);
  EXPECT_EQ(stream_obj.flush(frame), 0);
  EXPECT_THROW(stream_obj.feed(x, frame), detail::matxException);
  stream_obj.reset();
  EXPECT_EQ(stream_obj.feed(x, frame), N);
  EXPECT_EQ(stream_obj.flush(frame), L - 1);
  exec.sync();
}

// A transform-valued filter must be materialized once at construction, not
// indexed as an unmaterialized temporary, and the transform's arithmetic must
// land in the taps. conv1d(h, 2*ones, SAME) is a nested transform op equal to
// 2h, so the streamed result must be twice the one-shot over plain h.
TEST(StreamingConv, TransformValuedFilterMaterializes)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 200, L = 17, chunk = 32;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) {
    h(k) = std::cos(0.2f * static_cast<float>(k)) / static_cast<float>(L);
  }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) {
    sig(i) = std::sin(0.11f * static_cast<float>(i)) + 0.003f * static_cast<float>(i);
  }
  auto ref = make_tensor<T>({N});
  (ref = conv1d(sig, h, MATX_C_MODE_SAME)).run(exec);

  auto filt_op = conv1d(h, 2.0f * ones<T>({1}), MATX_C_MODE_SAME); // nested transform == 2h
  auto stream_obj = make_conv1d_stream<T>(filt_op, {.mode = MATX_C_MODE_SAME}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(chunk)});
  auto acc = make_tensor<T>({N});
  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    const index_t cnt = stream_obj.feed(slice(sig, {g}, {g + nl}), frame);
    if (cnt > 0) { (slice(acc, {off}, {off + cnt}) = slice(frame, {0}, {cnt})).run(exec); }
    off += cnt;
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off}, {off + tcnt}) = slice(frame, {0}, {tcnt})).run(exec); }
  off += tcnt;
  exec.sync();
  ASSERT_EQ(off, N);
  // Expect twice the plain-h one-shot (the filter is 2h).
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < N; i++) {
    const float expected = 2.0f * ref(i);
    max_abs = std::max(max_abs, std::fabs(expected));
    max_err = std::max(max_err, std::fabs(acc(i) - expected));
  }
  EXPECT_LT(max_err, 1e-4f * (1.0f + max_abs)) << "max_err=" << max_err;
}

// Stream `sig` (length N) through a fresh SAME-mode Conv1DStream, building each
// feed's segment from build_seg(chunk_slice, nl). Returns the reassembled
// length-N output. Lets a test vary the segment's operator form (bare tensor,
// transform, or a wrapper around one) without duplicating the streaming loop.
template <typename SegBuilder>
matx::tensor_t<float, 1> stream_conv_same(cudaExecutor &exec,
    const matx::tensor_t<float, 1> &sig, const matx::tensor_t<float, 1> &h,
    index_t N, index_t chunk, SegBuilder build_seg)
{
  auto stream_obj = make_conv1d_stream<float>(h, {.mode = MATX_C_MODE_SAME}, exec);
  auto frame = make_tensor<float>({stream_obj.max_output(chunk)});
  auto acc = make_tensor<float>({N});
  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    auto seg = build_seg(slice(sig, {g}, {g + nl}), nl);
    const index_t cnt = stream_obj.feed(seg, frame);
    if (cnt > 0) { (slice(acc, {off}, {off + cnt}) = slice(frame, {0}, {cnt})).run(exec); }
    off += cnt;
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off}, {off + tcnt}) = slice(frame, {0}, {tcnt})).run(exec); }
  off += tcnt;
  exec.sync();
  EXPECT_EQ(off, N);
  return acc;
}

// Max relative error between `a` and `scale * ref` over the first N elements.
inline float max_rel_err(const matx::tensor_t<float, 1> &a,
                         const matx::tensor_t<float, 1> &ref, float scale, index_t N)
{
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < N; i++) {
    const float expected = scale * ref(i);
    max_abs = std::max(max_abs, std::fabs(expected));
    max_err = std::max(max_err, std::fabs(a(i) - expected));
  }
  return max_err / (1.0f + max_abs);
}

// Directly verify the segment operator's PreRun/PostRun lifecycle runs exactly
// once per feed. The old design ran two independent conv1d(...).run() and
// (next_retain = ...).run() expressions, materializing a transform segment
// twice; the probe (no idempotency guard) fails if the lifecycle is run zero or
// two times. A final flush() reconstructs the full result.
TEST(StreamingConv, SegmentLifecycleRunExactlyOnce)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 200, L = 17, chunk = 32;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) { sig(i) = std::sin(0.11f * static_cast<float>(i)); }
  auto ref = make_tensor<T>({N});
  (ref = conv1d(sig, h, MATX_C_MODE_SAME)).run(exec);

  auto stream_obj = make_conv1d_stream<T>(h, {.mode = MATX_C_MODE_SAME}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(chunk)});
  auto acc = make_tensor<T>({N});
  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    PreRunLifecycle seg_life;
    auto seg = make_prerun_tester(slice(sig, {g}, {g + nl}), seg_life);
    const index_t cnt = stream_obj.feed(seg, frame);
    if (cnt > 0) { (slice(acc, {off}, {off + cnt}) = slice(frame, {0}, {cnt})).run(exec); }
    off += cnt;
    ExpectLifecycleClean(seg_life, "conv segment feed at " + std::to_string(g));
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off}, {off + tcnt}) = slice(frame, {0}, {tcnt})).run(exec); }
  off += tcnt;
  exec.sync();
  ASSERT_EQ(off, N);
  EXPECT_LT(max_rel_err(acc, ref, 1.0f, N), 1e-4f);
}

// A bare transform-valued segment must be materialized before the convolution
// reads it. Each fed segment is conv1d(chunk, 2*ones, SAME) == 2*chunk, and
// conv1d is linear in the signal, so the result must be twice the one-shot.
TEST(StreamingConv, TransformValuedSegmentMaterializes)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 200, L = 17, chunk = 32;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) { sig(i) = std::sin(0.11f * static_cast<float>(i)); }
  auto ref = make_tensor<T>({N});
  (ref = conv1d(sig, h, MATX_C_MODE_SAME)).run(exec);
  exec.sync();

  auto acc = stream_conv_same(exec, sig, h, N, chunk,
      [](auto cs, index_t) { return conv1d(cs, 2.0f * ones<T>({1}), MATX_C_MODE_SAME); });
  EXPECT_LT(max_rel_err(acc, ref, 2.0f, N), 1e-4f);
}

// A segment that composes a transform (slice(conv1d(...)) or a binary-op around
// it) does not carry the matx_transform_op trait, so a shallow rejection would
// let it through unmaterialized. Both wrapper forms equal the plain chunk here,
// so the streamed result must match the plain one-shot.
TEST(StreamingConv, NestedTransformSegmentMaterializes)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 200, L = 17, chunk = 32;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) { sig(i) = std::sin(0.11f * static_cast<float>(i)); }
  auto ref = make_tensor<T>({N});
  (ref = conv1d(sig, h, MATX_C_MODE_SAME)).run(exec);
  exec.sync();

  // slice(conv1d(...)) : the exact wrapper that slips past is_matx_transform_op.
  auto acc_slice = stream_conv_same(exec, sig, h, N, chunk,
      [](auto cs, index_t nl) {
        return slice(conv1d(cs, ones<T>({1}), MATX_C_MODE_SAME), {0}, {nl});
      });
  EXPECT_LT(max_rel_err(acc_slice, ref, 1.0f, N), 1e-4f) << "slice(conv1d)";

  // Binary-op wrapper around a transform.
  auto acc_binop = stream_conv_same(exec, sig, h, N, chunk,
      [](auto cs, index_t) { return 1.0f * conv1d(cs, ones<T>({1}), MATX_C_MODE_SAME); });
  EXPECT_LT(max_rel_err(acc_binop, ref, 1.0f, N), 1e-4f) << "scale*conv1d";
}

// Host-executor streaming: same contract on a HostExecutor (retain buffer is
// system-allocated; all work runs on the host). Reduced sweep.
TEST(StreamingConv, HostExecutorMatchesOneShot)
{
  SingleThreadedHostExecutor exec{};
  const index_t N = 1024;
  for (matxConvCorrMode_t mode : {MATX_C_MODE_FULL, MATX_C_MODE_SAME,
                                  MATX_C_MODE_VALID}) {
    for (index_t L : {index_t(33), index_t(32)}) {
      for (index_t chunk : {index_t(7), index_t(500)}) {
        run_case(exec, N, L, mode, chunk);
      }
    }
  }
}
