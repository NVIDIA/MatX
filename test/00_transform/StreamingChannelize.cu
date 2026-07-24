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

// Test for the ChannelizePolyStream object. Streaming a signal in chunks
// (feed() ... flush()) and concatenating the produced block outputs must equal a
// single one-shot channelize_poly over the whole input.

#include "matx.h"
#include "matx/streaming/channelize_poly_stream.h"

#include "gtest/gtest.h"
#include "prerun_tester.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <cuda/std/complex>

using namespace matx;
using namespace matx::test;

namespace {

// Streams the signal using the cyclic chunk-size schedule `chunks` (a
// single-entry schedule is a fixed chunk size).
template <typename InT, typename Exec>
bool run_case(Exec &exec, index_t N, index_t M, index_t D, index_t L,
              const std::vector<index_t> &chunks)
{
  using OutT = cuda::std::complex<float>;
  std::string sched;
  for (auto c : chunks) { sched += std::to_string(c) + ","; }
  using FiltT = float;

  auto h = make_tensor<FiltT>({L});
  for (index_t k = 0; k < L; ++k) {
    h(k) = std::cos(0.11f * static_cast<float>(k)) *
           std::exp(-0.004f * static_cast<float>(k));
  }
  auto sig = make_tensor<InT>({N});
  for (index_t n = 0; n < N; ++n) {
    const float t = static_cast<float>(n);
    if constexpr (is_complex_v<InT>) {
      sig(n) = InT{std::sin(0.05f * t) + 0.3f * std::sin(0.17f * t),
                   std::cos(0.03f * t) - 0.2f * std::sin(0.23f * t)};
    } else {
      sig(n) = std::sin(0.05f * t) + 0.3f * std::sin(0.17f * t) + 1e-4f * t;
    }
  }

  const index_t T = (N + D - 1) / D; // full number of output blocks per channel
  auto y_full = make_tensor<OutT>({T, M});
  (y_full = channelize_poly(sig, h, M, D)).run(exec);

  const index_t max_chunk = *std::max_element(chunks.begin(), chunks.end());
  auto stream_obj = make_channelize_poly_stream<InT>(h, {.num_channels = M, .decimation_factor = D}, exec);
  auto frame = make_tensor<OutT>({stream_obj.max_output(max_chunk), M});
  auto acc = make_tensor<OutT>({T, M});

  index_t off = 0;
  size_t ci = 0;
  for (index_t g = 0; g < N; ) {
    const index_t nl = std::min(chunks[ci++ % chunks.size()], N - g);
    auto in_chunk = slice(sig, {g}, {g + nl});
    g += nl;
    const index_t cnt = stream_obj.feed(in_chunk, frame);
    if (cnt > 0) {
      (slice(acc, {off, 0}, {off + cnt, M}) = slice(frame, {0, 0}, {cnt, M})).run(exec);
    }
    off += cnt;
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) {
    (slice(acc, {off, 0}, {off + tcnt, M}) = slice(frame, {0, 0}, {tcnt, M})).run(exec);
  }
  off += tcnt;
  exec.sync();

  EXPECT_EQ(off, T) << "M=" << M << " D=" << D << " chunks=" << sched;
  if (off != T) return false;

  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t t = 0; t < T; ++t) {
    for (index_t c = 0; c < M; ++c) {
      const OutT ref = y_full(t, c);
      const OutT got = acc(t, c);
      max_abs = std::max(max_abs, cuda::std::abs(ref));
      max_err = std::max(max_err, cuda::std::abs(got - ref));
    }
  }
  const bool ok = max_err < 1e-3f * (1.0f + max_abs);
  EXPECT_TRUE(ok) << "M=" << M << " D=" << D << " chunks=" << sched
                  << " max_err=" << max_err << " max_abs=" << max_abs;
  return ok;
}

// Fixed-chunk convenience wrapper.
template <typename InT, typename Exec>
bool run_case(Exec &exec, index_t N, index_t M, index_t D, index_t L,
              index_t chunk)
{
  return run_case<InT>(exec, N, M, D, L, std::vector<index_t>{chunk});
}

template <typename InT>
void sweep(cudaExecutor &exec)
{
  struct Cfg { index_t M, D; };
  const index_t N = 4096;
  for (Cfg c : {Cfg{4, 4}, Cfg{8, 8},           // maximally decimated
                Cfg{8, 4}, Cfg{8, 2},           // integer oversampled
                Cfg{6, 4}, Cfg{8, 6}, Cfg{9, 6}}) { // rational oversampled
    const index_t L = 4 * c.M; // P = 4 taps/branch
    for (index_t chunk : {index_t(1), index_t(2), index_t(3), index_t(7),
                          index_t(16), index_t(64), index_t(250), index_t(1000)}) {
      run_case<InT>(exec, N, c.M, c.D, L, chunk);
    }
  }
  // Degenerate M==D==1, L==1: history and lcm padding are both zero, so the
  // object's retain capacity floor (avoid a zero-size allocation) is what is
  // exercised here.
  run_case<InT>(exec, 100, 1, 1, 1, 7);
}

// Exercise streaming through every non-fused filter-kernel family. See the
// matching dispatch rationale in ChannelizePoly.cu (OutputElemWindow tests).
// Keep this separate from the small-configuration chunk sweep so the
// large-filter Generic cases do not run thousands of one-sample feeds.
template <typename InT>
void large_dispatch_sweep(cudaExecutor &exec)
{
  struct Cfg { index_t M, D, P; };
  const index_t N = 2051;
  for (Cfg c : {Cfg{16, 16, 8}, Cfg{64, 32, 8}, Cfg{256, 128, 8},
                Cfg{64, 32, 20}, Cfg{64, 64, 192}, Cfg{64, 48, 192}}) {
    const index_t L = c.P * c.M - 1; // partial final polyphase row
    for (index_t chunk : {index_t(7), index_t(113), index_t(509)}) {
      run_case<InT>(exec, N, c.M, c.D, L, chunk);
    }
  }
}

} // namespace

TEST(StreamingChannelize, RealInputMatchesOneShot)
{
  cudaExecutor exec{};
  sweep<float>(exec);
  large_dispatch_sweep<float>(exec);
}

TEST(StreamingChannelize, ComplexInputMatchesOneShot)
{
  cudaExecutor exec{};
  sweep<cuda::std::complex<float>>(exec);
  large_dispatch_sweep<cuda::std::complex<float>>(exec);
}

// Chunk sizes that vary across feeds: the lcm(M,D)-aligned retain length
// changes between consecutive feeds, so every (retain, chunk) handoff must
// keep the buffer start on the alignment lattice. Rational M/D included.
TEST(StreamingChannelize, VaryingChunkSchedule)
{
  cudaExecutor exec{};
  struct Cfg { index_t M, D; };
  const index_t N = 4096;
  const std::vector<std::vector<index_t>> schedules = {
      {1, 7, 64, 3, 1000, 2, 33},
      {1, 2, 3, 4, 5, 6, 7},
      {500, 1, 500, 2},
  };
  for (Cfg c : {Cfg{8, 8}, Cfg{8, 4}, Cfg{9, 6}}) {
    const index_t L = 4 * c.M;
    for (const auto &sched : schedules) {
      run_case<float>(exec, N, c.M, c.D, L, sched);
      run_case<cuda::std::complex<float>>(exec, N, c.M, c.D, L, sched);
    }
  }
}

// A flush() that throws (mis-shaped output buffer) must not consume the
// stream: retrying with a correct buffer emits the trailing block.
TEST(StreamingChannelize, FlushRetryAfterFailure)
{
  cudaExecutor exec{};
  using T = float;
  using CT = cuda::std::complex<float>;
  const index_t N = 100, M = 4, D = 3, L = 16;
  auto h = make_tensor<T>({L});
  auto x = make_tensor<T>({N});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  for (index_t i = 0; i < N; i++) { x(i) = static_cast<float>(i + 1); }
  auto stream_obj = make_channelize_poly_stream<T>(h, {.num_channels = M, .decimation_factor = D}, exec);
  auto frame = make_tensor<CT>({stream_obj.max_output(N), M});
  auto wrong_width = make_tensor<CT>({stream_obj.max_output(N), M - 1});
  const index_t T_blocks = (N + D - 1) / D;
  const index_t fed = stream_obj.feed(x, frame);
  ASSERT_EQ(T_blocks - fed, 1); // a partial trailing block exists (N % D != 0)
  EXPECT_THROW(stream_obj.flush(wrong_width), detail::matxException);
  // The failed flush did not end the stream; a retry emits the trailing block.
  EXPECT_EQ(stream_obj.flush(frame), 1);
  EXPECT_EQ(stream_obj.flush(frame), 0);
  exec.sync();
}

// flush() ends the stream: a second flush() emits zero rows, feed() throws
// until reset(), and reset() restores the object for a fresh stream that
// reproduces the same counts.
TEST(StreamingChannelize, FlushEndsStream)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 100, M = 4, D = 3, L = 16;
  auto h = make_tensor<T>({L});
  auto x = make_tensor<T>({N});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  for (index_t i = 0; i < N; i++) { x(i) = static_cast<float>(i + 1); }
  auto stream_obj = make_channelize_poly_stream<T>(h, {.num_channels = M, .decimation_factor = D}, exec);
  auto frame = make_tensor<cuda::std::complex<float>>({stream_obj.max_output(N), M});
  const index_t T_blocks = (N + D - 1) / D; // one-shot block count
  const index_t fed = stream_obj.feed(x, frame);
  EXPECT_EQ(fed + stream_obj.flush(frame), T_blocks);
  EXPECT_EQ(stream_obj.flush(frame), 0);
  EXPECT_THROW(stream_obj.feed(x, frame), detail::matxException);
  stream_obj.reset();
  EXPECT_EQ(stream_obj.feed(x, frame), fed);
  EXPECT_EQ(stream_obj.flush(frame), T_blocks - fed);
  exec.sync();
}

// A transform-valued filter must be materialized once at construction, not
// indexed as an unmaterialized temporary, and the transform's arithmetic must
// land in the taps. conv1d(h, 2*ones, SAME) is a nested transform op equal to
// 2h, and channelize is linear in the filter, so the streamed result must be
// twice the one-shot over plain h.
TEST(StreamingChannelize, TransformValuedFilterMaterializes)
{
  cudaExecutor exec{};
  using T = float;
  using CT = cuda::std::complex<float>;
  const index_t N = 512, M = 8, D = 4, L = 4 * M, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) {
    h(k) = std::cos(0.11f * static_cast<float>(k)) *
           std::exp(-0.004f * static_cast<float>(k));
  }
  auto sig = make_tensor<T>({N});
  for (index_t n = 0; n < N; n++) {
    const float t = static_cast<float>(n);
    sig(n) = std::sin(0.05f * t) + 0.3f * std::sin(0.17f * t) + 1e-4f * t;
  }
  const index_t T_blocks = (N + D - 1) / D;
  auto ref = make_tensor<CT>({T_blocks, M});
  (ref = channelize_poly(sig, h, M, D)).run(exec);

  auto filt_op = conv1d(h, 2.0f * ones<T>({1}), MATX_C_MODE_SAME); // nested transform == 2h
  auto stream_obj = make_channelize_poly_stream<T>(
      filt_op, {.num_channels = M, .decimation_factor = D}, exec);
  auto frame = make_tensor<CT>({stream_obj.max_output(chunk), M});
  auto acc = make_tensor<CT>({T_blocks, M});
  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    const index_t cnt = stream_obj.feed(slice(sig, {g}, {g + nl}), frame);
    if (cnt > 0) { (slice(acc, {off, 0}, {off + cnt, M}) = slice(frame, {0, 0}, {cnt, M})).run(exec); }
    off += cnt;
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off, 0}, {off + tcnt, M}) = slice(frame, {0, 0}, {tcnt, M})).run(exec); }
  off += tcnt;
  exec.sync();
  ASSERT_EQ(off, T_blocks);
  // Expect twice the plain-h one-shot (the filter is 2h).
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t t = 0; t < T_blocks; t++) {
    for (index_t c = 0; c < M; c++) {
      const CT expected = 2.0f * ref(t, c);
      max_abs = std::max(max_abs, cuda::std::abs(expected));
      max_err = std::max(max_err, cuda::std::abs(acc(t, c) - expected));
    }
  }
  EXPECT_LT(max_err, 1e-3f * (1.0f + max_abs)) << "max_err=" << max_err;
}

// Directly verify the segment operator's PreRun/PostRun lifecycle is run
// exactly once per feed -- not skipped, not duplicated -- using the lifecycle
// probe. This is the contract feed()'s manual bracket must satisfy, and a value
// check alone cannot distinguish one lifecycle from two. The first feed uses
// the segment directly (retain empty); later feeds wrap it in the retain
// concatenation. Both paths must run the lifecycle exactly once. A final
// flush() (which has no segment) then reconstructs the full result, confirming
// the data retained from the operator segments is correct.
TEST(StreamingChannelize, SegmentLifecycleRunExactlyOnce)
{
  cudaExecutor exec{};
  using T = float;
  using CT = cuda::std::complex<float>;
  const index_t N = 200, M = 8, D = 4, L = 4 * M, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  auto sig = make_tensor<T>({N});
  for (index_t n = 0; n < N; n++) { sig(n) = std::sin(0.11f * static_cast<float>(n)); }
  const index_t T_blocks = (N + D - 1) / D;
  auto ref = make_tensor<CT>({T_blocks, M});
  (ref = channelize_poly(sig, h, M, D)).run(exec);

  auto stream_obj = make_channelize_poly_stream<T>(
      h, {.num_channels = M, .decimation_factor = D}, exec);
  auto frame = make_tensor<CT>({stream_obj.max_output(chunk), M});
  auto acc = make_tensor<CT>({T_blocks, M});

  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    PreRunLifecycle seg_life;
    auto seg = make_prerun_tester(slice(sig, {g}, {g + nl}), seg_life);
    const index_t cnt = stream_obj.feed(seg, frame);
    if (cnt > 0) { (slice(acc, {off, 0}, {off + cnt, M}) = slice(frame, {0, 0}, {cnt, M})).run(exec); }
    off += cnt;
    // Counters are set host-side during feed(); assert without needing a sync.
    ExpectLifecycleClean(seg_life, "channelize segment feed at " + std::to_string(g));
  }
  // flush() ends the stream; it takes no segment, so nothing to probe here.
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off, 0}, {off + tcnt, M}) = slice(frame, {0, 0}, {tcnt, M})).run(exec); }
  off += tcnt;
  exec.sync();

  ASSERT_EQ(off, T_blocks);
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t t = 0; t < T_blocks; t++) {
    for (index_t c = 0; c < M; c++) {
      max_abs = std::max(max_abs, cuda::std::abs(ref(t, c)));
      max_err = std::max(max_err, cuda::std::abs(acc(t, c) - ref(t, c)));
    }
  }
  EXPECT_LT(max_err, 1e-3f * (1.0f + max_abs)) << "max_err=" << max_err;
}

// A feed rejected for an undersized output must not leave the segment's
// lifecycle unbalanced. The block count is validated before PreRun, so the
// throw happens before the segment lifecycle starts -- no PreRun without a
// matching PostRun, no leaked temporary. (With the old order, the probe would
// show prerun_count == 1 and postrun_count == 0.)
TEST(StreamingChannelize, RejectedFeedLeavesLifecycleBalanced)
{
  cudaExecutor exec{};
  using T = float;
  using CT = cuda::std::complex<float>;
  const index_t N = 80, M = 8, D = 4, L = 4 * M, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  auto sig = make_tensor<T>({N});
  for (index_t n = 0; n < N; n++) { sig(n) = std::sin(0.11f * static_cast<float>(n)); }

  auto stream_obj = make_channelize_poly_stream<T>(
      h, {.num_channels = M, .decimation_factor = D}, exec);
  // Correct channel dim but far too few rows -> reject on the block count.
  auto tiny = make_tensor<CT>({1, M});
  PreRunLifecycle seg_life;
  auto seg = make_prerun_tester(slice(sig, {0}, {chunk}), seg_life);
  EXPECT_THROW(stream_obj.feed(seg, tiny), detail::matxException);
  // Validation precedes PreRun, so the lifecycle never started (0 calls).
  ExpectLifecycleClean(seg_life, "rejected channelize feed", /*expected_calls=*/0);
}

// A transform-valued SEGMENT must have its lifecycle run inside feed() so its
// temporary is materialized before the internal channelize reads it. Each fed
// segment is conv1d(chunk, [1], SAME) == chunk (a transform op), so the result
// must match a one-shot over the plain signal.
TEST(StreamingChannelize, TransformValuedSegmentMaterializes)
{
  cudaExecutor exec{};
  using T = float;
  using CT = cuda::std::complex<float>;
  const index_t N = 512, M = 8, D = 4, L = 4 * M, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) {
    h(k) = std::cos(0.11f * static_cast<float>(k)) *
           std::exp(-0.004f * static_cast<float>(k));
  }
  auto sig = make_tensor<T>({N});
  for (index_t n = 0; n < N; n++) {
    const float t = static_cast<float>(n);
    sig(n) = std::sin(0.05f * t) + 0.3f * std::sin(0.17f * t) + 1e-4f * t;
  }
  const index_t T_blocks = (N + D - 1) / D;
  auto ref = make_tensor<CT>({T_blocks, M});
  (ref = channelize_poly(sig, h, M, D)).run(exec);

  auto stream_obj = make_channelize_poly_stream<T>(
      h, {.num_channels = M, .decimation_factor = D}, exec);
  auto frame = make_tensor<CT>({stream_obj.max_output(chunk), M});
  auto acc = make_tensor<CT>({T_blocks, M});
  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    // Transform-valued segment (equals the plain chunk).
    auto seg = conv1d(slice(sig, {g}, {g + nl}), ones<T>({1}), MATX_C_MODE_SAME);
    const index_t cnt = stream_obj.feed(seg, frame);
    if (cnt > 0) { (slice(acc, {off, 0}, {off + cnt, M}) = slice(frame, {0, 0}, {cnt, M})).run(exec); }
    off += cnt;
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off, 0}, {off + tcnt, M}) = slice(frame, {0, 0}, {tcnt, M})).run(exec); }
  off += tcnt;
  exec.sync();
  ASSERT_EQ(off, T_blocks);
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t t = 0; t < T_blocks; t++) {
    for (index_t c = 0; c < M; c++) {
      max_abs = std::max(max_abs, cuda::std::abs(ref(t, c)));
      max_err = std::max(max_err, cuda::std::abs(acc(t, c) - ref(t, c)));
    }
  }
  EXPECT_LT(max_err, 1e-3f * (1.0f + max_abs)) << "max_err=" << max_err;
}

// Host-executor streaming: same contract on a HostExecutor (retain buffer is
// system-allocated; the host channelizer impl handles the out_elem_offset
// window). Reduced sweep: maximally decimated, integer- and rational-
// oversampled, real (R2C-equivalent) and complex input.
TEST(StreamingChannelize, HostExecutorMatchesOneShot)
{
  SingleThreadedHostExecutor exec{};
  struct Cfg { index_t M, D; };
  const index_t N = 1024;
  for (Cfg c : {Cfg{8, 8}, Cfg{8, 4}, Cfg{9, 6}}) {
    const index_t L = 4 * c.M;
    for (index_t chunk : {index_t(7), index_t(250)}) {
      run_case<float>(exec, N, c.M, c.D, L, chunk);
      run_case<cuda::std::complex<float>>(exec, N, c.M, c.D, L, chunk);
    }
  }
}
