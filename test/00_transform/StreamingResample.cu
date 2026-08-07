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

// Test for the ResamplePolyStream object. Streaming a signal in chunks
// (feed() ... flush()) and concatenating the produced outputs must equal a single
// one-shot resample_poly over the whole input.
//
// This compares to a floating-point tolerance rather than exact equality (unlike
// StreamingChannelize / StreamingConv, which are bit-exact): resample_poly picks
// its kernel by output length, so small streaming windows / flush can use the
// WarpCentric (warp-reduce) kernel while the large one-shot uses ElemBlock
// (serial), giving ~1e-7 differences in summation order.

#include "matx.h"
#include "matx/streaming/resample_poly_stream.h"

#include "gtest/gtest.h"
#include "prerun_tester.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace matx;
using namespace matx::test;

namespace {

// Streams the signal using the cyclic chunk-size schedule `chunks` (a
// single-entry schedule is a fixed chunk size).
template <typename Exec>
bool run_case(Exec &exec, index_t N, index_t up, index_t down,
              index_t L, const std::vector<index_t> &chunks)
{
  using T = float;
  std::string sched;
  for (auto c : chunks) { sched += std::to_string(c) + ","; }
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; ++k) {
    h(k) = std::cos(0.12f * static_cast<float>(k)) *
           std::exp(-0.02f * static_cast<float>(k)) / static_cast<float>(L);
  }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; ++i) {
    const float t = static_cast<float>(i);
    sig(i) = std::sin(0.05f * t) + 0.4f * std::sin(0.2f * t) + 1e-4f * t;
  }

  const index_t M = (N * up + down - 1) / down; // ceil(N*up/down)
  auto y_full = make_tensor<T>({M});
  (y_full = resample_poly(sig, h, up, down)).run(exec);

  const index_t max_chunk = *std::max_element(chunks.begin(), chunks.end());
  auto stream_obj = make_resample_poly_stream<T>(h, {.up = up, .down = down}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(max_chunk)});
  auto acc = make_tensor<T>({M});

  index_t off = 0;
  size_t ci = 0;
  for (index_t g = 0; g < N; ) {
    const index_t nl = std::min(chunks[ci++ % chunks.size()], N - g);
    auto in_chunk = slice(sig, {g}, {g + nl});
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

  EXPECT_EQ(off, M) << "up=" << up << " down=" << down << " chunks=" << sched;
  if (off != M) return false;

  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < M; ++i) {
    max_abs = std::max(max_abs, std::fabs(y_full(i)));
    max_err = std::max(max_err, std::fabs(acc(i) - y_full(i)));
  }
  const bool ok = max_err < 1e-4f * (1.0f + max_abs);
  EXPECT_TRUE(ok) << "up=" << up << " down=" << down << " chunks=" << sched
                  << " max_err=" << max_err << " max_abs=" << max_abs;
  return ok;
}

// Fixed-chunk convenience wrapper.
template <typename Exec>
bool run_case(Exec &exec, index_t N, index_t up, index_t down, index_t L,
              index_t chunk)
{
  return run_case(exec, N, up, down, L, std::vector<index_t>{chunk});
}

} // namespace

TEST(StreamingResample, MatchesOneShot)
{
  cudaExecutor exec{};
  struct Cfg { index_t up, down, L; };
  const index_t N = 4096;
  for (Cfg c : {Cfg{3, 2, 61}, Cfg{2, 3, 61}, Cfg{5, 3, 101},
                Cfg{4, 2, 81}, Cfg{2, 4, 81},   // non-coprime
                Cfg{1, 2, 41}, Cfg{2, 1, 41},
                Cfg{1, 1, 41}, Cfg{2, 2, 41},   // up==down: identity copy path
                Cfg{3, 2, 62}, Cfg{2, 3, 60}}) { // even L
    for (index_t chunk : {1, 7, 64, 333, 1000}) {
      run_case(exec, N, c.up, c.down, c.L, chunk);
    }
  }
}

// Feed every fixed chunk size from 1 up to max(up, down). The polyphase phase
// advances by (chunk * up) mod down each feed and the D-aligned retain length
// cycles with period down, so a bug in the alignment/retain math can hide at
// one specific feed size while passing at others. Sweeping all sizes through
// one full period (max(up,down) covers both the up- and down-limited cases)
// pins that down. Several coprime and non-coprime up/down pairs are covered.
TEST(StreamingResample, AllFeedSizesUpToMaxUpDown)
{
  cudaExecutor exec{};
  struct Cfg { index_t up, down, L; };
  const index_t N = 4096;
  for (Cfg c : {Cfg{3, 2, 61}, Cfg{2, 3, 61}, Cfg{5, 3, 101}, Cfg{3, 5, 101},
                Cfg{7, 4, 121}, Cfg{4, 7, 121},
                Cfg{4, 2, 81}, Cfg{2, 4, 81}, Cfg{8, 6, 97}, // non-coprime
                Cfg{9, 4, 131}}) {
    const index_t kmax = std::max(c.up, c.down);
    for (index_t chunk = 1; chunk <= kmax; ++chunk) {
      run_case(exec, N, c.up, c.down, c.L, chunk);
    }
  }
}

// Short filters with large up give a computed history
// H = floor((L-1)/up_reduced) == 0, which requires special handling in the
// streaming implementation.Sweep H==0 configs (plus a couple
// H==1 neighbors to guard the boundary in the other direction) against the
// one-shot, at small chunk sizes and including tiny total streams.
TEST(StreamingResample, ZeroHistoryConfigs)
{
  cudaExecutor exec{};
  struct Cfg { index_t up, down, L; };
  for (index_t N : {index_t(2), index_t(4096)}) {
    for (Cfg c : {Cfg{3, 1, 3}, Cfg{3, 2, 3}, Cfg{5, 2, 3}, Cfg{5, 1, 5},
                  Cfg{8, 3, 5}, Cfg{7, 4, 3},                 // H == 0
                  Cfg{3, 1, 2}, Cfg{5, 2, 4},                 // H == 0, even L
                  Cfg{2, 1, 3}, Cfg{4, 1, 5}}) {              // H == 1
      for (index_t chunk : {index_t(1), index_t(2), index_t(3), index_t(7),
                            index_t(64), index_t(1000)}) {
        run_case(exec, N, c.up, c.down, c.L, chunk);
      }
    }
  }
}

// Chunk sizes that vary across feeds: the dr-aligned retain length changes
// between consecutive feeds, so every (retain, chunk) handoff -- not just the
// steady-state one of a fixed chunk -- must land on the global output grid.
// Includes a zero-history (H==0) config and a non-coprime ratio.
TEST(StreamingResample, VaryingChunkSchedule)
{
  cudaExecutor exec{};
  struct Cfg { index_t up, down, L; };
  const index_t N = 4096;
  const std::vector<std::vector<index_t>> schedules = {
      {1, 7, 64, 3, 1000, 2, 33},
      {1, 2, 3, 4, 5, 6, 7},
      {500, 1, 500, 2},
  };
  for (Cfg c : {Cfg{3, 2, 61}, Cfg{2, 4, 81}, Cfg{7, 4, 121}, Cfg{3, 1, 3}}) {
    for (const auto &sched : schedules) {
      run_case(exec, N, c.up, c.down, c.L, sched);
    }
  }
}

// A flush() that throws (undersized output buffer) must not consume the
// stream: retrying with a large-enough buffer emits the full tail.
TEST(StreamingResample, FlushRetryAfterFailure)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 100, up = 3, down = 2, L = 25;
  auto h = make_tensor<T>({L});
  auto x = make_tensor<T>({N});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  for (index_t i = 0; i < N; i++) { x(i) = static_cast<float>(i + 1); }
  auto stream_obj = make_resample_poly_stream<T>(h, {.up = up, .down = down}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(N)});
  auto small = make_tensor<T>({1});
  const index_t M = (N * up + down - 1) / down; // one-shot output count
  const index_t fed = stream_obj.feed(x, frame);
  ASSERT_GT(M - fed, 1); // the tail must exceed the undersized buffer
  EXPECT_THROW(stream_obj.flush(small), detail::matxException);
  // The failed flush did not end the stream; a retry emits the full tail.
  EXPECT_EQ(stream_obj.flush(frame), M - fed);
  EXPECT_EQ(stream_obj.flush(frame), 0);
  exec.sync();
}

// flush() ends the stream: a second flush() emits nothing, feed() throws
// until reset(), and reset() restores the object for a fresh stream that
// reproduces the same counts.
TEST(StreamingResample, FlushEndsStream)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 100, up = 3, down = 2, L = 25;
  auto h = make_tensor<T>({L});
  auto x = make_tensor<T>({N});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  for (index_t i = 0; i < N; i++) { x(i) = static_cast<float>(i + 1); }
  auto stream_obj = make_resample_poly_stream<T>(h, {.up = up, .down = down}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(N)});
  const index_t M = (N * up + down - 1) / down; // one-shot output count
  const index_t fed = stream_obj.feed(x, frame);
  EXPECT_EQ(fed + stream_obj.flush(frame), M);
  EXPECT_EQ(stream_obj.flush(frame), 0);
  EXPECT_THROW(stream_obj.feed(x, frame), detail::matxException);
  stream_obj.reset();
  EXPECT_EQ(stream_obj.feed(x, frame), fed);
  EXPECT_EQ(stream_obj.flush(frame), M - fed);
  exec.sync();
}

// A transform-valued filter must be materialized once at construction, not
// indexed as an unmaterialized temporary, and the transform's arithmetic must
// land in the taps. conv1d(h, 2*ones, SAME) is a nested transform op equal to
// 2h, so the streamed result must be twice the one-shot over plain h.
TEST(StreamingResample, TransformValuedFilterMaterializes)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 512, up = 3, down = 2, L = 41, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) {
    h(k) = std::cos(0.12f * static_cast<float>(k)) *
           std::exp(-0.02f * static_cast<float>(k)) / static_cast<float>(L);
  }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) {
    const float t = static_cast<float>(i);
    sig(i) = std::sin(0.05f * t) + 0.4f * std::sin(0.2f * t) + 1e-4f * t;
  }
  const index_t M = (N * up + down - 1) / down;
  auto ref = make_tensor<T>({M});
  (ref = resample_poly(sig, h, up, down)).run(exec);

  auto filt_op = conv1d(h, 2.0f * ones<T>({1}), MATX_C_MODE_SAME); // nested transform == 2h
  auto stream_obj = make_resample_poly_stream<T>(filt_op, {.up = up, .down = down}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(chunk)});
  auto acc = make_tensor<T>({M});
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
  ASSERT_EQ(off, M);
  // Expect twice the plain-h one-shot (the filter is 2h).
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < M; i++) {
    const float expected = 2.0f * ref(i);
    max_abs = std::max(max_abs, std::fabs(expected));
    max_err = std::max(max_err, std::fabs(acc(i) - expected));
  }
  EXPECT_LT(max_err, 1e-4f * (1.0f + max_abs)) << "max_err=" << max_err;
}

// Directly verify the segment operator's PreRun/PostRun lifecycle is run
// exactly once per feed -- not skipped, not duplicated -- using the lifecycle
// probe. This is the contract feed()'s manual bracket must satisfy, and a value
// check alone cannot distinguish one lifecycle from two. The first feed uses
// the segment directly (retain empty); later feeds wrap it in the retain
// concatenation. Both paths must run the lifecycle exactly once. A final
// flush() (which has no segment) then reconstructs the full result, confirming
// the data retained from the operator segments is correct.
TEST(StreamingResample, SegmentLifecycleRunExactlyOnce)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 200, up = 3, down = 2, L = 25, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) { sig(i) = std::sin(0.11f * static_cast<float>(i)); }
  const index_t M = (N * up + down - 1) / down;
  auto ref = make_tensor<T>({M});
  (ref = resample_poly(sig, h, up, down)).run(exec);

  auto stream_obj = make_resample_poly_stream<T>(h, {.up = up, .down = down}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(chunk)});
  auto acc = make_tensor<T>({M});

  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    PreRunLifecycle seg_life;
    auto seg = make_prerun_tester(slice(sig, {g}, {g + nl}), seg_life);
    const index_t cnt = stream_obj.feed(seg, frame);
    if (cnt > 0) { (slice(acc, {off}, {off + cnt}) = slice(frame, {0}, {cnt})).run(exec); }
    off += cnt;
    // Counters are set host-side during feed(); assert without needing a sync.
    ExpectLifecycleClean(seg_life, "resample segment feed at " + std::to_string(g));
  }
  // flush() ends the stream; it takes no segment, so nothing to probe here.
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off}, {off + tcnt}) = slice(frame, {0}, {tcnt})).run(exec); }
  off += tcnt;
  exec.sync();

  ASSERT_EQ(off, M);
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < M; i++) {
    max_abs = std::max(max_abs, std::fabs(ref(i)));
    max_err = std::max(max_err, std::fabs(acc(i) - ref(i)));
  }
  EXPECT_LT(max_err, 1e-4f * (1.0f + max_abs)) << "max_err=" << max_err;
}

// A feed rejected for an undersized output must not leave the segment's
// lifecycle unbalanced. The output count is validated before PreRun, so the
// throw happens before the segment lifecycle starts -- no PreRun without a
// matching PostRun, no leaked temporary. (With the old order, the probe would
// show prerun_count == 1 and postrun_count == 0.)
TEST(StreamingResample, RejectedFeedLeavesLifecycleBalanced)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 80, up = 3, down = 2, L = 25, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) { h(k) = 1.0f / static_cast<float>(k + 1); }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) { sig(i) = std::sin(0.11f * static_cast<float>(i)); }

  auto stream_obj = make_resample_poly_stream<T>(h, {.up = up, .down = down}, exec);
  auto tiny = make_tensor<T>({1}); // far smaller than the produced count -> reject
  PreRunLifecycle seg_life;
  auto seg = make_prerun_tester(slice(sig, {0}, {chunk}), seg_life);
  EXPECT_THROW(stream_obj.feed(seg, tiny), detail::matxException);
  // Validation precedes PreRun, so the lifecycle never started (0 calls).
  ExpectLifecycleClean(seg_life, "rejected resample feed", /*expected_calls=*/0);
}

// A transform-valued SEGMENT must have its lifecycle run inside feed() so its
// temporary is materialized before the internal resample reads it. Each fed
// segment is conv1d(chunk, [1], SAME) == chunk (a transform op), so the result
// must match a one-shot over the plain signal.
TEST(StreamingResample, TransformValuedSegmentMaterializes)
{
  cudaExecutor exec{};
  using T = float;
  const index_t N = 512, up = 3, down = 2, L = 41, chunk = 40;
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; k++) {
    h(k) = std::cos(0.12f * static_cast<float>(k)) *
           std::exp(-0.02f * static_cast<float>(k)) / static_cast<float>(L);
  }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; i++) {
    const float t = static_cast<float>(i);
    sig(i) = std::sin(0.05f * t) + 0.4f * std::sin(0.2f * t) + 1e-4f * t;
  }
  const index_t M = (N * up + down - 1) / down;
  auto ref = make_tensor<T>({M});
  (ref = resample_poly(sig, h, up, down)).run(exec);

  auto stream_obj = make_resample_poly_stream<T>(h, {.up = up, .down = down}, exec);
  auto frame = make_tensor<T>({stream_obj.max_output(chunk)});
  auto acc = make_tensor<T>({M});
  index_t off = 0;
  for (index_t g = 0; g < N; g += chunk) {
    const index_t nl = std::min(chunk, N - g);
    // Transform-valued segment (equals the plain chunk).
    auto seg = conv1d(slice(sig, {g}, {g + nl}), ones<T>({1}), MATX_C_MODE_SAME);
    const index_t cnt = stream_obj.feed(seg, frame);
    if (cnt > 0) { (slice(acc, {off}, {off + cnt}) = slice(frame, {0}, {cnt})).run(exec); }
    off += cnt;
  }
  const index_t tcnt = stream_obj.flush(frame);
  if (tcnt > 0) { (slice(acc, {off}, {off + tcnt}) = slice(frame, {0}, {tcnt})).run(exec); }
  off += tcnt;
  exec.sync();
  ASSERT_EQ(off, M);
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < M; i++) {
    max_abs = std::max(max_abs, std::fabs(ref(i)));
    max_err = std::max(max_err, std::fabs(acc(i) - ref(i)));
  }
  EXPECT_LT(max_err, 1e-4f * (1.0f + max_abs)) << "max_err=" << max_err;
}

// Host-executor streaming: same contract on a HostExecutor (retain buffer is
// system-allocated; the host resampler impl handles the out_offset window).
TEST(StreamingResample, HostExecutorMatchesOneShot)
{
  SingleThreadedHostExecutor exec{};
  struct Cfg { index_t up, down, L; };
  const index_t N = 1024;
  for (Cfg c : {Cfg{3, 2, 61}, Cfg{2, 4, 81}, Cfg{3, 2, 62}, Cfg{3, 1, 3}}) {
    for (index_t chunk : {index_t(7), index_t(333)}) {
      run_case(exec, N, c.up, c.down, c.L, chunk);
    }
  }
}
