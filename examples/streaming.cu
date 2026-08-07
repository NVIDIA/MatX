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

// Streaming filtering example.
//
// Demonstrates the canonical MatX streaming pattern: construct a streaming
// object once, then feed the signal in arbitrary-sized segments. The object owns
// the small retain buffer (and, for the polyphase transforms, the phase state),
// so per call the user only provides the new samples plus an output buffer and
// gets back the number of outputs this call wrote to the front of it. The concatenation of the
// per-segment outputs is equivalent to a single one-shot call over the whole
// stream, which the example verifies.
//
// Usage:
//   streaming [--op conv1d|resample_poly|channelize_poly]
//             [--n <samples>] [--filter <taps>] [--chunk <size>]
//             [--up <U>] [--down <D>]                (resample_poly)
//             [--channels <M>] [--decim <D>]         (channelize_poly)
//
// All three ops (conv1d, resample_poly, channelize_poly) stream through their
// make_*_stream object and verify against a one-shot call over the whole input.

#include "matx.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <cuda/std/complex>

using namespace matx;

namespace {

enum class StreamOp { Conv1D, ResamplePoly, ChannelizePoly };

struct Args {
  StreamOp op = StreamOp::Conv1D;
  index_t n = 1u << 20;   // total input samples
  index_t filter_len = 33;
  index_t chunk = 4096;   // streaming chunk size (new samples per feed)
  index_t up = 3;         // resample_poly upsample factor
  index_t down = 2;       // resample_poly downsample factor
  index_t channels = 8;   // channelize_poly number of channels (M)
  index_t decim = 4;      // channelize_poly decimation factor (D)
};

[[noreturn]] void usage(const char *prog)
{
  std::printf(
      "Usage: %s [--op conv1d|resample_poly|channelize_poly]\n"
      "          [--n <samples>] [--filter <taps>] [--chunk <size>]\n"
      "          [--up <U>] [--down <D>]           (resample_poly)\n"
      "          [--channels <M>] [--decim <D>]    (channelize_poly)\n",
      prog);
  std::exit(1);
}

Args parse_args(int argc, char **argv)
{
  Args a;
  auto need = [&](int &i) { if (++i >= argc) usage(argv[0]); return argv[i]; };
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--op")) {
      std::string v = need(i);
      if (v == "conv1d")               a.op = StreamOp::Conv1D;
      else if (v == "resample_poly")   a.op = StreamOp::ResamplePoly;
      else if (v == "channelize_poly") a.op = StreamOp::ChannelizePoly;
      else usage(argv[0]);
    } else if (!std::strcmp(argv[i], "--n")) {
      a.n = std::stoll(need(i));
    } else if (!std::strcmp(argv[i], "--filter")) {
      a.filter_len = std::stoll(need(i));
    } else if (!std::strcmp(argv[i], "--chunk")) {
      a.chunk = std::stoll(need(i));
    } else if (!std::strcmp(argv[i], "--up")) {
      a.up = std::stoll(need(i));
    } else if (!std::strcmp(argv[i], "--down")) {
      a.down = std::stoll(need(i));
    } else if (!std::strcmp(argv[i], "--channels")) {
      a.channels = std::stoll(need(i));
    } else if (!std::strcmp(argv[i], "--decim")) {
      a.decim = std::stoll(need(i));
    } else {
      usage(argv[0]);
    }
  }
  if (a.n <= 0 || a.filter_len <= 0 || a.chunk <= 0 || a.up <= 0 ||
      a.down <= 0 || a.channels <= 0 || a.decim <= 0 || a.decim > a.channels) {
    std::printf("Invalid argument value(s): all sizes/factors must be positive "
                "and decim <= channels\n");
    usage(argv[0]);
  }
  if (a.op == StreamOp::Conv1D && a.n < a.filter_len) {
    std::printf("Invalid conv1d argument value(s): input samples must be >= "
                "filter taps\n");
    usage(argv[0]);
  }
  return a;
}

// ---------------------------------------------------------------------------
// conv1d streaming demo
// ---------------------------------------------------------------------------
// Streams a signal through Conv1DStream in segments and checks that the
// concatenated output (feeds + flush) equals a one-shot
// conv1d(sig, h, MATX_C_MODE_SAME) over the whole input. SAME is the object's
// default mode.
void run_conv1d(const Args &a, cudaExecutor &exec)
{
  using T = float;
  const index_t N = a.n;
  const index_t L = a.filter_len;

  std::printf("op=conv1d  N=%lld  filter=%lld  chunk=%lld\n",
              (long long)N, (long long)L, (long long)a.chunk);

  // A simple decaying low-pass-ish FIR filter.
  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; ++k) {
    h(k) = std::cos(0.15f * static_cast<float>(k)) *
           std::exp(-0.03f * static_cast<float>(k)) / static_cast<float>(L);
  }

  // Input signal: a couple of tones plus a slow ramp so errors are visible.
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; ++i) {
    const float t = static_cast<float>(i);
    sig(i) = std::sin(0.05f * t) + 0.5f * std::sin(0.23f * t) + 1e-4f * t;
  }

  // One-shot reference: SAME mode, N outputs time-aligned with the input.
  auto ref = make_tensor<T>({N});
  (ref = conv1d(sig, h, MATX_C_MODE_SAME)).run(exec);

  // Canonical streaming pattern: a single reusable "frame"-sized output buffer.
  // Each feed() writes outputs to the front of this frame and returns the count;
  // a real pipeline would consume slice(frame, {0}, {count}) (hand it downstream,
  // process it, ...) and then reuse the frame for the next segment. Here we copy
  // into a full-length buffer so we can validate against the one-shot result.
  // example-begin conv1d_stream-1
  auto conv_stream = make_conv1d_stream<T>(h, {.mode = MATX_C_MODE_SAME}, exec);
  auto output_frame = make_tensor<T>({conv_stream.max_output(a.chunk)});
  // example-end conv1d_stream-1
  auto full = make_tensor<T>({N});

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  exec.sync();
  cudaEventRecord(start, exec.getStream());

  // example-begin conv1d_stream-2
  index_t off = 0;
  index_t nchunks = 0;
  for (index_t g = 0; g < N; g += a.chunk) {
    const index_t in_length = std::min(a.chunk, N - g);
    auto in_chunk = slice(sig, {g}, {g + in_length});

    // feed() writes outputs to the front of the reusable frame and returns how
    // many it produced. The produced region is slice(output_frame, {0}, {cnt}).
    // Everything runs on the object's bound stream, so the copy below is ordered
    // after this write and the next feed() is ordered after the copy.
    const index_t cnt = conv_stream.feed(in_chunk, output_frame);

    // For real applications, consume slice(output_frame, {0}, {cnt}) here

    // For validation only, copy the frame into the full-length buffer.
    if (cnt > 0) {
      (slice(full, {off}, {off + cnt}) = slice(output_frame, {0}, {cnt})).run(exec);
    }
    off += cnt;
    ++nchunks;
  }
  // End of stream: emit the trailing (right-zero-padded) SAME outputs.
  const index_t tcnt = conv_stream.flush(output_frame);
  if (tcnt > 0) {
    (slice(full, {off}, {off + tcnt}) = slice(output_frame, {0}, {tcnt})).run(exec);
  }
  off += tcnt;
  // example-end conv1d_stream-2

  cudaEventRecord(stop, exec.getStream());
  exec.sync();
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::printf("  streamed %lld samples in %lld chunks + flush (%.3f ms)\n",
              (long long)off, (long long)nchunks, ms);

  // Validate the emitted count before comparing values; on a mismatch the
  // comparison below would read unwritten elements of `full`.
  if (off != N) {
    std::printf("  emitted %lld outputs, expected %lld -> MISMATCH\n",
                (long long)off, (long long)N);
    return;
  }

  // Verify the reassembled stream against the one-shot reference.
  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < N; ++i) {
    max_abs = std::max(max_abs, std::fabs(ref(i)));
    max_err = std::max(max_err, std::fabs(full(i) - ref(i)));
  }
  const float rel = max_err / (1.0f + max_abs);
  const bool ok = rel < 1e-4f;
  std::printf("  max_err=%.3e  max_abs=%.3e  rel=%.3e  -> %s\n",
              max_err, max_abs, rel, ok ? "MATCH" : "MISMATCH");
}

// ---------------------------------------------------------------------------
// resample_poly streaming demo
// ---------------------------------------------------------------------------
// Same frame-based pattern as conv1d. Verified against a one-shot resample_poly over the full input.
void run_resample(const Args &a, cudaExecutor &exec)
{
  using T = float;
  const index_t N = a.n, L = a.filter_len, up = a.up, down = a.down;
  const index_t M = (N * up + down - 1) / down; // total outputs = ceil(N*up/down)

  std::printf("op=resample_poly  N=%lld  up=%lld down=%lld  filter=%lld  chunk=%lld\n",
              (long long)N, (long long)up, (long long)down,
              (long long)L, (long long)a.chunk);

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

  // One-shot reference.
  auto ref = make_tensor<T>({M});
  (ref = resample_poly(sig, h, up, down)).run(exec);

  // Streaming: one reusable frame buffer (sized via the object), consumed each call
  // example-begin resample_poly_stream-1
  auto resample_stream = make_resample_poly_stream<T>(h, {.up = up, .down = down}, exec);
  auto output_frame = make_tensor<T>({resample_stream.max_output(a.chunk)});
  // example-end resample_poly_stream-1
  auto full = make_tensor<T>({M});

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  exec.sync();
  cudaEventRecord(start, exec.getStream());

  // example-begin resample_poly_stream-2
  index_t off = 0, nchunks = 0;
  for (index_t g = 0; g < N; g += a.chunk) {
    const index_t in_length = std::min(a.chunk, N - g);
    auto in_chunk = slice(sig, {g}, {g + in_length});
    const index_t cnt = resample_stream.feed(in_chunk, output_frame); // outputs emitted

    // For real applications, consume slice(output_frame, {0}, {cnt}) here

    // For validation only, copy the frame into the full-length buffer.
    if (cnt > 0) {
      (slice(full, {off}, {off + cnt}) = slice(output_frame, {0}, {cnt})).run(exec);
    }
    off += cnt;
    ++nchunks;
  }
  // End of stream: flush the trailing (edge) outputs.
  const index_t tcnt = resample_stream.flush(output_frame);
  if (tcnt > 0) {
    (slice(full, {off}, {off + tcnt}) = slice(output_frame, {0}, {tcnt})).run(exec);
  }
  off += tcnt;
  // example-end resample_poly_stream-2

  cudaEventRecord(stop, exec.getStream());
  exec.sync();
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::printf("  streamed %lld -> %lld samples in %lld chunks + flush (%.3f ms)\n",
              (long long)N, (long long)off, (long long)nchunks, ms);

  // Validate the emitted count before comparing values; on a mismatch the
  // comparison below would read unwritten elements of `full`.
  if (off != M) {
    std::printf("  emitted %lld outputs, expected %lld -> MISMATCH\n",
                (long long)off, (long long)M);
    return;
  }

  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t i = 0; i < M; ++i) {
    max_abs = std::max(max_abs, std::fabs(ref(i)));
    max_err = std::max(max_err, std::fabs(full(i) - ref(i)));
  }
  const float rel = max_err / (1.0f + max_abs);
  const bool ok = rel < 1e-4f;
  std::printf("  max_err=%.3e  max_abs=%.3e  rel=%.3e  -> %s\n",
              max_err, max_abs, rel, ok ? "MATCH" : "MISMATCH");
}

// ---------------------------------------------------------------------------
// channelize_poly streaming demo
// ---------------------------------------------------------------------------
// Same frame-based pattern, but the output is 2D [blocks, channels]: each feed
// produces some number of whole output blocks (one time-step across all M
// channels) and returns them as [cnt, M] rows. Verified against a one-shot
// channelize_poly over the full input.
void run_channelize(const Args &a, cudaExecutor &exec)
{
  using T = float;                     // real input / filter
  using CT = cuda::std::complex<float>; // complex channelizer output
  const index_t N = a.n, L = a.filter_len, M = a.channels, D = a.decim;
  const index_t T_blocks = (N + D - 1) / D; // total output blocks = ceil(N/D)

  std::printf("op=channelize_poly  N=%lld  channels=%lld decim=%lld  filter=%lld  chunk=%lld\n",
              (long long)N, (long long)M, (long long)D,
              (long long)L, (long long)a.chunk);

  auto h = make_tensor<T>({L});
  for (index_t k = 0; k < L; ++k) {
    h(k) = std::cos(0.11f * static_cast<float>(k)) *
           std::exp(-0.004f * static_cast<float>(k)) / static_cast<float>(L);
  }
  auto sig = make_tensor<T>({N});
  for (index_t i = 0; i < N; ++i) {
    const float t = static_cast<float>(i);
    sig(i) = std::sin(0.05f * t) + 0.4f * std::sin(0.2f * t) + 1e-4f * t;
  }

  // One-shot reference: [blocks, channels].
  auto ref = make_tensor<CT>({T_blocks, M});
  (ref = channelize_poly(sig, h, M, D)).run(exec);

  // Streaming: one reusable [max_output(segment), M] frame, consumed each call;
  // here copied into `full` for validation. flush() adds the trailing block.
  // example-begin channelize_poly_stream-1
  auto channelize_stream = make_channelize_poly_stream<T>(h, {.num_channels = M, .decimation_factor = D}, exec);
  auto output_frame = make_tensor<CT>({channelize_stream.max_output(a.chunk), M});
  // example-end channelize_poly_stream-1
  auto full = make_tensor<CT>({T_blocks, M});

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  exec.sync();
  cudaEventRecord(start, exec.getStream());

  // example-begin channelize_poly_stream-2
  index_t off = 0, nchunks = 0;
  for (index_t g = 0; g < N; g += a.chunk) {
    const index_t in_length = std::min(a.chunk, N - g);
    auto in_chunk = slice(sig, {g}, {g + in_length});
    const index_t cnt = channelize_stream.feed(in_chunk, output_frame); // blocks emitted

    // For real applications, consume slice(output_frame, {0, 0}, {cnt, M}) here

    // For validation only, copy the frame into the full-length buffer.
    if (cnt > 0) {
      (slice(full, {off, 0}, {off + cnt, M}) = slice(output_frame, {0, 0}, {cnt, M})).run(exec);
    }
    off += cnt;
    ++nchunks;
  }
  // End of stream: flush the trailing (edge-padded) block, if any.
  const index_t tcnt = channelize_stream.flush(output_frame);
  if (tcnt > 0) {
    (slice(full, {off, 0}, {off + tcnt, M}) = slice(output_frame, {0, 0}, {tcnt, M})).run(exec);
  }
  off += tcnt;
  // example-end channelize_poly_stream-2

  cudaEventRecord(stop, exec.getStream());
  exec.sync();
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::printf("  streamed %lld -> %lld blocks x %lld channels in %lld chunks + flush (%.3f ms)\n",
              (long long)N, (long long)off, (long long)M, (long long)nchunks, ms);

  // Validate the emitted block count before comparing values; on a mismatch
  // the comparison below would read unwritten rows of `full`.
  if (off != T_blocks) {
    std::printf("  emitted %lld blocks, expected %lld -> MISMATCH\n",
                (long long)off, (long long)T_blocks);
    return;
  }

  float max_abs = 0.0f, max_err = 0.0f;
  for (index_t t = 0; t < T_blocks; ++t) {
    for (index_t c = 0; c < M; ++c) {
      max_abs = std::max(max_abs, cuda::std::abs(ref(t, c)));
      max_err = std::max(max_err, cuda::std::abs(full(t, c) - ref(t, c)));
    }
  }
  const float rel = max_err / (1.0f + max_abs);
  const bool ok = rel < 1e-4f;
  std::printf("  max_err=%.3e  max_abs=%.3e  rel=%.3e  -> %s\n",
              max_err, max_abs, rel, ok ? "MATCH" : "MISMATCH");
}

} // namespace

int main(int argc, char **argv)
{
  MATX_ENTER_HANDLER();

  const Args a = parse_args(argc, argv);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaExecutor exec{stream};

  switch (a.op) {
    case StreamOp::Conv1D:
      run_conv1d(a, exec);
      break;
    case StreamOp::ResamplePoly:
      run_resample(a, exec);
      break;
    case StreamOp::ChannelizePoly:
      run_channelize(a, exec);
      break;
  }

  exec.sync();
  cudaStreamDestroy(stream);

  MATX_CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
}
