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

// Streaming object for the 1D polyphase resampler. Feed samples in segments and
// obtain outputs equivalent to a single one-shot resample_poly over the
// concatenated stream.
//
// Streaming is implemented as a pure wrapper over the existing one-shot
// implementation. The only state is the retained history; its length is
// chosen after each segment so that the concatenated [retain | new] buffer
// starts at a global sample index that is a multiple of down_reduced = down /
// gcd(up, down). That start alignment makes the local output grid a
// subset of the global output grid; each output's filter phase then matches
// automatically.
//
// Reproducibility note. The streamed outputs match a single one-shot
// resample_poly to floating-point tolerance, but are not guaranteed
// bit-for-bit identical. resample_poly selects its kernel from the output
// length and that kernel selection can differ between a streaming run and
// a one-shot run over the full input. If exact reproducibility is ever required,
// the kernel selection can be forced to always choose the same kernel. Feel free
// to file an issue with the MatX project if this need arises. This only impacts
// reproducibility between a streaming run and a one-shot run over the full input;
// repeating two streaming runs with the same input sizes will yield the same
// kernel selection heuristics.

#pragma once

#include "matx/core/make_tensor.h"
#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"
#include "matx/executors/cuda.h"
#include "matx/generators/zeros.h"
#include "matx/operators/concat.h"
#include "matx/operators/resample_poly.h"
#include "matx/operators/slice.h"
#include "matx/streaming/stream_detail.h"

#include <numeric>

namespace matx {

/**
 * @brief Construction-time parameters for a streaming polyphase resampler
 * object.
 *
 * Aggregate with designated-initializer support, e.g.
 * `make_resample_poly_stream<float>(h, {.up = 3, .down = 2}, exec)`. A params
 * struct is used so future options can be added without changing call sites.
 */
struct ResamplePolyStreamParams {
  index_t up = 1;   ///< Upsample factor (must be positive)
  index_t down = 1; ///< Downsample factor (must be positive)
};

/**
 * @brief Streaming polyphase resampler object; construct via
 * make_resample_poly_stream().
 *
 * Resamples an arbitrarily long signal delivered in segments of any (possibly
 * varying) size. The only stream state is a small retained-history buffer; no
 * allocation scales with the segment size. All work runs asynchronously on the
 * executor bound at construction. An object serves one stream at a time and is
 * not thread-safe.
 */
template <typename InType, typename FilterOp, typename Exec>
class ResamplePolyStream {
public:
  ResamplePolyStream(const FilterOp &filter,
                           const ResamplePolyStreamParams &params,
                           Exec exec)
      : exec_(exec)
  {
    static_assert(is_cuda_executor_v<Exec> || is_host_executor_v<Exec>,
        "ResamplePolyStream supports CUDA and host executors");
    static_assert(FilterOp::Rank() == 1, "ResamplePolyStream supports 1D filters");
    if (params.up <= 0 || params.down <= 0) {
      MATX_THROW(matxInvalidParameter,
          "ResamplePolyStream: up and down must be positive");
    }
    L_ = filter.Size(FilterOp::Rank() - 1);
    if (L_ < 1) {
      MATX_THROW(matxInvalidParameter,
          "ResamplePolyStream: filter length must be >= 1");
    }
    // Only the gcd-reduced factors are stored
    const index_t g = std::gcd(params.up, params.down);
    ur_ = params.up / g;
    dr_ = params.down / g;
    // Filter footprint in input samples: floor((L-1)/ur). This rounds down to
    // 0 when L <= ur, but the streaming frame still needs one retained sample:
    // outputs positioned between the previous run's last sample and this run's
    // first sample belong to this feed (their windows reach forward to the new
    // sample), and a buffer that starts at the new sample cannot emit output
    // positions before its start. One sample suffices since half < ur here.
    H_ = cuda::std::max((L_ - 1) / ur_, index_t(1));
    // Max retain length: the retain update adds at most nonneg_mod(...) <= dr-1
    // to H. This is also the per-half capacity of the double-buffered retain,
    // so the allocation is exactly 2 * history_len(). One allocation, but the
    // retention is double-buffered and ping-pongs between two halves of the buffer.
    history_len_ = H_ + dr_ - 1;
    if constexpr (is_cuda_executor_v<Exec>) {
      make_tensor(retain_buf_, {2 * history_len_}, MATX_ASYNC_DEVICE_MEMORY,
                  exec_.getStream());
      make_tensor(filter_, {L_}, MATX_ASYNC_DEVICE_MEMORY, exec_.getStream());
    } else {
      make_tensor(retain_buf_, {2 * history_len_}, MATX_HOST_MALLOC_MEMORY);
      make_tensor(filter_, {L_}, MATX_HOST_MALLOC_MEMORY);
    }
    // Materialize the filter once. The assignment runs the full operator
    // lifecycle, so any filter operator (including transform expressions) is
    // evaluated here, and the object owns the result for its lifetime.
    (filter_ = filter).run(exec_);
    reset();
    // Synchronize to ensure full filter ownership before return
    exec_.sync();
  }

  /// Streaming objects are not copyable: a copy would share the retained
  /// history buffer while tracking its state independently, corrupting both
  /// streams. Move construction/assignment transfers ownership and is allowed.
  ResamplePolyStream(const ResamplePolyStream &) = delete;
  /// @copydoc ResamplePolyStream(const ResamplePolyStream &)
  ResamplePolyStream &operator=(const ResamplePolyStream &) = delete;
  /// Move-construct, transferring ownership of the stream state.
  ResamplePolyStream(ResamplePolyStream &&) = default;
  /// Move-assign, transferring ownership of the stream state.
  ResamplePolyStream &operator=(ResamplePolyStream &&) = default;

  /**
   * @brief Maximum number of trailing input samples retained between calls.
   *
   * Informational; the caller does not size anything with it. feed()
   * concatenates the retained history internally.
   *
   * @return Maximum retained history length in samples
   */
  index_t history_len() const { return history_len_; }

  /**
   * @brief Upper bound on the outputs a single feed() or flush() call can
   * produce.
   *
   * Size one reusable output buffer with max_output(largest segment size); it is
   * then large enough for every feed() of up to that many samples and for
   * flush().
   *
   * @param new_len Largest segment size that will be passed to feed()
   * @return Maximum outputs a single call may produce
   */
  index_t max_output(index_t new_len) const
  {
    const index_t worst_buf = new_len + history_len_;
    return (worst_buf * ur_ + dr_ - 1) / dr_;
  }

  /**
   * @brief Restart the stream for a new input signal.
   *
   * Drops all retained history and clears the end-of-stream state set by
   * flush().
   */
  void reset()
  {
    (retain_buf_ = zeros<InType>({retain_buf_.Size(0)})).run(exec_);
    retain_len_ = 0;
    retain_buf_ind_ = 0;
    flushed_ = false;
  }

  /**
   * @brief Feed a segment of new samples and receive the number of outputs it
   * produces.
   *
   * Emits every not-yet-emitted output whose input samples have fully arrived;
   * the per-call count varies with the resampling ratio and segment size (it can
   * be zero). Outputs are written to the front of `out`; the return value is the
   * number written. Consume `slice(out, {0}, {count})` (the produced region)
   * before reusing `out`. Runs asynchronously on the object's executor. Throws
   * matxInvalidParameter if called after flush(). Use reset() to start a new
   * stream.
   *
   * @tparam InOp 1D input operator type (deduced)
   * @tparam OutTensor 1D output tensor type (deduced)
   * @param new_samples New signal samples (1D, non-empty). Any MatX operator
   *   is accepted and its lifecycle is run each call. A transform-valued
   *   segment (e.g. ifft(...)) therefore works but allocates and evaluates a
   *   per-call temporary; for hot streaming loops prefer a directly-evaluable
   *   segment (a tensor, view, or generator) or materialize once and reuse.
   * @param out Output buffer with last-dim size >= max_output(input_segment_size);
   *   throws matxInvalidSize if smaller than the produced count
   * @return Number of outputs written to the front of `out` (may be 0). A
   *   zero-length slice is not valid, so create and use `slice(out, {0}, {count})`
   *   only when count > 0.
   */
  template <typename InOp, typename OutTensor>
  index_t feed(const InOp &new_samples, OutTensor &out)
  {
    static_assert(InOp::Rank() == 1, "ResamplePolyStream::feed expects a 1D segment");
    static_assert(cuda::std::is_same_v<typename InOp::value_type, InType>,
        "ResamplePolyStream::feed: input operator value_type must match the stream's "
        "InType (wrap the input in an explicit cast operator to convert)");
    static_assert(OutTensor::Rank() == 1, "ResamplePolyStream::feed expects 1D output");
    static_assert(is_tensor_v<OutTensor>,
        "ResamplePolyStream::feed: output must be a tensor or tensor view (writable, "
        "storage-backed); a transform or expression operator cannot be an output");
    if (flushed_) {
      MATX_THROW(matxInvalidParameter,
          "ResamplePolyStream::feed: stream already flushed; call reset() first");
    }
    const index_t nl = new_samples.Size(InOp::Rank() - 1);
    MATX_ASSERT_STR(nl > 0, matxInvalidSize, "ResamplePolyStream::feed: empty segment");

    const index_t buf_len = retain_len_ + nl;
    const index_t retain_len_next = cuda::std::min(
        H_ + nonneg_mod(retain_len_ + nl - H_, dr_), buf_len);
    // New retain is written into the other ping-pong half (disjoint from the
    // half the concat buffer reads), so no aliasing.
    const index_t nxt = (1 - retain_buf_ind_) * history_len_;
    auto next_retain = slice(retain_buf_, {nxt}, {nxt + retain_len_next});

    // Validate the output before running the segment's lifecycle. The count
    // depends only on sizes, not segment data, so an undersized buffer throws
    // here -- before any segment temporary is allocated or filled.
    const auto plan = resample_plan(buf_len, /*is_flush=*/false, nl);
    if (out.Size(OutTensor::Rank() - 1) < plan.cnt) {
      MATX_THROW(matxInvalidSize,
          "ResamplePolyStream: output buffer smaller than the produced count");
    }

    // Materialize the segment for the two reads below (the internal resample and
    // the retain-buffer update). The guard runs PreRun now and the matching
    // PostRun on every exit -- including an exception from exec_.Exec -- so the
    // lifecycle stays balanced and any temporary is freed. A directly-evaluable
    // segment (tensor, slice, generator) has a no-op lifecycle. The retain update
    // uses exec_.Exec (not run()) so it does not re-enter the segment's lifecycle.
    detail::SegmentLifecycleGuard<InOp, Exec> segment_guard(new_samples, exec_);

    if (retain_len_ == 0) {
      if (plan.cnt > 0) { resample_exec(new_samples, plan.lo, plan.cnt, out); }
      auto new_tail = slice(new_samples, {nl - retain_len_next}, {nl});
      auto retain_copy = (next_retain = new_tail);
      exec_.Exec(retain_copy);
    } else {
      auto retain = cur_retain();
      auto buf = concat(0, retain, new_samples);
      if (plan.cnt > 0) { resample_exec(buf, plan.lo, plan.cnt, out); }
      auto buf_tail = slice(buf, {buf_len - retain_len_next}, {buf_len});
      auto retain_copy = (next_retain = buf_tail);
      exec_.Exec(retain_copy);
    }

    retain_buf_ind_ = 1 - retain_buf_ind_;
    retain_len_ = retain_len_next;
    return plan.cnt;
  }

  /**
   * @brief Emit the end-of-stream trailing outputs.
   *
   * Produces the remaining outputs whose filter windows extend past the last
   * input sample (computed with implicit zero padding on the right), matching
   * the trailing outputs of the one-shot resample_poly. The first call emits
   * the trailing outputs and ends the stream. Subsequent calls return 0, and
   * feed() throws until reset() starts a new stream. A flush() that throws (for
   * example, an undersized output buffer) does not end the stream and can be
   * retried. Runs asynchronously on the object's executor.
   *
   * @tparam OutTensor 1D output tensor type (deduced)
   * @param out Output buffer with last-dim size >= max_output(input_segment_size);
   *   throws matxInvalidSize if smaller than the produced count
   * @return Number of outputs written to the front of `out` (may be 0). A
   *   zero-length slice is not valid, so create and use `slice(out, {0}, {count})`
   *   only when count > 0.
   */
  template <typename OutTensor>
  index_t flush(OutTensor &out)
  {
    static_assert(OutTensor::Rank() == 1, "ResamplePolyStream::flush expects 1D output");
    static_assert(is_tensor_v<OutTensor>,
        "ResamplePolyStream::flush: output must be a tensor or tensor view (writable, "
        "storage-backed); a transform or expression operator cannot be an output");
    if (flushed_ || retain_len_ == 0) {
      flushed_ = true;
      return index_t(0);
    }
    // flush() reads the retain buffer (a tensor), not an operator segment, so
    // there is no segment lifecycle to run here. Validate before committing
    // end-of-stream so a failed flush() (e.g. an undersized buffer) can be
    // retried.
    const auto plan = resample_plan(retain_len_, /*is_flush=*/true, 0);
    if (out.Size(OutTensor::Rank() - 1) < plan.cnt) {
      MATX_THROW(matxInvalidSize,
          "ResamplePolyStream: output buffer smaller than the produced count");
    }
    if (plan.cnt > 0) { resample_exec(cur_retain(), plan.lo, plan.cnt, out); }
    flushed_ = true;
    return plan.cnt;
  }

private:
  // Non-negative modulo: result in [0, m) for any integer a (including negative a).
  // C++ truncated modulo reduces the magnitude, so a single +m correction suffices.
  static constexpr index_t nonneg_mod(index_t a, index_t m)
  {
    const index_t r = a % m;
    return (r < 0) ? (r + m) : r;
  }

  // View of the active retain half:
  // [retain_buf_ind_*history_len_, retain_buf_ind_*history_len_ + retain_len_).
  auto cur_retain() const
  {
    const index_t base = retain_buf_ind_ * history_len_;
    return slice(retain_buf_, {base}, {base + retain_len_});
  }

  // Compute the output window this call owns (start index lo and count cnt)
  detail::StreamSlicePlan resample_plan(index_t buf_len, bool is_flush,
                                        index_t nl) const
  {
    // ceil(buf_len*up/down); equals the raw-ratio value, so it matches the
    // one-shot output-length convention.
    const index_t Mloc = (buf_len * ur_ + dr_ - 1) / dr_;
    // Filter centering, (L-1)/2 (named to avoid shadowing the CUDA fp16
    // `half` type under -Wshadow).
    const index_t half_len = (L_ - 1) / 2;
    const index_t lo_num = retain_len_ * ur_ - 1 - half_len;
    // lo = # outputs already emitted (footprint newest inside the retain)
    const index_t lo = (lo_num < 0) ? index_t(0) : (lo_num / dr_ + 1);

    // hi = last output to emit (interior: newest input in the new segment;
    //      flush: through the local end, i.e. the trailing padded outputs)
    index_t hi;
    if (is_flush) {
      hi = Mloc - 1;
    } else {
      const index_t hi_num = (retain_len_ + nl) * ur_ - 1 - half_len;
      hi = (hi_num < 0) ? index_t(-1) : (hi_num / dr_);
      if (hi > Mloc - 1) { hi = Mloc - 1; }
    }

    const index_t cnt = (hi >= lo) ? (hi - lo + 1) : index_t(0);
    return {lo, cnt};
  }

  // Run one-shot resample_poly over `buf`, writing outputs [lo, lo+cnt) of the
  // full grid into the first cnt elements of `out`. `out` must already be
  // validated to hold cnt (see resample_plan); cnt must be > 0.
  template <typename BufOp, typename OutTensor>
  void resample_exec(const BufOp &buf, index_t lo, index_t cnt, OutTensor &out)
  {
    auto os = slice(out, {0}, {cnt});
    if (ur_ == 1 && dr_ == 1) {
      // up == down is a special case; just copy the owned inputs directly.
      // feed() owns the segment's lifecycle around this call, so copy via
      // exec_.Exec, not run(), to avoid additional segment PreRun/PostRun.
      auto id_copy = (os = slice(buf, {lo}, {lo + cnt}));
      exec_.Exec(id_copy);
    } else {
      // Compute outputs [lo, lo+cnt) of buf's full resample grid
      if constexpr (is_cuda_executor_v<Exec>) {
        detail::matxResamplePoly1DInternal(os, buf, filter_, ur_, dr_, exec_.getStream(), lo);
      } else {
        detail::matxResamplePoly1DInternal(os, buf, filter_, ur_, dr_, exec_, lo);
      }
    }
  }

  // Materialized copy of the filter (owned; see the constructor).
  matx::tensor_t<typename FilterOp::value_type, 1> filter_;
  Exec exec_;
  index_t L_;
  index_t ur_, dr_;       // reduced factors (up/gcd, down/gcd) for phase math
  index_t H_;             // history = floor((L-1)/ur)
  index_t history_len_;   // max retain length == per-half retain capacity
  matx::tensor_t<InType, 1> retain_buf_; // one alloc, two ping-pong halves
  index_t retain_len_ = 0; // current retained length (encodes phase alignment)
  int retain_buf_ind_ = 0; // active ping-pong half of retain_buf_ (0 or 1)
  bool flushed_ = false;   // end-of-stream reached; cleared by reset()
};

/**
 * @brief Create a streaming polyphase resampler object.
 *
 * The object resamples an arbitrarily long signal by `params.up` / `params.down`,
 * delivered in segments of any (possibly varying) size. Feed segments via @ref matx::ResamplePolyStream::feed "feed()"
 * and call @ref matx::ResamplePolyStream::flush "flush()" once at end of stream. The concatenation of the produced
 * outputs equals a single one-shot
 * `resample_poly(signal, filter, params.up, params.down)` over the whole
 * signal.
 *
 * The streamed outputs match the one-shot to floating-point tolerance but are
 * not guaranteed bit-for-bit identical: `resample_poly` selects its kernel from
 * the output length, and a streaming call (which computes a smaller window of
 * outputs) can select a kernel with a different summation order than the
 * full-length one-shot. Repeating a streaming run with the same segment sizes
 * produces reproducible results given an identical GPU on the same system.
 *
 * The object owns a small retained-history buffer sized from the filter and
 * the reduced resampling ratio. No allocation scales with the segment size.
 * The output buffer provided to @ref matx::ResamplePolyStream::feed "feed()" / @ref matx::ResamplePolyStream::flush "flush()" must be large enough to hold the
 * maximum number of outputs that can be produced by a single call. This value
 * is returned by @ref matx::ResamplePolyStream::max_output "max_output(max_input_segment_size)". If the maximum segment size
 * is known a priori, then a single output buffer can be allocated and reused for
 * all calls.
 *
 * Each @ref matx::ResamplePolyStream::feed "feed()" / @ref matx::ResamplePolyStream::flush "flush()" call writes the outputs to the front of the
 * output buffer and returns the number written (which may be 0), so no dynamic
 * memory is allocated during the call. Consume the produced region
 * `slice(out, {0}, {count})` before reusing the output buffer. A zero-length
 * slice is not valid, so create and use the slice only when count > 0.
 *
 * @tparam InType Sample type of the input stream (as in make_tensor<T>)
 * @tparam FilterOp Type of the filter operator (deduced)
 * @tparam Exec Executor type (CUDA or host; deduced, cudaExecutor by default)
 * @param filter FIR prototype filter (1D, length >= 1). The object
 *   materializes a copy of the filter at construction, so any filter operator
 *   (including a transform expression) is evaluated once and need not outlive
 *   the object.
 * @param params Stream parameters; see ResamplePolyStreamParams
 * @param exec Executor bound to this stream's lifetime. All @ref matx::ResamplePolyStream::feed "feed()"/@ref matx::ResamplePolyStream::flush "flush()"
 *   work runs on it. For CUDA executors the retain buffer is stream-ordered
 *   device memory. For host executors the retain buffer is system-allocated
 *   memory. The executor (and, for CUDA, its stream) must outlive the object.
 * @return A streaming object exposing @ref matx::ResamplePolyStream::feed "feed()", @ref matx::ResamplePolyStream::flush "flush()", @ref matx::ResamplePolyStream::max_output "max_output()", @ref matx::ResamplePolyStream::reset "reset()",
 *   and @ref matx::ResamplePolyStream::history_len "history_len()"
 */
template <typename InType, typename FilterOp, typename Exec = cudaExecutor>
auto make_resample_poly_stream(const FilterOp &filter,
                               const ResamplePolyStreamParams &params,
                               Exec exec = {})
{
  return ResamplePolyStream<InType, FilterOp, Exec>(filter, params, exec);
}

} // namespace matx
