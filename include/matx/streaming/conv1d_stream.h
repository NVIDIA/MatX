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

// Streaming object for 1D FIR convolution. Feed samples in segments and obtain
// outputs equivalent to a single one-shot conv1d over the concatenated stream
// in the requested mode (FULL, SAME, or VALID), for total stream lengths of at
// least the filter length (see the short-stream contract note below).
//
// All three modes share one causal engine: each feed runs a VALID conv over
// [retain(L-1) | new], whose outputs are exactly FULL[e-nl, e) of the global
// stream (e = total samples fed). A mode is just a leading skip and a
// complementary trailing flush over that engine:
//
//   mode   skip s (leading outputs dropped)   flush_len (emitted at flush())
//   FULL   0                                  L-1
//   SAME   odd L: (L-1)/2; even L: L/2-1      L-1-L/2  (== s for both parities)
//   VALID  L-1                                0
//
// where s matches the one-shot SAME start offset (kernels/conv.cuh). flush()
// emits FULL[e, e+flush_len) by convolving [retain | zeros(flush_len)].
//
// Contract for streams shorter than the filter (N < L): the object's signal
// and filter roles are fixed at construction, so the streamed result is always
// the input-aligned one -- SAME emits exactly N outputs FULL[s, s+N), computed
// with zero padding at the stream edges, and VALID emits max(0, N-L+1)
// fully-immersed outputs (no edge padding). The one-shot
// conv1d instead swaps operand roles when the signal is smaller than the
// filter (output sized to the larger operand); convolution is commutative, so
// the swap changes which slice of the same full convolution is returned, not
// its values. Streaming output therefore equals the one-shot whenever
// N >= L; FULL mode is role-symmetric and matches for any N.

#pragma once

#include "matx/core/make_tensor.h"
#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"
#include "matx/executors/cuda.h"
#include "matx/generators/zeros.h"
#include "matx/operators/concat.h"
#include "matx/operators/conv.h"
#include "matx/operators/slice.h"
#include "matx/streaming/stream_detail.h"

namespace matx {

/**
 * @brief Construction-time parameters for a streaming 1D convolution object.
 *
 * Aggregate with designated-initializer support, e.g.
 * `make_conv1d_stream<float>(h, {.mode = MATX_C_MODE_FULL}, exec)`. A params
 * struct is used so future options can be added without changing call sites.
 */
struct Conv1DStreamParams {
  /**
   * Edge mode of the equivalent one-shot conv1d over the concatenated stream.
   * SAME (default): one output per input sample, time-aligned with the input
   * (the filter's group delay is removed). FULL: causal; one output per input
   * sample during feed() plus the L-1 trailing outputs at flush(). VALID: only
   * outputs whose filter window is fully immersed in real samples.
   */
  matxConvCorrMode_t mode = MATX_C_MODE_SAME;
};

/**
 * @brief Streaming 1D convolution object; construct via make_conv1d_stream().
 *
 * Filters an arbitrarily long signal delivered in segments of any (possibly
 * varying) size. The only stream state is a filter-length retain buffer; no
 * allocation scales with the segment size. All work runs asynchronously on the
 * executor bound at construction. An object serves one stream at a time and is
 * not thread-safe. The convolution uses the direct (time-domain) method, which
 * limits the filter to 1024 taps.
 */
template <typename InType, typename FilterOp, typename Exec>
class Conv1DStream {
public:
  Conv1DStream(const FilterOp &filter, const Conv1DStreamParams &params,
                     Exec exec)
      : exec_(exec)
  {
    static_assert(is_cuda_executor_v<Exec> || is_host_executor_v<Exec>,
        "Conv1DStream supports CUDA and host executors");
    static_assert(FilterOp::Rank() == 1, "Conv1DStream currently supports 1D filters");
    L_ = filter.Size(FilterOp::Rank() - 1);
    if (L_ < 1) {
      MATX_THROW(matxInvalidParameter, "Conv1DStream: filter length must be >= 1");
    }
    // The object always uses direct (time-domain) convolution, whose kernel
    // supports at most CONV1D_MAX_MIN_DIMENSION_DIRECT (1024) filter taps.
    if (L_ > detail::CONV1D_MAX_MIN_DIMENSION_DIRECT) {
      MATX_THROW(matxInvalidParameter,
          "Conv1DStream: filter length exceeds the direct convolution limit "
          "(CONV1D_MAX_MIN_DIMENSION_DIRECT)");
    }
    switch (params.mode) {
      case MATX_C_MODE_FULL:
        skip_ = 0;
        flush_len_ = L_ - 1;
        break;
      case MATX_C_MODE_SAME:
        // Start offset of one-shot SAME within FULL (kernels/conv.cuh); the
        // trailing SAME outputs beyond FULL[N-1] number L-1-L/2 (== skip_ for
        // both parities).
        skip_ = (L_ % 2 == 1) ? (L_ - 1) / 2 : L_ / 2 - 1;
        flush_len_ = L_ - 1 - L_ / 2;
        break;
      case MATX_C_MODE_VALID:
        skip_ = L_ - 1;
        flush_len_ = 0;
        break;
      default:
        MATX_THROW(matxInvalidParameter, "Conv1DStream: unsupported mode");
    }
    // One allocation holding both ping-pong halves ([0,H) and [H,2H)) instead
    // of two separate tensors — fewer allocations / less fragmentation when
    // many objects exist. max(H,1) avoids a zero-size allocation when L==1.
    // CUDA executors get stream-ordered device memory; host executors get
    // system-allocated memory.
    const index_t rsz = cuda::std::max(L_ - 1, index_t(1));
    if constexpr (is_cuda_executor_v<Exec>) {
      make_tensor(retain_buf_, {2 * rsz}, MATX_ASYNC_DEVICE_MEMORY, exec_.getStream());
      make_tensor(filter_, {L_}, MATX_ASYNC_DEVICE_MEMORY, exec_.getStream());
    } else {
      make_tensor(retain_buf_, {2 * rsz}, MATX_HOST_MALLOC_MEMORY);
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
  Conv1DStream(const Conv1DStream &) = delete;
  /// @copydoc Conv1DStream(const Conv1DStream &)
  Conv1DStream &operator=(const Conv1DStream &) = delete;
  /// Move-construct, transferring ownership of the stream state.
  Conv1DStream(Conv1DStream &&) = default;
  /// Move-assign, transferring ownership of the stream state.
  Conv1DStream &operator=(Conv1DStream &&) = default;

  /**
   * @brief Number of trailing input samples retained between calls (L - 1).
   *
   * Informational; the caller does not size anything with it. feed()
   * concatenates the retained history internally.
   *
   * @return Retained history length in samples
   */
  index_t history_len() const { return L_ - 1; }

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
    return cuda::std::max(new_len, flush_len_);
  }

  /**
   * @brief Restart the stream for a new input signal.
   *
   * Clears the retained history (zero initial conditions), the mode's startup
   * skip, and the end-of-stream state set by flush().
   */
  void reset()
  {
    (retain_buf_ = zeros<InType>({retain_buf_.Size(0)})).run(exec_);
    retain_buf_ind_ = 0;
    skip_rem_ = skip_;
    flushed_ = false;
  }

  /**
   * @brief Feed a segment of new samples and receive the number of outputs it
   * produces.
   *
   * Outputs are written to the front of `out`; the return value is the number
   * of outputs written. Consume `slice(out, {0}, {count})` (the produced region)
   * before reusing `out`. Runs asynchronously on the object's executor. Emits
   * up to one output per new sample; the first feeds of a SAME or VALID stream
   * emit fewer (possibly zero) while the mode's leading outputs are skipped.
   * Throws matxInvalidParameter if called after flush(). Use reset() to start a
   * new stream.
   *
   * @tparam InOp 1D input operator type (deduced)
   * @tparam OutTensor 1D output tensor type (deduced)
   * @param new_samples New signal samples (1D, non-empty). Any MatX operator is
   *   accepted and its lifecycle is run each call. A transform-valued segment
   *   (e.g. ifft(...) or slice(ifft(...))) therefore works but allocates and
   *   evaluates a per-call temporary; for hot streaming loops prefer a
   *   directly-evaluable segment (a tensor, view, or generator) or materialize
   *   once and reuse.
   * @param out Output buffer with last-dim size >= max_output(input_segment_size);
   *   throws matxInvalidSize if smaller than the produced count
   * @return Number of outputs written to the front of `out` (may be 0). A
   *   zero-length slice is not valid, so create and use `slice(out, {0}, {count})`
   *   only when count > 0.
   */
  template <typename InOp, typename OutTensor>
  index_t feed(const InOp &new_samples, OutTensor &out)
  {
    static_assert(InOp::Rank() == 1, "Conv1DStream::feed expects a 1D segment");
    static_assert(cuda::std::is_same_v<typename InOp::value_type, InType>,
        "Conv1DStream::feed: input operator value_type must match the stream's "
        "InType (wrap the input in an explicit cast operator to convert)");
    static_assert(OutTensor::Rank() == 1, "Conv1DStream::feed expects a 1D output");
    static_assert(is_tensor_v<OutTensor>,
        "Conv1DStream::feed: output must be a tensor or tensor view (writable, "
        "storage-backed); a transform or expression operator cannot be an output");
    if (flushed_) {
      MATX_THROW(matxInvalidParameter,
          "Conv1DStream::feed: stream already flushed; call reset() first");
    }
    const index_t nl = new_samples.Size(InOp::Rank() - 1);
    MATX_ASSERT_STR(nl > 0, matxInvalidSize, "Conv1DStream::feed: empty segment");

    // This feed's VALID outputs are FULL[e-nl, e); drop the leading `d` still
    // owed to the mode's startup skip by trimming the buffer front instead
    // (VALID over sig[d:] starts at FULL[e-nl+d]; no skipped output computed).
    // The count depends only on sizes, so validate the output before running
    // the segment's lifecycle. Output slices are taken only when cnt > 0, since
    // MatX tensors cannot represent a zero-length slice.
    const index_t d = cuda::std::min(skip_rem_, nl);
    const index_t cnt = nl - d;
    if (out.Size(OutTensor::Rank() - 1) < cnt) {
      MATX_THROW(matxInvalidSize,
          "Conv1DStream::feed: output buffer smaller than the produced count");
    }

    // Materialize the segment for the two reads below (the convolution and the
    // retain-buffer update). The guard runs PreRun now and the matching PostRun
    // on every exit -- including an exception -- so the lifecycle stays balanced
    // and any temporary is freed. A transform segment, or an operator built on
    // one (e.g. slice(ifft(...)) whose PreRun forwards to the nested transform),
    // is materialized exactly once; a directly-evaluable segment (tensor, slice,
    // generator) has a no-op lifecycle. The convolution and the retain update
    // run through the impl / exec_.Exec (not run()) so neither re-enters the
    // segment's lifecycle.
    detail::SegmentLifecycleGuard<InOp, Exec> segment_guard(new_samples, exec_);

    if (L_ == 1) { // degenerate length-1 filter: pointwise, no retained history
      if (cnt > 0) {
        conv1d_impl(slice(out, {0}, {cnt}), new_samples, filter_, MATX_C_MODE_VALID,
                    MATX_C_METHOD_DIRECT, exec_);
      }
      skip_rem_ -= d; // skip_ == 0 for L == 1, so d == 0; kept for uniformity
      return cnt;
    }

    // Lazy [retain | new] view. The retain lives in half `retain_buf_ind_` of
    // the shared buffer.
    auto retain = cur_retain();
    auto sig = concat(0, retain, new_samples);
    if (cnt > 0) {
      conv1d_impl(slice(out, {0}, {cnt}), slice(sig, {d}, {L_ - 1 + nl}), filter_,
                  MATX_C_MODE_VALID, MATX_C_METHOD_DIRECT, exec_);
    }

    // New retain = trailing L-1 samples of [retain | new]. Written into the
    // OTHER half (disjoint from the half `sig` reads), so no aliasing. Uses
    // exec_.Exec (not run()) so it does not re-enter the segment's lifecycle.
    const index_t nxt = (1 - retain_buf_ind_) * (L_ - 1);
    auto next_retain = slice(retain_buf_, {nxt}, {nxt + L_ - 1});
    auto sig_tail    = slice(sig, {nl}, {nl + L_ - 1});
    auto retain_copy = (next_retain = sig_tail);
    exec_.Exec(retain_copy);
    retain_buf_ind_ = 1 - retain_buf_ind_;
    skip_rem_ -= d;

    return cnt;
  }

  /**
   * @brief Emit the end-of-stream trailing outputs for the configured mode.
   *
   * FULL emits the L-1 trailing (right-zero-padded) outputs, SAME emits
   * L-1-L/2, and VALID emits none. The first call emits the trailing outputs
   * and ends the stream. Subsequent calls return 0, and feed() throws until
   * reset() starts a new stream. A flush() that throws (for example, an
   * undersized output buffer) does not end the stream and can be retried. Runs
   * asynchronously on the object's executor.
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
    static_assert(OutTensor::Rank() == 1, "Conv1DStream::flush expects a 1D output");
    static_assert(is_tensor_v<OutTensor>,
        "Conv1DStream::flush: output must be a tensor or tensor view (writable, "
        "storage-backed); a transform or expression operator cannot be an output");
    if (flushed_) {
      return index_t(0);
    }
    const index_t d = cuda::std::min(skip_rem_, flush_len_);
    const index_t cnt = flush_len_ - d;
    if (cnt == 0) {
      flushed_ = true;
      return index_t(0);
    }
    if (out.Size(OutTensor::Rank() - 1) < cnt) {
      MATX_THROW(matxInvalidSize,
          "Conv1DStream::flush: output buffer smaller than the produced count");
    }
    // [retain | zeros(flush_len)] has length L-1+flush_len; VALID over its
    // [d:] suffix yields exactly the cnt trailing outputs. flush() reads the
    // retain buffer (a tensor) and a zeros generator -- no operator segment --
    // so there is no segment lifecycle to run here.
    auto retain = cur_retain();
    auto sig = concat(0, retain, zeros<InType>({flush_len_}));
    conv1d_impl(slice(out, {0}, {cnt}), slice(sig, {d}, {L_ - 1 + flush_len_}), filter_,
                MATX_C_MODE_VALID, MATX_C_METHOD_DIRECT, exec_);
    // Commit end-of-stream only after validation and scheduling succeed, so a
    // failed flush() (e.g. an undersized output buffer) can be retried.
    flushed_ = true;
    return cnt;
  }

private:
  // View of the active retain half ([retain_buf_ind_*H, retain_buf_ind_*H + H)).
  auto cur_retain() const
  {
    const index_t base = retain_buf_ind_ * (L_ - 1);
    return slice(retain_buf_, {base}, {base + L_ - 1});
  }

  // Materialized copy of the filter (owned; see the constructor).
  matx::tensor_t<typename FilterOp::value_type, 1> filter_;
  Exec exec_;
  index_t L_;
  index_t skip_;      // leading FULL outputs the mode drops (see table above)
  index_t flush_len_; // trailing outputs the mode emits at flush()
  // Single allocation; halves [0,H) and [H,2H) ping-pong to avoid update alias.
  matx::tensor_t<InType, 1> retain_buf_;
  index_t skip_rem_ = 0;    // startup skip still owed (bounded by skip_)
  int retain_buf_ind_ = 0;  // active ping-pong half of retain_buf_ (0 or 1)
  bool flushed_ = false;    // end-of-stream reached; cleared by reset()
};

/**
 * @brief Create a streaming 1D convolution object.
 *
 * The object filters an arbitrarily long signal delivered in segments of any
 * (possibly varying) size. Feed segments via @ref matx::Conv1DStream::feed "feed()" and call @ref matx::Conv1DStream::flush "flush()" once at
 * end of stream. The concatenation of the produced outputs equals a single
 * one-shot `conv1d(signal, filter, params.mode)` over the whole signal
 * whenever the total signal length N is at least the filter length L. For
 * N < L the one-shot swaps its operand roles (the smaller operand becomes the
 * filter and the output is sized to the larger). The object instead keeps the
 * roles fixed and produces the input-aligned result, where SAME emits exactly
 * N outputs and VALID emits max(0, N-L+1). FULL is role-symmetric and matches
 * the one-shot for any N.
 *
 * The object computes the convolution with the direct (time-domain) method.
 * The filter is therefore limited to 1024 taps, and the constructor throws for
 * longer filters. FFT-based streaming convolution may be added in a future
 * update.
 *
 * The object owns a filter-length retain buffer. No allocation scales with the
 * segment size. The output buffer provided to @ref matx::Conv1DStream::feed "feed()" / @ref matx::Conv1DStream::flush "flush()" must be large
 * enough to hold the maximum number of outputs that can be produced by a
 * single call. This value is returned by @ref matx::Conv1DStream::max_output "max_output(max_input_segment_size)".
 * If the maximum segment size is known a priori, then a single output buffer
 * can be allocated and reused for all calls.
 *
 * Each @ref matx::Conv1DStream::feed "feed()" / @ref matx::Conv1DStream::flush "flush()" call writes the outputs to the front of the
 * output buffer and returns the number written (which may be 0), so no dynamic
 * memory is allocated during the call. Consume the produced region
 * `slice(out, {0}, {count})` before reusing the output buffer. A zero-length
 * slice is not valid, so create and use the slice only when count > 0.
 *
 * @tparam InType Sample type of the input stream (as in make_tensor<T>)
 * @tparam FilterOp Type of the filter operator (deduced)
 * @tparam Exec Executor type (CUDA or host; deduced, cudaExecutor by default)
 * @param filter FIR filter (1D, length 1 to 1024). The object materializes a
 *   copy of the filter at construction, so any filter operator (including a
 *   transform expression) is evaluated once and need not outlive the object.
 * @param params Stream parameters; see Conv1DStreamParams
 * @param exec Executor bound to this stream's lifetime. All @ref matx::Conv1DStream::feed "feed()"/@ref matx::Conv1DStream::flush "flush()"
 *   work runs on it. For CUDA executors the retain buffer is stream-ordered
 *   device memory. For host executors the retain buffer is system-allocated
 *   memory. The executor (and, for CUDA, its stream) must outlive the object.
 * @return A streaming object exposing @ref matx::Conv1DStream::feed "feed()", @ref matx::Conv1DStream::flush "flush()", @ref matx::Conv1DStream::max_output "max_output()", @ref matx::Conv1DStream::reset "reset()",
 *   and @ref matx::Conv1DStream::history_len "history_len()"
 */
template <typename InType, typename FilterOp, typename Exec = cudaExecutor>
auto make_conv1d_stream(const FilterOp &filter, const Conv1DStreamParams &params,
                        Exec exec = {})
{
  return Conv1DStream<InType, FilterOp, Exec>(filter, params, exec);
}

} // namespace matx
