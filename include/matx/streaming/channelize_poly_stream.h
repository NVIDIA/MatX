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

// Streaming object for the 1D polyphase channelizer. Feed samples in segments and
// obtain output blocks equivalent to a single one-shot channelize_poly over the
// concatenated stream.
//
// Like Conv1DStream / ResamplePolyStream, this is a pure wrapper over the
// EXISTING one-shot channelizer -- no streaming kernel. The trick: keep the retained
// input history sized so the concatenated buffer starts at a global index that
// is a multiple of lcm(num_channels, decimation). Then the local commutator/DFT
// phase equals the global phase, so a windowed one-shot channelize over the
// buffer -- computing only the output blocks (time steps) this feed owns via the
// channelize_poly_impl out_elem_offset range -- reproduces the global result.
//
// Why lcm(M, D) and not D: the buffer start must satisfy buf_start == 0 (mod D)
// so local output blocks coincide with global blocks, AND buf_start == 0 (mod M)
// so the commutator/DFT phase aligns. Together that is buf_start == 0
// (mod lcm(M, D)). The retain length alone encodes the alignment, so no absolute
// counters are kept (overflow-safe). The output is rank-2 [blocks, M]; we window
// the block (time) dimension and keep all channels.

#pragma once

#include "matx/core/make_tensor.h"
#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"
#include "matx/executors/cuda.h"
#include "matx/generators/zeros.h"
#include "matx/operators/channelize_poly.h"
#include "matx/operators/concat.h"
#include "matx/operators/slice.h"
#include "matx/streaming/stream_detail.h"

#include <numeric>

namespace matx {

/**
 * @brief Construction-time parameters for a streaming polyphase channelizer
 * object.
 *
 * Aggregate with designated-initializer support, e.g.
 * `make_channelize_poly_stream<float>(h, {.num_channels = 8,
 * .decimation_factor = 4}, exec)`. A params struct is used so future options
 * can be added without changing call sites.
 */
struct ChannelizePolyStreamParams {
  index_t num_channels = 1;      ///< Number of channels M (positive; 1 is the degenerate single-channel FIR)
  index_t decimation_factor = 1; ///< Decimation factor D (positive, <= num_channels)
};

/**
 * @brief Streaming polyphase channelizer object; construct via
 * make_channelize_poly_stream().
 *
 * Channelizes an arbitrarily long signal delivered in segments of any (possibly
 * varying) size into rank-2 [blocks, num_channels] outputs, where each block
 * (row) is one output time step across all channels. The only stream state is
 * a small retained-history buffer; no allocation scales with the segment size.
 * All work runs asynchronously on the executor bound at construction. An object
 * serves one stream at a time and is not thread-safe.
 */
template <typename InType, typename FilterOp, typename Exec>
class ChannelizePolyStream {
public:
  ChannelizePolyStream(const FilterOp &filter,
                             const ChannelizePolyStreamParams &params, Exec exec)
      : M_(params.num_channels), D_(params.decimation_factor),
        exec_(exec)
  {
    static_assert(is_cuda_executor_v<Exec> || is_host_executor_v<Exec>,
        "ChannelizePolyStream supports CUDA and host executors");
    static_assert(FilterOp::Rank() == 1, "ChannelizePolyStream supports 1D filters");
    if (M_ <= 0 || D_ <= 0) {
      MATX_THROW(matxInvalidParameter,
          "ChannelizePolyStream: num_channels and decimation_factor must be positive");
    }
    if (D_ > M_) {
      MATX_THROW(matxInvalidParameter,
          "ChannelizePolyStream: decimation_factor must be <= num_channels");
    }
    if (filter.Size(FilterOp::Rank() - 1) < 1) {
      MATX_THROW(matxInvalidParameter,
          "ChannelizePolyStream: filter length must be >= 1");
    }
    // Filter footprint (L-1, the prototype-window span, independent of M and
    // D); the retain update below extends this by up to
    // lcm(M,D)-1 samples to keep the buffer start phase-aligned.
    H_ = filter.Size(FilterOp::Rank() - 1) - 1;
    const index_t g = std::gcd(M_, D_);
    lcm_ = M_ / g * D_; // lcm(M, D): required buffer-start alignment
    // Max retain length: the retain update adds at most nonneg_mod(...) <= lcm-1
    // to H. This is also the per-half capacity of the double-buffered retain,
    // so the allocation is exactly 2 * history_len(). One allocation, two
    // ping-pong halves (the retain update reads a view that includes itself).
    // max(.,1) at the allocation avoids a zero-size tensor in the degenerate
    // M==D==1, L==1 case (history_len_ == 0: nothing is ever retained and every
    // retain slice is empty, so the floored element is never touched). CUDA
    // executors get stream-ordered device memory; host executors get
    // system-allocated memory.
    history_len_ = H_ + lcm_ - 1;
    const index_t alloc_half = cuda::std::max(history_len_, index_t(1));
    if constexpr (is_cuda_executor_v<Exec>) {
      make_tensor(retain_buf_, {2 * alloc_half}, MATX_ASYNC_DEVICE_MEMORY,
                  exec_.getStream());
      make_tensor(filter_, {H_ + 1}, MATX_ASYNC_DEVICE_MEMORY, exec_.getStream());
    } else {
      make_tensor(retain_buf_, {2 * alloc_half}, MATX_HOST_MALLOC_MEMORY);
      make_tensor(filter_, {H_ + 1}, MATX_HOST_MALLOC_MEMORY);
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
  ChannelizePolyStream(const ChannelizePolyStream &) = delete;
  /// @copydoc ChannelizePolyStream(const ChannelizePolyStream &)
  ChannelizePolyStream &operator=(const ChannelizePolyStream &) = delete;
  /// Move-construct, transferring ownership of the stream state.
  ChannelizePolyStream(ChannelizePolyStream &&) = default;
  /// Move-assign, transferring ownership of the stream state.
  ChannelizePolyStream &operator=(ChannelizePolyStream &&) = default;

  /**
   * @brief Number of channels the stream produces (the output column count).
   *
   * @return Number of channels
   */
  index_t num_channels() const { return M_; }

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
   * @brief Upper bound on the output blocks a single feed() or flush() call
   * can produce.
   *
   * Size one reusable [max_output(largest segment size), num_channels] output
   * buffer; it is then large enough for every feed() of up to that many
   * samples and for flush().
   *
   * @param new_len Largest segment size that will be passed to feed()
   * @return Maximum output blocks (rows) a single call may produce
   */
  index_t max_output(index_t new_len) const
  {
    const index_t worst_buf = new_len + history_len_;
    return (worst_buf + D_ - 1) / D_;
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
   * @brief Feed a segment of new samples and receive the output blocks it
   * produces.
   *
   * Emits every not-yet-emitted block whose decimation_factor input samples
   * have fully arrived. The per-call count varies with the segment size (it can
   * be zero). Blocks are written to the leading rows of `out` and returned as
   * a [count, num_channels] slice of `out`. Runs asynchronously on the
   * object's executor. Consume (or copy) the returned slice before reusing
   * `out`. Throws matxInvalidParameter if called after flush(). Use reset()
   * to start a new stream.
   *
   * @tparam InOp 1D input operator type (deduced)
   * @tparam OutTensor 2D output tensor type (deduced)
   * @param new_samples New signal samples (1D, non-empty). Any MatX operator
   *   is accepted and its lifecycle is run each call. A transform-valued
   *   segment (e.g. ifft(...)) therefore works but allocates and evaluates a
   *   per-call temporary; for hot streaming loops prefer a directly-evaluable
   *   segment (a tensor, view, or generator) or materialize once and reuse.
   * @param out Output buffer shaped [>= max_output(segment size), num_channels];
   *   throws matxInvalidSize on a channel-count mismatch or if the row count
   *   is smaller than the produced block count
   * @return Slice of `out` containing the produced blocks (possibly empty)
   */
  template <typename InOp, typename OutTensor>
  auto feed(const InOp &new_samples, OutTensor &out)
  {
    static_assert(InOp::Rank() == 1, "ChannelizePolyStream::feed expects a 1D segment");
    static_assert(std::is_same_v<typename InOp::value_type, InType>,
        "ChannelizePolyStream::feed: input operator value_type must match the stream's "
        "InType (wrap the input in an explicit cast operator to convert)");
    static_assert(OutTensor::Rank() == 2, "ChannelizePolyStream::feed expects a 2D [blocks, M] output");
    static_assert(is_tensor_v<OutTensor>,
        "ChannelizePolyStream::feed: output must be a tensor or tensor view (writable, "
        "storage-backed); a transform or expression operator cannot be an output");
    if (flushed_) {
      MATX_THROW(matxInvalidParameter,
          "ChannelizePolyStream::feed: stream already flushed; call reset() first");
    }
    const index_t nl = new_samples.Size(InOp::Rank() - 1);
    MATX_ASSERT_STR(nl > 0, matxInvalidSize, "ChannelizePolyStream::feed: empty segment");

    const index_t buf_len = retain_len_ + nl;
    const index_t retain_len_next = cuda::std::min(
        H_ + nonneg_mod(buf_len - H_, lcm_), buf_len);
    // New retain is written into the OTHER ping-pong half (disjoint from the
    // half the concat buffer reads), so no aliasing.
    const index_t nxt = (1 - retain_buf_ind_) * history_len_;
    auto next_retain = slice(retain_buf_, {nxt}, {nxt + retain_len_next});

    // Validate the output before running the segment's lifecycle. The count and
    // channel dimension depend only on sizes, not segment data, so a mis-shaped
    // or undersized buffer throws here -- before any segment temporary is
    // allocated or filled.
    const auto plan = channelize_plan(buf_len, /*is_flush=*/false);
    if (out.Size(1) != M_) {
      MATX_THROW(matxInvalidSize,
          "ChannelizePolyStream: output channel dimension must equal num_channels");
    }
    if (out.Size(0) < plan.cnt) {
      MATX_THROW(matxInvalidSize,
          "ChannelizePolyStream: output buffer smaller than the produced block count");
    }

    // Materialize the segment for the two reads below (the internal channelize
    // and the retain-buffer update). The guard runs PreRun now and the matching
    // PostRun on every exit -- including an exception from exec_.Exec -- so the
    // lifecycle stays balanced and any temporary is freed. A directly-evaluable
    // segment (tensor, slice, generator) has a no-op lifecycle. The retain update
    // uses exec_.Exec (not run()) so it does not re-enter the segment's lifecycle.
    detail::SegmentLifecycleGuard<InOp, Exec> segment_guard(new_samples, exec_);

    if (retain_len_ == 0) {
      if (plan.cnt > 0) { channelize_exec(new_samples, plan.lo, plan.cnt, out); }
      auto new_tail = slice(new_samples, {nl - retain_len_next}, {nl});
      auto retain_copy = (next_retain = new_tail);
      exec_.Exec(retain_copy);
    } else {
      auto retain = cur_retain();
      auto buf = concat(0, retain, new_samples);
      if (plan.cnt > 0) { channelize_exec(buf, plan.lo, plan.cnt, out); }
      auto buf_tail = slice(buf, {buf_len - retain_len_next}, {buf_len});
      auto retain_copy = (next_retain = buf_tail);
      exec_.Exec(retain_copy);
    }

    retain_buf_ind_ = 1 - retain_buf_ind_;
    retain_len_ = retain_len_next;
    return slice(out, {0, 0}, {plan.cnt, M_});
  }

  /**
   * @brief Emit the end-of-stream trailing block, if any.
   *
   * When the total stream length is not a multiple of decimation_factor, the
   * final block is only partially covered by real samples. flush() emits that
   * final block (computed with implicit zero padding on the right), matching the
   * final block of the one-shot channelize_poly. When the total length is a
   * multiple of decimation_factor, every block was fully covered and already
   * emitted, and flush() produces zero rows. The first call emits the trailing
   * block and ends the stream. Subsequent calls return an empty slice, and
   * feed() throws until reset() starts a new stream. A flush() that throws
   * (for example, an undersized or mis-shaped output buffer) does not end the
   * stream and can be retried. Runs asynchronously on the object's executor.
   *
   * @tparam OutTensor 2D output tensor type (deduced)
   * @param out Output buffer shaped [>= max_output(segment size), num_channels];
   *   throws matxInvalidSize on a channel-count mismatch or if the row count
   *   is smaller than the produced block count
   * @return Slice of `out` containing the produced blocks (possibly empty)
   */
  template <typename OutTensor>
  auto flush(OutTensor &out)
  {
    static_assert(OutTensor::Rank() == 2, "ChannelizePolyStream::flush expects a 2D [blocks, M] output");
    static_assert(is_tensor_v<OutTensor>,
        "ChannelizePolyStream::flush: output must be a tensor or tensor view (writable, "
        "storage-backed); a transform or expression operator cannot be an output");
    if (flushed_ || retain_len_ == 0) {
      flushed_ = true;
      return slice(out, {0, 0}, {index_t(0), M_});
    }
    // flush() reads the retain buffer (a tensor), not an operator segment, so
    // there is no segment lifecycle to run here. Validate before committing
    // end-of-stream so a failed flush() (e.g. an undersized or mis-shaped
    // buffer) can be retried.
    const auto plan = channelize_plan(retain_len_, /*is_flush=*/true);
    if (out.Size(1) != M_) {
      MATX_THROW(matxInvalidSize,
          "ChannelizePolyStream: output channel dimension must equal num_channels");
    }
    if (out.Size(0) < plan.cnt) {
      MATX_THROW(matxInvalidSize,
          "ChannelizePolyStream: output buffer smaller than the produced block count");
    }
    if (plan.cnt > 0) { channelize_exec(cur_retain(), plan.lo, plan.cnt, out); }
    flushed_ = true;
    return slice(out, {0, 0}, {plan.cnt, M_});
  }

private:
  // Non-negative modulo: result in [0, m) for ANY integer a (not just
  // a >= -m): a % m is already in (-m, m) -- C++ truncated % reduces the
  // magnitude -- so a single +m correction suffices. The argument is negative
  // during warm-up when the buffer is shorter than the history (buf_len < H).
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

  // Compute the block window this call owns (start block lo and count cnt)
  //   lo = # blocks already emitted (they sit inside the retained history, which
  //        is D-aligned, so exactly floor(retain_len_/D) of them are in the buffer).
  //   hi = last block to emit (interior: last block whose newest input sample has
  //        arrived, floor(buf_len/D)-1; flush: through the local end,
  //        ceil(buf_len/D)-1, i.e. the trailing edge-padded block).
  detail::StreamSlicePlan channelize_plan(index_t buf_len, bool is_flush) const
  {
    const index_t Tloc = (buf_len + D_ - 1) / D_; // ceil(buf_len/D) local blocks
    const index_t lo = retain_len_ / D_;
    const index_t hi = is_flush ? (Tloc - 1) : (buf_len / D_ - 1);
    const index_t cnt = (hi >= lo) ? (hi - lo + 1) : index_t(0);
    return {lo, cnt};
  }

  // Run a windowed one-shot channelize over `buf`, writing blocks [lo, lo+cnt)
  // of the full grid into the first cnt rows of `out`. `out` must already be
  // validated for shape and count (see channelize_plan); cnt must be > 0.
  template <typename BufOp, typename OutTensor>
  void channelize_exec(const BufOp &buf, index_t lo, index_t cnt, OutTensor &out)
  {
    using out_value_t = typename OutTensor::value_type;
    using accum_t = typename inner_op_type_t<out_value_t>::type;
    auto os = slice(out, {0, 0}, {cnt, M_});
    // filter_ is the materialized filter tensor owned by the object.
    // AccumType is real (the output's inner type), promoted to complex
    // internally as needed.
    if constexpr (is_cuda_executor_v<Exec>) {
      matx::channelize_poly_impl<decltype(os), BufOp, decltype(filter_), accum_t>(
          os, buf, filter_, M_, D_, exec_.getStream(), /*out_elem_offset=*/lo);
    } else {
      matx::channelize_poly_impl<decltype(os), BufOp, decltype(filter_), accum_t>(
          os, buf, filter_, M_, D_, exec_, /*out_elem_offset=*/lo);
    }
  }

  // Materialized copy of the filter (owned; see the constructor).
  matx::tensor_t<typename FilterOp::value_type, 1> filter_;
  index_t M_, D_;         // num_channels, decimation_factor
  Exec exec_;
  index_t H_;             // history = L - 1
  index_t lcm_;           // lcm(M, D): buffer-start alignment
  index_t history_len_;   // max retain length == per-half retain capacity
  matx::tensor_t<InType, 1> retain_buf_; // one alloc, two ping-pong halves
  index_t retain_len_ = 0;         // current retained length (encodes phase alignment)
  int retain_buf_ind_ = 0; // active ping-pong half of retain_buf_ (0 or 1)
  bool flushed_ = false;   // end-of-stream reached; cleared by reset()
};

/**
 * @brief Create a streaming polyphase channelizer object.
 *
 * The object channelizes an arbitrarily long signal, delivered in segments of
 * any (possibly varying) size, into params.num_channels channels decimated by
 * params.decimation_factor. Callers provide input segments via @ref matx::ChannelizePolyStream::feed "feed()" and call
 * @ref matx::ChannelizePolyStream::flush "flush()" once at the end of the stream. The concatenation of the produced
 * [blocks, num_channels] slices equals a single one-shot
 * `channelize_poly(signal, filter, num_channels, decimation_factor)` over the
 * whole signal. As with `channelize_poly`, the output element type must be complex.
 *
 * The object owns a small retained-history buffer sized from the filter,
 * num_channels, and decimation_factor. No allocation scales with the segment
 * size. Each call to @ref matx::ChannelizePolyStream::feed "feed()" or @ref matx::ChannelizePolyStream::flush "flush()" accepts an output buffer shaped
 * [rows, num_channels] and returns a slice of that buffer containing the
 * produced output blocks. The output buffer must have at least
 * @ref matx::ChannelizePolyStream::max_output "max_output(max_input_segment_size)" rows. If the maximum segment size is
 * known a priori, then a single output buffer can be allocated and reused for
 * all calls.
 *
 * The operator returned by @ref matx::ChannelizePolyStream::feed "feed()" / @ref matx::ChannelizePolyStream::flush "flush()" is a `slice` of the output buffer
 * and thus it aliases the output buffer's memory. This avoids dynamic memory
 * allocation during the @ref matx::ChannelizePolyStream::feed "feed()" / @ref matx::ChannelizePolyStream::flush "flush()" calls, but the user must ensure to
 * consume the returned slice before reusing the output buffer.
 *
 * @tparam InType Sample type of the input stream (as in make_tensor<T>)
 * @tparam FilterOp Type of the filter operator (deduced)
 * @tparam Exec Executor type (CUDA or host; deduced, cudaExecutor by default)
 * @param filter FIR prototype filter (1D, length >= 1). The object
 *   materializes a copy of the filter at construction, so any filter operator
 *   (including a transform expression) is evaluated once and need not outlive
 *   the object.
 * @param params Stream parameters; see ChannelizePolyStreamParams
 * @param exec Executor bound to this stream's lifetime. All @ref matx::ChannelizePolyStream::feed "feed()"/@ref matx::ChannelizePolyStream::flush "flush()"
 *   work runs on it. For CUDA executors the retain buffer is stream-ordered
 *   device memory. For host executors the retain buffer is system-allocated memory. The
 *   executor (and, for CUDA, its stream) must outlive the object.
 * @return A streaming object exposing @ref matx::ChannelizePolyStream::feed "feed()", @ref matx::ChannelizePolyStream::flush "flush()", @ref matx::ChannelizePolyStream::max_output "max_output()", @ref matx::ChannelizePolyStream::reset "reset()",
 *   @ref matx::ChannelizePolyStream::history_len "history_len()", and @ref matx::ChannelizePolyStream::num_channels "num_channels()"
 */
template <typename InType, typename FilterOp, typename Exec = cudaExecutor>
auto make_channelize_poly_stream(const FilterOp &filter,
                                 const ChannelizePolyStreamParams &params,
                                 Exec exec = {})
{
  return ChannelizePolyStream<InType, FilterOp, Exec>(filter, params, exec);
}

} // namespace matx
