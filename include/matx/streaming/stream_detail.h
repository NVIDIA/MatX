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

// Shared internal helpers for the streaming polyphase objects.

#pragma once

#include "matx/core/operator_options.h"
#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/operators/concat.h"
#include "matx/operators/slice.h"

namespace matx {
namespace detail {

// Output plan for one streaming feed()/flush() call: the number of outputs this
// call owns and the start index of that window in the local output grid.
// Computed from sizes alone (no segment data), so a feed() can validate the
// output buffer BEFORE running the segment operator's lifecycle.
struct StreamSlicePlan {
  index_t lo;
  index_t cnt;
};

// RAII balance for an explicitly-run operator lifecycle. Construction runs the
// operand's PreRun (materializing any operator that stages into a temporary)
// and destruction runs the matching PostRun on every exit path, including
// exceptions. A throw between the two (e.g. from exec.Exec, or a size check)
// therefore neither leaks the temporary nor leaves a half-run lifecycle. Both
// calls are guarded by is_matx_op, so a non-MatX operand is a no-op.
template <typename Op, typename ExecT>
class SegmentLifecycleGuard {
public:
  SegmentLifecycleGuard(const Op &op, ExecT &exec) : op_(op), exec_(exec)
  {
    if constexpr (is_matx_op<Op>()) {
      op_.PreRun(NoShape{}, exec_);
    }
  }

  ~SegmentLifecycleGuard()
  {
    if constexpr (is_matx_op<Op>()) {
      op_.PostRun(NoShape{}, exec_);
    }
  }

  SegmentLifecycleGuard(const SegmentLifecycleGuard &) = delete;
  SegmentLifecycleGuard &operator=(const SegmentLifecycleGuard &) = delete;

private:
  const Op &op_;
  ExecT &exec_;
};

// 1D view of two unit-stride segments in separate allocations, head followed
// by tail, read as one signal. The streaming objects use it in place of
// concat(0, retain, segment) when both are unit-stride tensors: each read is a
// pointer select and a single load, where ConcatOp branches between the
// operands' accessors. Transforms that recognize it (is_split_unit_stride_input_v)
// keep their raw-pointer fast paths for the other operands, and kernels that
// re-read input in an inner loop can read a whole index range from one segment
// through split_head()/split_tail(). VisitStreamBuffer selects it only for
// CUDA executors (host executors get concat), though operator() also works on
// the host for host-accessible memory. Construct it through VisitStreamBuffer,
// which checks the strides.
template <typename T>
class SplitUnitStride1DOp : public BaseOp<SplitUnitStride1DOp<T>> {
public:
  using value_type = T;
  using matx_split_unit_stride_input = bool;

  SplitUnitStride1DOp(const T *head, index_t head_len, const T *tail, index_t tail_len)
      : head_(head), tail_(tail), head_len_(head_len), tail_len_(tail_len) {}

  __MATX_INLINE__ std::string str() const { return "split_unit_stride_1d"; }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
  {
    return head_len_ + tail_len_;
  }

  // Index arithmetic stays in the caller's index type: kernels that pass
  // 32-bit indices have already checked that the whole buffer fits.
  template <typename Idx>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T operator()(Idx idx) const
  {
    const Idx head_len = static_cast<Idx>(head_len_);
    const T *p = (idx < head_len) ? (head_ + idx) : (tail_ + (idx - head_len));
    return *p;
  }

  // Raw segment access that is_split_unit_stride_input_v promises (see
  // detail::WithInputRangeReader). Indices in [0, split_head_len()) read
  // split_head(); the rest read split_tail() at idx - split_head_len().
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T *split_head() const { return head_; }
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ const T *split_tail() const { return tail_; }
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t split_head_len() const { return head_len_; }

  // Executor entry point (e.g. when a transform materializes its input).
  // get_capability pins ELEMENTS_PER_THREAD to ONE, but the CUDA executor
  // instantiates its kernel for every EPT and picks one at runtime, so the
  // multi-element form must still compile. It gathers element by element
  // (idx is in units of EPT, as for tensors) rather than returning zeros.
  template <typename CapType, typename Idx>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto operator()(Idx idx) const
  {
    if constexpr (CapType::ept == ElementsPerThread::ONE) {
      return (*this)(idx);
    } else {
      constexpr int N = static_cast<int>(CapType::ept);
      Vector<T, N> v;
      MATX_LOOP_UNROLL
      for (int k = 0; k < N; k++) {
        v.data[k] = (*this)(idx * static_cast<Idx>(N) + static_cast<Idx>(k));
      }
      return v;
    }
  }

  template <OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const
  {
    if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
      return cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
    } else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
      static_assert(cuda::std::is_same_v<remove_cvref_t<InType>, detail::AliasedMemoryQueryInput>,
                    "ALIASED_MEMORY capability requires AliasedMemoryQueryInput");
      // Either segment overlapping the queried range [start_ptr, end_ptr) aliases.
      auto overlaps = [&in](const T *p, index_t n) {
        if (n <= 0) {
          return false;
        }
        const void *start = static_cast<const void *>(p);
        const void *end = static_cast<const void *>(p + n);
        return start < in.end_ptr && in.start_ptr < end;
      };
      return overlaps(head_, head_len_) || overlaps(tail_, tail_len_);
    } else {
      return capability_attributes<Cap>::default_value;
    }
  }

private:
  const T *head_;
  const T *tail_;
  index_t head_len_;
  index_t tail_len_;
};

// Whether VisitStreamBuffer may read [head | tail] through SplitUnitStride1DOp
// for these types. The split reader indexes Data() as a dense array, so sparse
// tensors and planar-complex element types (separate real and imaginary
// planes) are excluded. Both segments' strides are still checked at runtime.
template <typename Exec, typename HeadTensor, typename TailOp>
inline constexpr bool split_stream_buffer_eligible_v =
    is_cuda_executor_v<Exec> && is_tensor_view_v<HeadTensor> && is_tensor_view_v<TailOp> &&
    HeadTensor::Rank() == 1 && TailOp::Rank() == 1 &&
    !is_sparse_tensor_v<HeadTensor> && !is_sparse_tensor_v<TailOp> &&
    !is_planar_complex_v<typename HeadTensor::value_type> &&
    !is_planar_complex_v<typename TailOp::value_type>;

// If [head[head_offset:] | tail] can be read through SplitUnitStride1DOp (a
// CUDA executor and unit-stride dense tensor views), calls fn with it and
// returns true; otherwise returns false without calling fn.
template <typename Exec, typename HeadTensor, typename TailOp, typename Fn>
bool VisitSplitStreamBuffer([[maybe_unused]] const HeadTensor &head, [[maybe_unused]] index_t head_offset,
                            [[maybe_unused]] const TailOp &tail, [[maybe_unused]] Fn &fn)
{
  if constexpr (split_stream_buffer_eligible_v<Exec, HeadTensor, TailOp>) {
    if (head.Stride(0) == 1 && tail.Stride(0) == 1) {
      fn(SplitUnitStride1DOp<typename TailOp::value_type>(
          head.Data() + head_offset, head.Size(0) - head_offset, tail.Data(), tail.Size(0)));
      return true;
    }
  }
  return false;
}

// Calls fn(buf) exactly once, where buf reads as [head | tail]: a
// SplitUnitStride1DOp when the executor is CUDA and both segments are
// unit-stride dense tensor views, otherwise concat(0, head, tail).
template <typename Exec, typename HeadTensor, typename TailOp, typename Fn>
void VisitStreamBuffer(const HeadTensor &head, const TailOp &tail, Fn &&fn)
{
  if (!VisitSplitStreamBuffer<Exec>(head, index_t(0), tail, fn)) {
    fn(concat(0, head, tail));
  }
}

// Same, but buf reads as [head[head_offset:] | tail]. head must be non-empty
// and head_offset must be in [0, head.Size(0)]. This is a separate overload so
// callers that never trim the head do not instantiate the sliced fallback.
template <typename Exec, typename HeadTensor, typename TailOp, typename Fn>
void VisitStreamBuffer(const HeadTensor &head, index_t head_offset, const TailOp &tail, Fn &&fn)
{
  MATX_ASSERT_STR(head_offset >= 0 && head_offset <= head.Size(0), matxInvalidParameter,
                  "VisitStreamBuffer: head_offset must be in [0, head.Size(0)]");
  if (!VisitSplitStreamBuffer<Exec>(head, head_offset, tail, fn)) {
    auto buf = concat(0, head, tail);
    fn(slice(buf, {head_offset}, {buf.Size(0)}));
  }
}

}  // namespace detail
}  // namespace matx
