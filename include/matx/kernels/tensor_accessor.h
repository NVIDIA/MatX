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

#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "matx/core/defines.h"
#include "matx/core/type_utils.h"

namespace matx {
namespace detail {

// Lazily-evaluated "pointer type of Op::Data()" helper. Placing
// decltype(op.Data()) inside a cuda::std::conditional_t would eagerly evaluate it
// in both branches and fail to compile for operators without a Data() method
// (e.g. ConstVal, CloneOp, ZipVecOp). Partial specialization defers the
// decltype until is_tensor_view_v<Op> is known true.
template <typename Op, bool HasData>
struct data_ptr_of_op { using type = cuda::std::nullptr_t; };

template <typename Op>
struct data_ptr_of_op<Op, true> {
    using type = decltype(cuda::std::declval<const Op&>().Data());
};

template <typename Op>
using data_ptr_of_op_t = typename data_ptr_of_op<Op, is_tensor_view_v<Op>>::type;

// 1D operators that read as two unit-stride segments in separate allocations,
// head followed by tail (the streaming objects' SplitUnitStride1DOp). Such an
// operator provides split_head(), split_tail() and split_head_len(): indices
// in [0, split_head_len()) read split_head(), the rest read split_tail() at
// idx - split_head_len(). There is no single base pointer, so TensorAccessor
// still reads it through its own operator(). Transforms use the trait to:
//  - let it join an IsUnitStride launch, so it doesn't force the other
//    operands onto the strided path (its UNIT_STRIDE_LAST capability is the
//    default true), and
//  - in kernels that re-read input in an inner loop, pick the segment once
//    for a whole index range and read it through a raw pointer.
template <typename Op>
inline constexpr bool is_split_unit_stride_input_v =
    requires { typename remove_cvref_t<Op>::matx_split_unit_stride_input; };

// Returns fir(read, shift), where read(i - shift) returns input element i for
// every i in [lo, hi] that the caller reads; callers index read() with i - shift.
// For a two-segment split input (is_split_unit_stride_input_v), reading through
// the accessor selects the head or tail segment per element: a compare, two
// selects and an address computation per read, which dominates issue-bound FIR
// loops that read each input element once per tap. When [lo, hi] lies in one
// segment, read loads through that segment's pointer instead, so the select
// happens once per range. The shift is applied to the caller's loop indices
// rather than inside read() so the loop stays a plain pointer walk, as for a
// tensor input. Ranges that straddle the boundary, and all other input types,
// read through input_acc. UseSegments = false also reads through input_acc:
// the second copy of the loop costs registers, and whether that outweighs the
// per-element select depends on the kernel and element size, so callers decide.
template <bool UseSegments = true, typename InType, typename InAcc, typename IdxT, typename Fir>
__MATX_DEVICE__ __MATX_INLINE__ auto WithInputRangeReader(const InType &input, const InAcc &input_acc,
                                                       IdxT lo, IdxT hi, Fir &&fir)
{
  if constexpr (UseSegments && is_split_unit_stride_input_v<InType>) {
    // The segment path indexes the op directly, without input_acc's batch
    // binding, which is only correct for a 1D input.
    static_assert(InType::Rank() == 1,
                  "WithInputRangeReader: two-segment split inputs must be 1D");
    const IdxT head_len = static_cast<IdxT>(input.split_head_len());
    const typename InType::value_type *seg = nullptr;
    IdxT shift = 0;
    if (lo >= head_len) {
      seg = input.split_tail();
      shift = head_len;
    } else if (hi < head_len) {
      seg = input.split_head();
    }
    if (seg != nullptr) {
      return fir([seg](IdxT i) { return seg[i]; }, shift);
    }
  }
  return fir([&input_acc](IdxT i) { return input_acc(i); }, IdxT{0});
}

// Forward declaration — defined below.
template <typename Op, bool IsUnitStride, int NumBound, typename IdxT = index_t>
struct BoundAccessor;

// ----------------------------------------------------------------------------
// TensorAccessor
//
// Wraps a MatX operator and picks, per instantiation, between a raw-pointer
// fast path and the operator's operator() fallback based on
// (IsUnitStride && is_tensor_view_v<Op>).
//
// Fast path: element access is base_ptr[i0*Stride(0) + ... + i_{N-1}], with
// the last-dim stride folded to 1 (IsUnitStride guarantees that). Stores the
// data pointer and Stride(0)..Stride(Rank-2) as members, so grid-constant
// LDCs for those values are materialized once at construction instead of
// being re-issued inside the hot loop.
//
// Slow path: forwards to op(indices...). Works for any MatX operator,
// including computed ones without Data()/Stride() (ConstVal, CloneOp, ...).
//
// Rank 0 always takes the slow path — nothing to elide.
//
// bind(leading...) strips the leading K dimensions and returns a
// BoundAccessor with (Rank - K) remaining dims. On the fast path it
// precomputes a shifted base pointer so the bound dims cost nothing at
// access time; on the slow path it captures the bound indices and forwards
// them to op(leading..., is...).
// ----------------------------------------------------------------------------
// IdxT selects the width of the offset arithmetic. It defaults to index_t.
// Callers that have already proven every index the kernel forms fits in 32 bits
// can pass int32_t to collapse the 64-bit integer arithmetic into 32-bit arithmetic.
template <typename Op, bool IsUnitStride, typename IdxT = index_t, int RankParam = Op::Rank()>
struct TensorAccessor {
    static_assert(is_matx_op<Op>(), "TensorAccessor requires a MatX operator");
    static constexpr bool FastPath = IsUnitStride && is_tensor_view_v<Op>;
    static constexpr int Rank = RankParam;
    using value_type = typename Op::value_type;

    // Number of outer strides to cache. Rank 0 and 1 have none; store a
    // size-1 buffer in those cases to avoid zero-length arrays.
    static constexpr int NumOuter = (Rank >= 2) ? (Rank - 1) : 0;
    static constexpr int NumOuterStored = (NumOuter > 0) ? NumOuter : 1;

    const Op& op_;
    data_ptr_of_op_t<Op> data_;
    IdxT outer_strides_[NumOuterStored];

    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    explicit TensorAccessor(const Op& op) : op_(op), data_(nullptr) {
        if constexpr (FastPath) {
            data_ = op.Data();
            if constexpr (NumOuter > 0) {
                MATX_LOOP_UNROLL
                for (int d = 0; d < NumOuter; ++d) {
                    outer_strides_[d] = static_cast<IdxT>(op.Stride(d));
                }
            }
        }
    }

    // Rank-0 access.
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    decltype(auto) operator()() const requires (Rank == 0) { return op_(); }

    // Rank >= 1 access; arity must match Rank.
    template <typename... Is>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    decltype(auto) operator()(Is... is) const
        requires (Rank >= 1 && sizeof...(Is) == Rank)
    {
        if constexpr (FastPath) {
            const IdxT idx_arr[] = { static_cast<IdxT>(is)... };
            // Last-dim stride is 1 under IsUnitStride.
            IdxT offset = idx_arr[Rank - 1];
            if constexpr (NumOuter > 0) {
                MATX_LOOP_UNROLL
                for (int d = 0; d < NumOuter; ++d) {
                    offset += idx_arr[d] * outer_strides_[d];
                }
            }
            return data_[offset];
        } else {
            return op_(is...);
        }
    }

    // Bind the first sizeof...(Leading) dimensions. Returns a BoundAccessor
    // with (Rank - sizeof...(Leading)) remaining dims.
    template <typename... Leading>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    auto bind(Leading... leading) const
        requires (sizeof...(Leading) > 0 && sizeof...(Leading) <= Rank)
    {
        return BoundAccessor<Op, IsUnitStride, sizeof...(Leading), IdxT>(
            *this, static_cast<index_t>(leading)...);
    }
};

// ----------------------------------------------------------------------------
// BoundAccessor
//
// A TensorAccessor with the first NumBound leading dimensions fixed.
// Accesses supply only the remaining indices.
//
// Fast path: base_ = op.Data() + Σ leading[d] * Stride(d) for d in [0, NumBound);
// rem_outer_strides_ holds the still-needed Stride(NumBound .. Rank-2).
// Per-access work is identical to TensorAccessor's fast path for the lower
// rank — no per-access cost for the bound dims.
//
// Slow path: stores the bound indices and forwards them, together with the
// per-access indices, to op(leading..., is...).
// ----------------------------------------------------------------------------
template <typename Op, bool IsUnitStride, int NumBound, typename IdxT>
struct BoundAccessor {
    static_assert(NumBound > 0, "BoundAccessor requires at least one bound dim");
    static_assert(is_matx_op<Op>(), "BoundAccessor requires a MatX operator");
    static constexpr bool FastPath = IsUnitStride && is_tensor_view_v<Op>;
    static constexpr int OriginalRank = Op::Rank();
    static constexpr int Rank = OriginalRank - NumBound;
    static_assert(Rank >= 0, "BoundAccessor: bound more dims than the op has");
    using value_type = typename Op::value_type;

    static constexpr int NumRemOuter = (Rank >= 2) ? (Rank - 1) : 0;
    static constexpr int NumRemOuterStored = (NumRemOuter > 0) ? NumRemOuter : 1;

    const Op& op_;
    data_ptr_of_op_t<Op> base_;              // fast path: already advanced past bound coords
    IdxT rem_outer_strides_[NumRemOuterStored];
    index_t bound_idxs_[NumBound];           // slow path: forward to op(bound..., is...)

    // Construct from a parent TensorAccessor + leading index pack.
    template <typename... Leading>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    BoundAccessor(const TensorAccessor<Op, IsUnitStride, IdxT>& parent, Leading... leading)
        : op_(parent.op_), base_(nullptr)
    {
        static_assert(sizeof...(Leading) == NumBound, "Mismatched bind arity");
        const index_t leading_arr[] = { static_cast<index_t>(leading)... };

        MATX_LOOP_UNROLL
        for (int d = 0; d < NumBound; ++d) {
            bound_idxs_[d] = leading_arr[d];
        }

        if constexpr (FastPath) {
            base_ = parent.data_;
            IdxT off = 0;
            MATX_LOOP_UNROLL
            for (int d = 0; d < NumBound; ++d) {
                // parent.outer_strides_ only covers dims 0..OriginalRank-2.
                // The last dim's stride is implicitly 1 under IsUnitStride
                // and is not stored, so bind of that dim must use 1 directly
                // rather than read past the populated slots.
                const IdxT stride_d = (d < OriginalRank - 1)
                    ? parent.outer_strides_[d]
                    : static_cast<IdxT>(1);
                off += static_cast<IdxT>(leading_arr[d]) * stride_d;
            }
            base_ += off;
            if constexpr (NumRemOuter > 0) {
                MATX_LOOP_UNROLL
                for (int d = 0; d < NumRemOuter; ++d) {
                    rem_outer_strides_[d] = parent.outer_strides_[NumBound + d];
                }
            }
        }
    }

    // Rank-0 remainder: nullary.
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    decltype(auto) operator()() const requires (Rank == 0) {
        if constexpr (FastPath) {
            return base_[0];
        } else {
            return slow_forward(cuda::std::make_index_sequence<NumBound>{});
        }
    }

    // Rank >= 1 remainder: variadic, arity == Rank.
    template <typename... Is>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    decltype(auto) operator()(Is... is) const
        requires (Rank >= 1 && sizeof...(Is) == Rank)
    {
        if constexpr (FastPath) {
            const IdxT idx_arr[] = { static_cast<IdxT>(is)... };
            IdxT offset = idx_arr[Rank - 1];
            if constexpr (NumRemOuter > 0) {
                MATX_LOOP_UNROLL
                for (int d = 0; d < NumRemOuter; ++d) {
                    offset += idx_arr[d] * rem_outer_strides_[d];
                }
            }
            return base_[offset];
        } else {
            return slow_forward(cuda::std::make_index_sequence<NumBound>{}, is...);
        }
    }

private:
    template <size_t... BoundIs, typename... NewIs>
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
    decltype(auto) slow_forward(cuda::std::index_sequence<BoundIs...>, NewIs... new_is) const {
        return op_(bound_idxs_[BoundIs]..., new_is...);
    }
};

// ----------------------------------------------------------------------------
// bind_first_n<N>(acc, arr)
//
// Bind the first N entries of a cuda::std::array-like index container as
// leading dims into acc. Returns a BoundAccessor when N > 0, or the accessor
// unchanged when N == 0. Used by kernels that receive their batch coords
// through BlockToIdx() and need to fold them into an accessor in one shot.
// ----------------------------------------------------------------------------
template <typename Acc, typename ArrT, size_t... Is>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
auto bind_first_n_impl(const Acc& acc, const ArrT& arr, cuda::std::index_sequence<Is...>) {
    return acc.bind(arr[Is]...);
}

template <int N, typename Acc, typename ArrT>
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__
auto bind_first_n(const Acc& acc, const ArrT& arr) {
    if constexpr (N == 0) {
        return acc;
    } else {
        return bind_first_n_impl(acc, arr, cuda::std::make_index_sequence<N>{});
    }
}

}  // namespace detail
}  // namespace matx
