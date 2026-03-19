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

#include <cmath>

namespace matx {

// This header implements a float-float type (fltflt) that uses two single-precision floating
// point values to represent a higher-precision value. When normalized, the components of the
// float-float representation are non-overlapping and the hi component is larger in magnitude
// than the lo component. Because floats are used to represent both components, only the
// mantissa is effectively increased -- the number of exponent bits, and thus the dynamic range
// represented by a float-float value, is the same as a single-precision floating point value.
// The primary reference followed for the implementation in this file is:
//   "Extended-Precision Floating-Point Numbers for GPU Computation", Andrew Thall,
//   https://andrewthall.org/papers/df64_qf128.pdf
// That paper cites key work from D. E. Knuth, T. J. Dekker, A. H. Karp and others.

// fltflt represents an unevaluated floating point sum of two non-overlapping fp32 components.
// The hi component is the most significant part of the sum, and the lo component is the least significant part.
struct alignas(8) fltflt {
    float hi;
    float lo;

    // The default constructor does not initialize the components, so the value is indeterminate.
    __MATX_INLINE__ fltflt() = default;
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit fltflt(double x)
        : hi(static_cast<float>(x)), lo(static_cast<float>(x - static_cast<double>(hi))) {}
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit fltflt(float x) : hi(x), lo(0.0f) {}
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit fltflt(float hi_, float lo_) : hi(hi_), lo(lo_) {}
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit operator double() const {
        return static_cast<double>(hi) + static_cast<double>(lo);
    }
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit operator float() const { return hi; }
};

// The constructors and conversion operators in the fltflt struct allow conversion to double and float
// via static_cast<double>(fltflt_val) and similar for float. The fltflt_to_* functions are provided for completeness.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ double fltflt_to_double(fltflt x) {
    return static_cast<double>(x.hi) + static_cast<double>(x.lo);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fltflt_to_float(fltflt x) {
    return static_cast<float>(x);
}

// The fltflt_make* functions are provided for completeness, but users can directly use
// static_cast<fltflt>() as well
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_make_from_double(double x) {
    return static_cast<fltflt>(x);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_make_from_float(float x) {
    return static_cast<fltflt>(x);
}

namespace detail {
// Provide host/device wrappers for CUDA round-to-nearest intrinsics. On host, we fall back
// to standard operations. These helpers allow fltflt arithmetic to be callable from host
// code in a .cu translation unit (NVCC host pass), while still using fast intrinsics in
// device code.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fadd_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fsub_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fmul_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fmaf_rn(float a, float b, float c)
{
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    // Use fmaf on host for better precision when available.
    return ::fmaf(a, b, c);
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fdividef_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    // In many cases, __fdividef() is sufficient (e.g., when computing an initial estimate
    // for a Newton-Raphson iteration). However, for now we stick with the more accurate
    // __fdiv_rn() intrinsic until we have more robust error bounds.
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fltflt_rsqrt(float x)
{
#if defined(__CUDA_ARCH__)
    // rsqrtf has up to 2 ULP of error. This is less precise than 1.0f / ::sqrtf(x), which
    // would be 0.5 ULP of error. We currently use rsqrtf() because it is significantly faster
    // while maintaining 44+ bits of precision in testing thus far, but we may need to revisit
    // this in the future.
    return rsqrtf(x);
#else
    return 1.0f / ::sqrtf(x);
#endif
}
} // namespace detail

// fltflt_two_sum is the Two-Sum algorithm given by Thall, which he attributes to Knuth.
// This corresponds to function twoSum() from Thall's paper, which implements Algorithm 2.
// This algorithm produces a normalized (non-overlapping) expansion.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_two_sum(float a, float b) {
    const float s = detail::fadd_rn(a, b);
    const float v = detail::fsub_rn(s, a);
    const float e = detail::fadd_rn(
        detail::fsub_rn(a, detail::fsub_rn(s, v)),
        detail::fsub_rn(b, v));
    return fltflt{ s, e };
}

// fltflt_fast_two_sum is the Fast-Two-Sum algorithm given by Thall, which he attributes
// to Dekker. This corresponds to function quickTwoSum() from Thall's paper, which
// implements Algorithm 3. This algorithm produces a normalized (non-overlapping) expansion,
// but unlike fltflt_two_sum, it assumes that |a| >= |b|.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fast_two_sum(float a, float b) {
    const float s = detail::fadd_rn(a, b);
    const float e = detail::fsub_rn(b, detail::fsub_rn(s, a));
    return fltflt{ s, e };
}

// fltflt_two_prod_fma is the Two-Product-FMA algorithm given by Thall, which he attributes
// to Hida. This corresponds to function FMA-twoProd() from Thall's paper, which
// implements Algorithm 5. This algorithm produces a normalized (non-overlapping) expansion
// using a fused multiply-add operation.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_two_prod_fma(float a, float b) {
    const float x = detail::fmul_rn(a, b);
    const float y = detail::fmaf_rn(a, b, -x);
    return fltflt{ x, y };
}

// fltflt_add is the df64_add() function given by Thall. This function uses two_sum()
// for the hi and lo components followed by addition of the cross terms and
// re-normalization to a non-overlapping expansion.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add(fltflt a, fltflt b) {
    fltflt s = fltflt_two_sum(a.hi, b.hi);
    const fltflt t = fltflt_two_sum(a.lo, b.lo);
    s.lo = detail::fadd_rn(s.lo, t.hi);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    s.lo = detail::fadd_rn(s.lo, t.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    return s;
}

// This overload is an optimization of fltflt_add() for the case where b is
// a float, and thus b.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add(fltflt a, float b) {
    fltflt s = fltflt_two_sum(a.hi, b);
    s.lo = detail::fadd_rn(s.lo, a.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    return s;
}

// This overload is an optimization of fltflt_add() for the case where a is
// a float, and thus b.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add(float a, fltflt b) {
    return fltflt_add(b, a);
}

// fltflt_add_same_sign() is an optimized version of fltflt_add() suitable for cases where a
// and b have the same sign. This version uses 11 FLOPs versus 20 FLOPs for the more general
// fltflt_add(). This implementation corresponds to the original version from Dekker and is
// given in Algorithm 14.1 of "Handbook of Floating-Point Arithmetic" by Muller et al. Rather
// than include a conditional on the magnitude of a and b to use fltflt_fast_two_sum(), we
// use fltflt_two_sum() at the cost of more FLOPs but without a branch.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add_same_sign(fltflt a, fltflt b) {
    const fltflt r = fltflt_two_sum(a.hi, b.hi);
    const float s = detail::fadd_rn(detail::fadd_rn(r.lo, b.lo), a.lo);
    return fltflt_fast_two_sum(r.hi, s);
}

// This overload is an optimization of fltflt_add_same_sign() for the case where b is
// a float, and thus b.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add_same_sign(fltflt a, float b) {
    const fltflt r = fltflt_two_sum(a.hi, b);
    const float s = detail::fadd_rn(r.lo, a.lo);
    return fltflt_fast_two_sum(r.hi, s);
}

// This overload is an optimization of fltflt_add_same_sign() for the case where a is
// a float, and thus a.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add_same_sign(float a, fltflt b) {
    return fltflt_add_same_sign(b, a);
}

// fltflt_sub() subtracts b from a. It delegates to fltflt_add() with a negated b.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(fltflt a, fltflt b) {
    const fltflt neg_b = fltflt{ -b.hi, -b.lo };
    return fltflt_add(a, neg_b);
}

// This overload is an optimization of fltflt_sub() for the case where b is
// a float, and thus b.lo is zero. It delegates to fltflt_add() with a negated b.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(fltflt a, float b) {
    return fltflt_add(a, -b);
}

// This overload is an optimization of fltflt_sub() for the case where a is
// a float, and thus a.lo is zero. It delegates to fltflt_add() with a negated b.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(float a, fltflt b) {
    return fltflt_add(fltflt{ -b.hi, -b.lo }, a);
}

// fltflt_mul() is the df64_mult() function given by Thall. This function uses the
// two_prod_fma() function for the hi components followed by addition of the cross terms
// and re-normalization to a non-overlapping expansion.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul(fltflt a, fltflt b) {
    fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
    p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
    p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);
    p = fltflt_fast_two_sum(p.hi, p.lo);
    return p;
}

// This overload is an optimization of fltflt_mul() for the case where b is
// a float, and thus b.lo is zero. This function uses one fewer fmaf_rn() operation.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul(fltflt a, float b) {
    fltflt p = fltflt_two_prod_fma(a.hi, b);
    p.lo = detail::fmaf_rn(a.lo, b, p.lo);
    p = fltflt_fast_two_sum(p.hi, p.lo);
    return p;
}

// This overload is an optimization of fltflt_mul() for the case where a is
// a float, and thus a.lo is zero. This function uses one fewer fmaf_rn() operation.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul(float a, fltflt b) {
    return fltflt_mul(b, a);
}

// fltflt_fma() computes a * b + c with two normalizations.
// This is more efficient than fltflt_add(fltflt_mul(a, b), c), which uses three normalizations.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, fltflt b, fltflt c) {
    // The first three operations match fltflt_mul()
    fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
    p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
    p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);

    // fltflt_mul() renormalizes at this point using fltflt_fast_two_sum(p.hi, p.lo), but
    // we skip that step, add the c hi component, add the p.lo component, renormalize,
    // and finally add the c.lo component.

    fltflt s = fltflt_two_sum(p.hi, c.hi);
    s.lo = detail::fadd_rn(s.lo, p.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    s.lo = detail::fadd_rn(s.lo, c.lo);

    // Single final normalization
    s = fltflt_fast_two_sum(s.hi, s.lo);

    return s;
}

// A version of fltflt_fma() where c is a float. This is slightly more efficient than
// fltflt_fma(a, b, fltflt{ c, 0.0f }).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, fltflt b, float c) {
    // The first three operations match fltflt_mul()
    fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
    p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
    p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);

    // fltflt_mul() renormalizes at this point using fltflt_fast_two_sum(p.hi, p.lo), but
    // we skip that step, add c, and then add the p.lo component.

    fltflt s = fltflt_two_sum(p.hi, c);
    s.lo = detail::fadd_rn(s.lo, p.lo);

    // Single final normalization
    s = fltflt_fast_two_sum(s.hi, s.lo);

    return s;
}

// A version of fltflt_fma() where a is a float. This is slightly more efficient than
// using fltflt_fma(fltflt{ a, 0.0f }, b, c).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(float a, fltflt b, fltflt c) {
    // The first three operations match fltflt_mul()
    fltflt p = fltflt_two_prod_fma(a, b.hi);
    p.lo = detail::fmaf_rn(a, b.lo, p.lo);

    // fltflt_mul() renormalizes at this point using fltflt_fast_two_sum(p.hi, p.lo), but
    // we skip that step, add the c hi component, add the p.lo component, renormalize,
    // and finally add the c.lo component.
    fltflt s = fltflt_two_sum(p.hi, c.hi);
    s.lo = detail::fadd_rn(s.lo, p.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    s.lo = detail::fadd_rn(s.lo, c.lo);

    // Single final normalization
    s = fltflt_fast_two_sum(s.hi, s.lo);

    return s;
}

// A version of fltflt_fma() where b is a float. This is slightly more efficient than
// using fltflt_fma(a, fltflt{ b, 0.0f }, c).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, float b, fltflt c) {
    return fltflt_fma(b, a, c);
}

// A version of fltflt_fma() where b and c are floats. This is more efficient than
// using fltflt_fma(a, fltflt{ b, 0.0f }, fltflt{ c, 0.0f }).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(fltflt a, float b, float c) {
    // The first three operations match fltflt_mul()
    fltflt p = fltflt_two_prod_fma(a.hi, b);
    p.lo = detail::fmaf_rn(a.lo, b, p.lo);

    // fltflt_mul() renormalizes at this point using fltflt_fast_two_sum(p.hi, p.lo), but
    // we skip that step, add the c hi component, add the p.lo component, renormalize,
    // and finally add the c.lo component.

    fltflt s = fltflt_two_sum(p.hi, c);
    s.lo = detail::fadd_rn(s.lo, p.lo);

    // Single final normalization
    s = fltflt_fast_two_sum(s.hi, s.lo);

    return s;
}

// A version of fltflt_fma() where a and c are floats. This is more efficient than
// using fltflt_fma(fltflt{ a, 0.0f }, b, fltflt{ c, 0.0f }).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(float a, fltflt b, float c) {
    return fltflt_fma(b, a, c);
}

// fltflt_div() is the df64_div() function given by Thall, which he attributes to Karp.
// This function implements Algorithm 6 from Thall's paper. For the initial approximation,
// we use the __fdividef() intrinsic on the device.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(fltflt a, fltflt b) {
    const float xn = (b.hi == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt diff = fltflt_sub(a, fltflt_mul(b, yn));
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add(prod, yn);
}

// This overload is an optimization of fltflt_div() for the case where b is
// a float, and thus b.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(fltflt a, float b) {
    const float xn = (b == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt diff = fltflt_sub(a, fltflt_two_prod_fma(b, yn));
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add(prod, yn);
}

// This overload is an optimization of fltflt_div() for the case where a is
// a float, and thus a.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(float a, fltflt b) {
    const float xn = (b.hi == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
    const float yn = detail::fmul_rn(a, xn);
    const fltflt diff = fltflt_sub(a, fltflt_mul(b, yn));
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add(prod, yn);
}

// fltflt_round_to_nearest() rounds a float-float value to the nearest integer with
// ties rounded toward even. Note that rounding "toward even" means that, in the case
// of a tie, the least significant bit of the mantissa is set to 0 (i.e., we round to
// the even significand).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_round_to_nearest(fltflt a) {
    constexpr float FAST_PATH_THRESHOLD = 8388608.0f;
    if (fabsf(a.hi) < FAST_PATH_THRESHOLD) {
        const float candidate = nearbyintf(a.hi);

        const float err = detail::fsub_rn(a.hi, candidate);

        if (fabsf(err) != 0.5f) {
            return fltflt{ candidate, 0.0f };
        } else {
            // We should not have errors > 0.5 ulp(a.hi). Since ulp is at most 1, the max error should
            // be 0.5 for the boundary case.
            fltflt result{ candidate, 0.0f };
            if (a.lo == 0.0f) {
                // Perfect tie, round to even
                const float corrected = (fmodf(candidate, 2.0f) == 0.0f) ? candidate : candidate + copysignf(1.0f, err);
                result.hi = corrected;
            } else if ((err > 0 && a.lo > 0) || (err < 0 && a.lo < 0)) {
                result.hi = detail::fadd_rn(candidate, copysignf(1.0f, err));
            }
            // We do not need to renormalize because we know the full integral part fits
            // exactly in hi due to the original magnitude check.
            return result;
        }
    } else { // |a.hi| >= 2^23, so a.hi is an integer
        float r_lo = nearbyintf(a.lo);
        const float frac = detail::fsub_rn(a.lo, r_lo);

        if (fabsf(frac) > 0.5f) {
            r_lo = detail::fadd_rn(r_lo, copysignf(1.0f, frac));
        } else if (fabsf(frac) == 0.5f) {
            // Check if hi + r_lo would be odd (sum is odd iff parities differ)
            bool hi_is_odd = (fmodf(a.hi, 2.0f) != 0.0f);
            bool rlo_is_odd = (fmodf(r_lo, 2.0f) != 0.0f);
            if (hi_is_odd != rlo_is_odd) {  // XOR of parities
                r_lo = detail::fadd_rn(r_lo, copysignf(1.0f, frac));
            }
        }

        // Renormalize
        return fltflt_fast_two_sum(a.hi, r_lo);
    }
}

// fltflt_round_toward_zero() truncates a fltflt value to an integer with the
// result being truncated toward zero (vs floor, which will truncate toward
// negative infinity).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_round_toward_zero(fltflt a) {
    if (fabsf(a.hi) < 8388608.0f) { // |a.hi| < 2^23, so a.hi is not an integer
        const float hi_trunc = truncf(a.hi);
        // If hi is exactly an integer, then lo can cause a boundary crossing
        if (hi_trunc == a.hi) {
            // If hi is 1.0 and lo is -1e-9, value is 0.999... -> trunc to 0.0
            // This happens when signs are opposite.
            if ((a.hi > 0.0f && a.lo < 0.0f) || (a.hi < 0.0f && a.lo > 0.0f)) {
                // Pull toward zero by 1 unit
                return fltflt{ a.hi + (a.hi > 0.0f ? -1.0f : 1.0f), 0.0f };
            } else {
                // Signs match or lo is 0: truncation is just hi. Fallthrough case.
            }
        }
        return fltflt{ hi_trunc, 0.0f };
    } else { // |a.hi| >= 2^23, so a.hi is an integer
        float lo_trunc = truncf(a.lo);
        if (lo_trunc != a.lo) { // lo has a fractional part, so we may need a correction
            // If lo is opposite sign of hi,
            // the fractional part nudges us across an integer boundary.
            if ((a.hi > 0.0f && a.lo < 0.0f) || (a.hi < 0.0f && a.lo > 0.0f)) {
                // If hi=pos, lo=neg (e.g., 10, -0.5), we need 9.
                // If hi=neg, lo=pos (e.g., -10, 0.5), we need -9.
                const float adj = (a.hi > 0.0f) ? -1.0f : 1.0f;
                lo_trunc = lo_trunc + adj;
            }
        }
        return fltflt_fast_two_sum(a.hi, lo_trunc);
    }
}

// fltflt_floor() returns an integer truncated toward negative infinity. This is
// the largest integer that is not larger than the value a.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_floor(fltflt a) {
    if (fabsf(a.hi) < 8388608.0f) { // |a.hi| < 2^23, so a.hi might not be an integer
        const float hi_floor = floorf(a.hi);
        // If hi was exactly an integer and lo is negative,
        // the actual value is just below hi, so floor should be hi - 1
        if (hi_floor == a.hi && a.lo < 0.0f) {
            return fltflt{ a.hi - 1.0f, 0.0f };
        }
        return fltflt{ hi_floor, 0.0f };
    } else { // |a.hi| >= 2^23, so a.hi is already an integer
        const float lo_floor = floorf(a.lo);
        // Renormalize the result
        return fltflt_fast_two_sum(a.hi, lo_floor);
    }
}

// fltflt_sqrt() is the df64_sqrt() function given by Thall, which he attributes to Karp.
// This function implements Algorithm 7 from Thall's paper. It uses the
// two_prod_fma() function for the hi components followed by subtraction of the square
// of the result and re-normalization to a non-overlapping expansion.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sqrt(fltflt a) {
    const float xn = (a.hi == 0.0f) ? 0.0f : detail::fltflt_rsqrt(a.hi);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt ynsqr = fltflt_two_prod_fma(yn, yn);
    const fltflt diff = fltflt_sub(a, ynsqr);
    fltflt prod = fltflt_two_prod_fma(xn, 0.5f * diff.hi);
    return fltflt_add(prod, yn);
}

// fltflt_sqrt_fast() is a faster approximation of fltflt_sqrt() that uses a single FMA to
// compute the residual a - yn^2 instead of full fltflt subtraction. The FMA computes
// a.hi - yn*yn exactly (exact multiply, single rounding), and adding a.lo recovers the
// input's low-order bits. The result has precision comparable to fltflt_sqrt for most
// values at roughly 1/5 the cost (~7 FLOPs vs ~35+). We do see differences for some
// inputs. For example, for 1e9*pi + sqrt(2), fltflt_sqrt() matches the fp64
// baseline in all mantissa bits and fltflt_sqrt_fast() matches the first 45 mantissa bits.
// This function may eventually become the default sqrt() implementation.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sqrt_fast(fltflt a) {
    const float xn = (a.hi == 0.0f) ? 0.0f : detail::fltflt_rsqrt(a.hi);
    const float yn = detail::fmul_rn(a.hi, xn);
    const float residual = detail::fadd_rn(
        detail::fmaf_rn(-yn, yn, a.hi), a.lo);
    const float correction = detail::fmul_rn(
        detail::fmul_rn(xn, 0.5f), residual);
    return fltflt_fast_two_sum(yn, correction);
}

// fltflt_norm3d() computes sqrt(dx^2 + dy^2 + dz^2) with minimal intermediate
// normalizations. Instead of the separate fltflt_mul + fltflt_fma + fltflt_fma + fltflt_sqrt_fast
// chain (5 normalizations, ~50 ops), this function computes all three exact squares,
// accumulates with a single normalization, and applies fltflt_sqrt_fast (~39 ops).
// The three inputs are assumed to be normalized fltflt values.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_norm3d(fltflt dx, fltflt dy, fltflt dz) {
    // Exact squares of hi components (each captures full rounding error)
    const fltflt px = fltflt_two_prod_fma(dx.hi, dx.hi);
    const fltflt py = fltflt_two_prod_fma(dy.hi, dy.hi);
    const fltflt pz = fltflt_two_prod_fma(dz.hi, dz.hi);

    // Sum the three .hi values using two_sum to capture rounding errors
    const fltflt s = fltflt_two_sum(px.hi, py.hi);
    const fltflt t = fltflt_two_sum(s.hi, pz.hi);

    // Accumulate all eight low-order terms into a single float:
    //   - two_sum rounding errors: s.lo, t.lo
    //   - two_prod_fma error terms: px.lo, py.lo, pz.lo
    //   - cross terms from squaring: 2*dx.hi*dx.lo, 2*dy.hi*dy.lo, 2*dz.hi*dz.lo
    // All terms are O(eps) relative to t.hi, so their sum is at most 8*eps*|t.hi|.
    // This may result in slight precision loss due to potential overlap between
    // lo and t.hi, but this should still be valid for ~44 bits prior to the sqrt.
    float lo = detail::fadd_rn(t.lo, s.lo);
    lo = detail::fadd_rn(lo, px.lo);
    lo = detail::fadd_rn(lo, py.lo);
    lo = detail::fadd_rn(lo, pz.lo);
    lo = detail::fmaf_rn(detail::fadd_rn(dx.hi, dx.hi), dx.lo, lo);
    lo = detail::fmaf_rn(detail::fadd_rn(dy.hi, dy.hi), dy.lo, lo);
    lo = detail::fmaf_rn(detail::fadd_rn(dz.hi, dz.hi), dz.lo, lo);

    // Single normalization before sqrt
    const fltflt sum_sq = fltflt_fast_two_sum(t.hi, lo);

    return fltflt_sqrt_fast(sum_sq);
}

// Scalar sqrt overload so unary operator dispatch can handle fltflt expressions
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt sqrt(fltflt a) { return fltflt_sqrt(a); }

// fltflt_abs() returns the absolute value of a. This function assumes that a is normalized
// and thus that the sign of the hi component is the same as the sign of the value. If the
// value is not normalized, then it is possible for a.hi to be 0 and thus the sign of the value
// is the sign of the lo component. We do not handle this case in this implementation, but
// in implementations that allow non-normalized values, it should either be added or the
// value should be re-normalized prior to calling this function.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_abs(fltflt a) {
    if (a.hi < 0.0f) {
        return fltflt{ -a.hi, -a.lo };
    }
    return a;
}

// Scalar abs overload so unary operator dispatch can handle fltflt expressions
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt abs(fltflt a) { return fltflt_abs(a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(fltflt a, fltflt b) { return fltflt_add(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(fltflt a, float b) { return fltflt_add(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(float a, fltflt b) { return fltflt_add(b, a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a, fltflt b) { return fltflt_sub(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a, float b) { return fltflt_sub(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(float a, fltflt b) { return fltflt_sub(a, b); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a) { return fltflt{ -a.hi, -a.lo }; }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, fltflt b) { return fltflt_mul(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, float b) { return fltflt_mul(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(float a, fltflt b) { return fltflt_mul(b, a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, fltflt b) { return fltflt_div(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, float b) { return fltflt_div(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(float a, fltflt b) { return fltflt_div(a, b); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(fltflt a, fltflt b) { return a.hi == b.hi && a.lo == b.lo; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(fltflt a, float b) { return a.hi == b && a.lo == 0.0f; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(float a, fltflt b) { return b.hi == a && b.lo == 0.0f; }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(fltflt a, fltflt b) { return !(a == b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(fltflt a, float b) { return !(a == b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(float a, fltflt b) { return !(a == b); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(fltflt a, fltflt b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(fltflt a, float b) { return a.hi < b || (a.hi == b && a.lo < 0.0f); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(float a, fltflt b) { return a < b.hi || (a == b.hi && b.lo > 0.0f); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(fltflt a, fltflt b) { return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(fltflt a, float b) { return a.hi > b || (a.hi == b && a.lo > 0.0f); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(float a, fltflt b) { return a > b.hi || (a == b.hi && b.lo < 0.0f); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(fltflt a, fltflt b) { return a < b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(fltflt a, float b) { return a < b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(float a, fltflt b) { return a < b || a == b; }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(fltflt a, fltflt b) { return a > b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(fltflt a, float b) { return a > b || a == b; }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(float a, fltflt b) { return a > b || a == b; }

// fltflt_fmod() computes the floating-point remainder of division. In other words,
// fltflt_fmod(a, b) = a - n * b where n = trunc(a/b) and trunc() truncates to an integer
// toward zero. If b is zero, returns {NaN, NaN}.
//
// Note: For very large quotients (|a/b| > 10^6), precision is limited by the fltflt type's
// absolute precision at large magnitudes (~|value| * 2^-44). The current implementation does
// not use range reduction or similar techniques to improve precision in these cases, but a
// future implementation may do so.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fmod(fltflt a, fltflt b) {
    if (b.hi == 0.0f && b.lo == 0.0f) {
        return fltflt{cuda::std::numeric_limits<float>::quiet_NaN(),
                      cuda::std::numeric_limits<float>::quiet_NaN()};
    }

    float sign = 1.0f;
    if (a < 0.0f) {
        sign = -1.0f;
        a = -a;
    }
    b = fltflt_abs(b);

    const fltflt q = fltflt_div(a, b);
    const fltflt trunc_q = fltflt_round_toward_zero(q);
    fltflt result = -fltflt_fma(trunc_q, b, -a);

    while (result >= b) {
        result = fltflt_sub(result, b);
    }
    while (result < 0.0f) {
        result = fltflt_add(result, b);
    }

    return fltflt{ sign * result.hi, sign * result.lo };
}

// fltflt_fmod() overload where b is a float.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fmod(fltflt a, float b) {
    if (b == 0.0f) {
        return fltflt{cuda::std::numeric_limits<float>::quiet_NaN(),
                      cuda::std::numeric_limits<float>::quiet_NaN()};
    }

    float sign = 1.0f;
    if (a < 0.0f) {
        sign = -1.0f;
        a = -a;
    }
    b = fabsf(b);

    const fltflt q = fltflt_div(a, b);
    const fltflt trunc_q = fltflt_round_toward_zero(q);
    fltflt result = -fltflt_fma(trunc_q, b, -a);

    while (result >= b) {
        result = fltflt_sub(result, b);
    }
    while (result < 0.0f) {
        result = fltflt_add(result, b);
    }

    return fltflt{ sign * result.hi, sign * result.lo };
}

} // namespace matx
