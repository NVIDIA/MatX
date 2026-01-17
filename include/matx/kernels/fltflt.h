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
struct fltflt {
    float hi;
    float lo;

    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ explicit operator double() const;
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ explicit operator float() const;
};

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_make_from_double(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return fltflt{ hi, lo };
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_make_from_float(float x) {
    return fltflt{ x, 0.0f };
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ double fltflt_to_double(fltflt x) {
    return (double)x.hi + (double)x.lo;
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fltflt_to_float(fltflt x) {
    return x.hi;
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
    return __fdividef(a, b);
#else
    return a / b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fltflt_rsqrt(float x)
{
#if defined(__CUDA_ARCH__)
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
    return { s, e };
}

// fltflt_fast_two_sum is the Fast-Two-Sum algorithm given by Thall, which he attributes
// to Dekker. This corresponds to function quickTwoSum() from Thall's paper, which
// implements Algorithm 3. This algorithm produces a normalized (non-overlapping) expansion,
// but unlike fltflt_two_sum, it assumes that |a| >= |b|.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fast_two_sum(float a, float b) {
    const float s = detail::fadd_rn(a, b);
    const float e = detail::fsub_rn(b, detail::fsub_rn(s, a));
    return { s, e };
}

// fltflt_two_prod_fma is the Two-Product-FMA algorithm given by Thall, which he attributes
// to Hida. This corresponds to function FMA-twoProd() from Thall's paper, which
// implements Algorithm 5. This algorithm produces a normalized (non-overlapping) expansion
// using a fused multiply-add operation.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_two_prod_fma(float a, float b) {
    const float x = detail::fmul_rn(a, b);
    const float y = detail::fmaf_rn(a, b, -x);
    return { x, y };
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

// fltflt_add_float is an optimization of df64_add() for the case where b is
// a float, and thus b.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add_float(fltflt a, float b) {
    fltflt s = fltflt_two_sum(a.hi, b);
    s.lo = detail::fadd_rn(s.lo, a.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    return s;
}

// fltflt_sub() subtracts b from a. It delegates to fltflt_add() with a negated b.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(fltflt a, fltflt b) {
    const fltflt neg_b = { -b.hi, -b.lo };
    return fltflt_add(a, neg_b);
}

// fltflt_sub_float() is an optimization of fltflt_sub() for the case where b is
// a float, and thus b.lo is zero. It delegates to fltflt_add_float() with a negated b.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub_float(fltflt a, float b) {
    return fltflt_add_float(a, -b);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub_float(float a, fltflt b) {
    return fltflt_add_float({-b.hi, -b.lo}, a);
}

// fltflt_mul() is the df64_mult() function given by Thall. This function uses the
// two_prod_fma() function for the hi components followed by addition of the cross terms
// and re-normalization to a non-overlapping expansion.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul(fltflt a, fltflt b) {
    fltflt p;
    p = fltflt_two_prod_fma(a.hi, b.hi);
    p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
    p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);
    p = fltflt_fast_two_sum(p.hi, p.lo);
    return p;
}

// fltflt_mul_float() is an optimization of fltflt_mul() for the case where b is
// a float, and thus b.lo is zero. This function uses one fewer fmaf_rn() operation.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_mul_float(fltflt a, float b) {
    fltflt p;
    p = fltflt_two_prod_fma(a.hi, b);
    p.lo = detail::fmaf_rn(a.lo, b, p.lo);
    p = fltflt_fast_two_sum(p.hi, p.lo);
    return p;
}

// fltflt_div() is the df64_div() function given by Thall, which he attributes to Karp.
// This function implements Algorithm 6 from Thall's paper. For the initial approximation,
// we use the __fdividef() intrinsic on the device.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(fltflt a, fltflt b) {
    const float xn = (b.hi == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt diff = fltflt_sub(a, fltflt_mul_float(b, yn));
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add_float(prod, yn);
}

// fltflt_div_float_denom() is an optimization of fltflt_div() for the case where b is
// a float, and thus b.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div_float_denom(fltflt a, float b) {
    const float xn = (b == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt diff = fltflt_sub(a, fltflt_two_prod_fma(b, yn));
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add_float(prod, yn);
}

// fltflt_div_float_numer() is an optimization of fltflt_div() for the case where a is
// a float, and thus a.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div_float_numer(float a, fltflt b) {
    const float xn = (b.hi == 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
    const float yn = detail::fmul_rn(a, xn);
    const fltflt diff = fltflt_sub_float(a, fltflt_mul_float(b, yn));
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add_float(prod, yn);
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
    return fltflt_add_float(prod, yn);
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
        return { -a.hi, -a.lo };
    }
    return a;
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt::operator double() const {
    return fltflt_to_double(*this);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt::operator float() const {
    return fltflt_to_float(*this);
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(fltflt a, fltflt b) { return fltflt_add(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(fltflt a, float b) { return fltflt_add_float(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator+(float a, fltflt b) { return fltflt_add_float(b, a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a, fltflt b) { return fltflt_sub(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a, float b) { return fltflt_sub_float(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(float a, fltflt b) { return fltflt_sub_float(a, b); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a) { return { -a.hi, -a.lo }; }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, fltflt b) { return fltflt_mul(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, float b) { return fltflt_mul_float(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(float a, fltflt b) { return fltflt_mul_float(b, a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, fltflt b) { return fltflt_div(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, float b) { return fltflt_div_float_denom(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(float a, fltflt b) { return fltflt_div_float_numer(a, b); }

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

} // namespace matx