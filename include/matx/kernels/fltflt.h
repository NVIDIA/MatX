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
};

static __host__ __device__ __forceinline__ fltflt fltflt_make_from_double(double x) {
    float hi = (float)x;
    float lo = (float)(x - (double)hi);
    return fltflt{ hi, lo };
}

static __host__ __device__ __forceinline__ fltflt fltflt_make_from_float(float x) {
    return fltflt{ x, 0.0f };
}

static __host__ __device__ __forceinline__ double fltflt_to_double(fltflt x) {
    return (double)x.hi + (double)x.lo;
}

static __host__ __device__ __forceinline__ float fltflt_to_float(fltflt x) {
    return x.hi;
}

#ifdef __CUDACC__

// fltflt_two_sum is the Two-Sum algorithm given by Thall, which he attributes to Knuth.
// This corresponds to function twoSum() from Thall's paper, which implements Algorithm 2.
// This algorithm produces a normalized (non-overlapping) expansion.
static __device__ __forceinline__ fltflt fltflt_two_sum(float a, float b) {
    const float s = __fadd_rn(a, b);
    const float v = __fsub_rn(s, a);
    const float e = __fadd_rn(
        __fsub_rn(a,__fsub_rn(s, v)),
        __fsub_rn(b, v));
    return { s, e };
}

// fltflt_fast_two_sum is the Fast-Two-Sum algorithm given by Thall, which he attributes
// to Dekker. This corresponds to function quickTwoSum() from Thall's paper, which
// implements Algorithm 3. This algorithm produces a normalized (non-overlapping) expansion,
// but unlike fltflt_two_sum, it assumes that |a| >= |b|.
static __device__ __forceinline__ fltflt fltflt_fast_two_sum(float a, float b) {
    const float s = __fadd_rn(a, b);
    const float e = __fsub_rn(b, __fsub_rn(s, a));
    return { s, e };
}

// fltflt_two_prod_fma is the Two-Product-FMA algorithm given by Thall, which he attributes
// to Hida. This corresponds to function FMA-twoProd() from Thall's paper, which
// implements Algorithm 5. This algorithm produces a normalized (non-overlapping) expansion
// using a fused multiply-add operation.
static __device__ __forceinline__ fltflt fltflt_two_prod_fma(float a, float b) {
    const float x = __fmul_rn(a, b);
    const float y = __fmaf_rn(a, b, -x);
    return { x, y };
}

// fltflt_add is the df64_add() function given by Thall. This function uses two_sum()
// for the hi and lo components followed by addition of the cross terms and
// re-normalization to a non-overlapping expansion.
static __device__ __forceinline__ fltflt fltflt_add(fltflt a, fltflt b) {
    fltflt s = fltflt_two_sum(a.hi, b.hi);
    const fltflt t = fltflt_two_sum(a.lo, b.lo);
    s.lo = __fadd_rn(s.lo, t.hi);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    s.lo = __fadd_rn(s.lo, t.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    return s;
}

// fltflt_add_float is an optimization of df64_add() for the case where b is
// a float, and thus b.lo is zero.
static __device__ __forceinline__ fltflt fltflt_add_float(fltflt a, float b) {
    fltflt s = fltflt_two_sum(a.hi, b);
    s.lo = __fadd_rn(s.lo, a.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    return s;
}

// fltflt_sub() subtracts b from a. It delegates to fltflt_add() with a negated b.
static __device__ __forceinline__ fltflt fltflt_sub(fltflt a, fltflt b) {
    const fltflt neg_b = { -b.hi, -b.lo };
    return fltflt_add(a, neg_b);
}

// fltflt_sub_float() is an optimization of fltflt_sub() for the case where b is
// a float, and thus b.lo is zero. It delegates to fltflt_add_float() with a negated b.
static __device__ __forceinline__ fltflt fltflt_sub_float(fltflt a, float b) {
    return fltflt_add_float(a, -b);
}

// fltflt_mul() is the df64_mult() function given by Thall. This function uses the
// two_prod_fma() function for the hi components followed by addition of the cross terms
// and re-normalization to a non-overlapping expansion.
static __device__ __forceinline__ fltflt fltflt_mul(fltflt a, fltflt b) {
    fltflt p;
    p = fltflt_two_prod_fma(a.hi, b.hi);
    p.lo = __fmaf_rn(a.hi, b.lo, p.lo);
    p.lo = __fmaf_rn(a.lo, b.hi, p.lo);
    p = fltflt_fast_two_sum(p.hi, p.lo);
    return p;
}

// fltflt_sqrt() is the df64_sqrt() function given by Thall, which he attributes to Karp.
// This function implements Algorithm 7 from Thall's paper. It uses the
// two_prod_fma() function for the hi components followed by subtraction of the square
// of the result and re-normalization to a non-overlapping expansion.
static __device__ __forceinline__ fltflt fltflt_sqrt(fltflt a) {
    const float xn = (a.hi == 0.0f) ? 0.0f :rsqrtf(a.hi);
    const float yn = __fmul_rn(a.hi, xn);
    const fltflt ynsqr = fltflt_two_prod_fma(yn, yn);
    const fltflt diff = fltflt_sub(a, ynsqr);
    fltflt prod = fltflt_two_prod_fma(xn, 0.5f * diff.hi);
    return fltflt_add_float(prod, yn);
}

#endif // defined(__CUDACC__)

} // namespace matx