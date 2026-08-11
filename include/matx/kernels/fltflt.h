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
#include <cuda/std/bit>
#include <cuda/std/limits>

namespace matx {

namespace detail {
// CUDA intrinsics and operators acquire an .ftz modifier when compiled with
// --ftz=true (including through --use_fast_math). Float-float arithmetic
// requires gradual underflow. is_ftz_enabled() compares the smallest FP32
// subnormal with zero, which NVCC folds according to the active device FTZ mode.
// This allows the non-FTZ build to retain optimizer-visible intrinsics while the
// FTZ build selects explicit PTX instructions without .ftz. This is intentionally
// an ordinary condition: __uint_as_float is not constexpr, and constexpr bit casts
// that were tested always used standard C++ gradual-underflow semantics and failed
// to correctly determine the FTZ mode.
//
// Note that we do not handle FTZ detection or mitigation for host code. Host
// code implements FTZ via mode bits rather than per-instruction encodings, so
// we do not have a practical way to handle it here. Users of the host fltflt
// functions should investigate FTZ and DAZ handling on their platform.
#if defined(__CUDA_ARCH__)
static __MATX_DEVICE__ __MATX_INLINE__ bool is_ftz_enabled()
{
    return __uint_as_float(0x00000001U) == 0.0f;
}

// -G disables the optimization that folds the FTZ probe. It also removes the
// optimizer visibility that the intrinsic path preserves, so explicit PTX is
// the correct unconditional debug path in both FTZ modes.
#if defined(__clang__) && defined(__CUDA__)
// Our is_ftz_enabled() function does not properly detect ftz mode in Clang. Always use
// explicit PTX so gradual underflow is preserved with and without FTZ enabled.
#define MATX_FLTFLT_USE_PTX true
#elif defined(__CUDACC_DEBUG__)
#define MATX_FLTFLT_USE_PTX true
#else
#define MATX_FLTFLT_USE_PTX ::matx::detail::is_ftz_enabled()
#endif
#endif

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fadd_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
        return result;
    }
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fsub_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("sub.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
        return result;
    }
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fmul_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
        return result;
    }
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fmaf_rn(float a, float b, float c)
{
#if defined(__CUDA_ARCH__)
    // Unlike __fmaf_rn(), this intrinsic ignores the -ftz=true compiler flag.
    return __fmaf_ieee_rn(a, b, c);
#else
    // Use fmaf on host for better precision when available.
    return ::fmaf(a, b, c);
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fdividef_rn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
        return result;
    }
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fneg(float a)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("neg.f32 %0, %1;" : "=f"(result) : "f"(a));
        return result;
    }
    return -a;
#else
    return -a;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fabs_noftz(float a)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("abs.f32 %0, %1;" : "=f"(result) : "f"(a));
        return result;
    }
    return ::fabsf(a);
#else
    return ::fabsf(a);
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fcopysign(float magnitude, float sign)
{
    return ::copysignf(magnitude, sign);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ double float_to_double(float a)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        double result;
        asm("cvt.f64.f32 %0, %1;" : "=d"(result) : "f"(a));
        return result;
    }
    return static_cast<double>(a);
#else
    return static_cast<double>(a);
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float double_to_float_rn(double a)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(a));
        return result;
    }
    return __double2float_rn(a);
#else
    return static_cast<float>(a);
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fround_to_nearest(float a)
{
    return ::nearbyintf(a);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fround_toward_zero(float a)
{
    return ::truncf(a);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fround_down(float a)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("cvt.rmi.f32.f32 %0, %1;" : "=f"(result) : "f"(a));
        return result;
    }
    return ::floorf(a);
#else
    return ::floorf(a);
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool feq(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        unsigned int result;
        asm("{ .reg .pred p; setp.eq.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(result) : "f"(a), "f"(b));
        return result != 0U;
    }
    return a == b;
#else
    return a == b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool fne(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        unsigned int result;
        asm("{ .reg .pred p; setp.neu.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(result) : "f"(a), "f"(b));
        return result != 0U;
    }
    return a != b;
#else
    return a != b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool flt(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        unsigned int result;
        asm("{ .reg .pred p; setp.lt.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(result) : "f"(a), "f"(b));
        return result != 0U;
    }
    return a < b;
#else
    return a < b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool fle(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        unsigned int result;
        asm("{ .reg .pred p; setp.le.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(result) : "f"(a), "f"(b));
        return result != 0U;
    }
    return a <= b;
#else
    return a <= b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool fgt(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        unsigned int result;
        asm("{ .reg .pred p; setp.gt.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(result) : "f"(a), "f"(b));
        return result != 0U;
    }
    return a > b;
#else
    return a > b;
#endif
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool fge(float a, float b)
{
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        unsigned int result;
        asm("{ .reg .pred p; setp.ge.f32 p, %1, %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(result) : "f"(a), "f"(b));
        return result != 0U;
    }
    return a >= b;
#else
    return a >= b;
#endif
}

// All callers pass an integral float. Determine parity from its representation
// without invoking floating-point arithmetic.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool fis_odd_integer(float a)
{
    const unsigned int bits = cuda::std::bit_cast<unsigned int>(a);
    const int exponent = static_cast<int>((bits >> 23) & 0xFFU) - 127;
    if (exponent < 0 || exponent > 23) {
        return false;
    }
    const unsigned int significand = (bits & 0x007FFFFFU) | 0x00800000U;
    return ((significand >> (23 - exponent)) & 1U) != 0U;
}
} // namespace detail

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
// The reference for the FPAN-based implementation of fltflt_add() is:
//   "High-Performance Branch-Free Algorithms for Extended-Precision Floating-Point Arithmetic",
//   David K. Zhang and Alex Aiken, Proceedings of the International Conference for High Performance
//   Computing, Networking, Storage and Analysis, 2025.

// fltflt represents an unevaluated floating point sum of two non-overlapping fp32 components.
// The hi component is the most significant part of the sum, and the lo component is the least significant part.
struct alignas(8) fltflt {
    float hi;
    float lo;

    // The default constructor does not initialize the components, so the value is indeterminate. Some versions of
    // nvcc will warn about __host__ and __device__ annotations on default constructors because default
    // constructors will not run in all conditions (e.g., in static shared memory CUDA kernel allocations).
    __MATX_INLINE__ fltflt() = default;
    // On device, extract hi/lo from IEEE-754 double bits with no FP64 instructions.
    //
    // Accuracy guarantees:
    //   - fl(hi + lo) == hi  (fast2sum ensures no rounding error when adding lo back) for |x| <= FLT_MAX
    //   - |x - (hi+lo)| <= 8 ulp(x) for |x| <= FLT_MAX
    //   - NaN, Inf, subnormal doubles, and doubles outside float range fall back to
    //     hi = (float)x.
    //   - If hi is NaN or Inf, lo can be arbitrary.
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit fltflt(double x) {
#if defined(__CUDA_ARCH__)
        // Constexpr evaluation on device uses the standard double-subtraction path.
        if (__builtin_is_constant_evaluated()) {
            hi = static_cast<float>(x);
            if (cuda::std::isfinite(hi)) {
                lo = static_cast<float>(x - static_cast<double>(hi));
                // Constant evaluation emits no device FP32 arithmetic instructions.
                float s = hi + lo;
                lo = lo - (s - hi);
                hi = s;
            } else {
                lo = 0.0f;
            }
        } else {
            unsigned long long xbits = __double_as_longlong(x);
            unsigned int sign = static_cast<unsigned int>(xbits >> 63);
            unsigned int e_x = static_cast<unsigned int>((xbits >> 52) & 0x7FFU);
            unsigned long long mant = xbits & 0x000FFFFFFFFFFFFFULL;
            // hi_exp: float biased exponent = (e_x - 1023) + 127 = e_x - 896.
            int hi_exp = static_cast<int>(e_x) - 896;
            if (e_x == 0 || hi_exp <= 0 || hi_exp >= 255) {
                if (MATX_FLTFLT_USE_PTX) {
                    hi = detail::double_to_float_rn(x);
                } else {
                    hi = static_cast<float>(x);
                }
                lo = 0.0f;
            } else {
                // hi: top 23 explicit mantissa bits, round-nearest, ties away from zero.
                // use + to mux in the mantissa, as we may need to carry into the exponent.
                hi = __int_as_float((sign << 31) | ((unsigned int)hi_exp << 23) +
                                    (((unsigned int)(mant >> 28) + 1) >> 1));
                // r: remainder as signed integer (we shift by 3 to get the 29 mantissa bits, with top-most bit the sign bit)
                int r = static_cast<int>(static_cast<unsigned int>(mant) << 3);
                // fast2sum: adjust hi to round-to-nearest and absorb the correction into lo,
                // guaranteeing fl(hi + lo) == hi.
                // two special cases can result in overflow here:
                // 1. |x| >= FLT_MAX + ulp(FLT_MAX)/2,
                //   input: hi == +/-Inf, lo normal.
                //   output: hi == +/-Inf, lo == NaN.
                // 2. |x| > FLT_MAX + ulp(FLT_MAX)/2 - ulp(ulp(FLT_MAX)/2)/2:
                //   input: hi == +/-FLT_MAX, lo == +/-ulp(FLT_MAX)/2
                //   output: hi == +/-Inf, lo == -/+Inf.
                if (MATX_FLTFLT_USE_PTX) {
                    // Keep the correction fused so an exact half-subnormal is not rounded
                    // to zero before it can adjust hi.
                    const float scale = __int_as_float((sign << 31) | (hi_exp << 23));
                    const float scaled_r = detail::fmul_rn(__int2float_rn(r), 0x1p-55f);
                    float s = detail::fmaf_rn(scaled_r, scale, hi);
                    lo = detail::fmaf_rn(scaled_r, scale, detail::fsub_rn(hi, s));
                    hi = s;
                } else {
                    // Preserve the original expression form when FTZ is disabled so the
                    // compiler can fold constant conversions and contract the fast2sum.
                    lo = (__int2float_rn(r) * 0x1p-55f) *
                         __int_as_float((sign << 31) | (hi_exp << 23));
                    float s = hi + lo;
                    lo = lo - (s - hi);
                    hi = s;
                }
            }
        }
#else
        hi = static_cast<float>(x);
        if (cuda::std::isfinite(hi)) {
            lo = static_cast<float>(x - static_cast<double>(hi));
            if (__builtin_is_constant_evaluated()) {
                // Constant evaluation emits no device FP32 arithmetic instructions.
                float s = hi + lo;
                lo = lo - (s - hi);
                hi = s;
            } else {
                float s = detail::fadd_rn(hi, lo);
                lo = detail::fsub_rn(lo, detail::fsub_rn(s, hi));
                hi = s;
            }
        } else {
            lo = 0.0f;
        }
#endif
    }
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit fltflt(float x) : hi(x), lo(0.0f) {}
    // This constructor stores the components as supplied and assumes callers pass
    // normalized components when they require the float-float invariants.
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit fltflt(float hi_, float lo_) : hi(hi_), lo(lo_) {}
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit operator double() const {
        if (__builtin_is_constant_evaluated()) {
            return static_cast<double>(hi) + static_cast<double>(lo);
        }
        return detail::float_to_double(hi) + detail::float_to_double(lo);
    }
    __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ constexpr explicit operator float() const { return hi; }
};

// The constructors and conversion operators in the fltflt struct allow conversion to double and float
// via static_cast<double>(fltflt_val) and similar for float. The fltflt_to_* functions are provided for completeness.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ double fltflt_to_double(fltflt x) {
    return static_cast<double>(x);
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
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ float fltflt_rsqrt(float x)
{
#if defined(__CUDA_ARCH__)
    // rsqrt.approx.f32 has up to 2 ULP of error. This is less precise than
    // 1.0f / ::sqrtf(x), which
    // would be 0.5 ULP of error. We currently use the approximate instruction because it is
    // significantly faster while maintaining 44+ bits of precision in testing thus far, but
    // we may need to revisit this in the future.
    if (MATX_FLTFLT_USE_PTX) {
        float result;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
        return result;
    }
    return ::rsqrtf(x);
#else
    return fdividef_rn(1.0f, ::sqrtf(x));
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
    const float y = detail::fmaf_rn(a, b, detail::fneg(x));
    return fltflt{ x, y };
}

// FPAN-based two-term addition from Zhang & Aiken (SC'25), Figure 2.
//
// The Thall df64_add() form chains two FastTwoSums in series on the
// critical path (the lo-side path traverses both), giving depth 13 fp32
// ops. The FPAN form below runs the FastTwoSum on the hi parts (q)
// concurrently with the lo-side add (st_lo), so q's 3 ops sit entirely
// off the critical path and the lo-side path traverses only one
// FastTwoSum. Critical-path depth: 10 fp32 ops vs 13 for Thall.
//
// Both forms use the same 20 fp32 ops total, so steady-state throughput
// is identical. The latency win shows up when the SM cannot fully hide
// the per-call dependency chain (low occupancy, serial accumulators,
// reduction tails). For reference, Thall's df64_add reads:
//   fltflt s = fltflt_two_sum(a.hi, b.hi);
//   const fltflt t = fltflt_two_sum(a.lo, b.lo);
//   s.lo = detail::fadd_rn(s.lo, t.hi);
//   s = fltflt_fast_two_sum(s.hi, s.lo);
//   s.lo = detail::fadd_rn(s.lo, t.lo);
//   s = fltflt_fast_two_sum(s.hi, s.lo);
//   return s;
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_add(fltflt a, fltflt b) {
    fltflt s = fltflt_two_sum(a.hi, b.hi);
    const fltflt t = fltflt_two_sum(a.lo, b.lo);
    const fltflt q = fltflt_fast_two_sum(s.hi, t.hi);
    const float st_lo = detail::fadd_rn(s.lo, t.lo);
    const float stq_lo = detail::fadd_rn(st_lo, q.lo);
    return fltflt_fast_two_sum(q.hi, stq_lo);
}

// This overload is an optimization of fltflt_add() for the case where b is
// a float, and thus b.lo is zero. The FPAN restructuring above does not
// apply here because there is no second TwoSum to lift; this form
// remains the Thall-style chain (TwoSum -> add -> FastTwoSum, ~9 fp32
// ops on the critical path).
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
    const fltflt neg_b = fltflt{ detail::fneg(b.hi), detail::fneg(b.lo) };
    return fltflt_add(a, neg_b);
}

// This overload is an optimization of fltflt_sub() for the case where b is
// a float, and thus b.lo is zero. It delegates to fltflt_add() with a negated b.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(fltflt a, float b) {
    return fltflt_add(a, detail::fneg(b));
}

// This overload is an optimization of fltflt_sub() for the case where a is
// a float, and thus a.lo is zero. It delegates to fltflt_add() with a negated b.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sub(float a, fltflt b) {
    return fltflt_add(fltflt{ detail::fneg(b.hi), detail::fneg(b.lo) }, a);
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
    // and finally add the c.lo component and the low-low product term. The low-low term
    // can become significant when a*b cancels with c.

    fltflt s = fltflt_two_sum(p.hi, c.hi);
    s.lo = detail::fadd_rn(s.lo, p.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    s.lo = detail::fadd_rn(s.lo, c.lo);
    s.lo = detail::fmaf_rn(a.lo, b.lo, s.lo);

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
    // we skip that step, add c, and then add the p.lo component and the low-low product term.

    fltflt s = fltflt_two_sum(p.hi, c);
    s.lo = detail::fadd_rn(s.lo, p.lo);
    s.lo = detail::fmaf_rn(a.lo, b.lo, s.lo);

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

// A version of fltflt_fma() where a and b are floats. This is more efficient than
// using fltflt_fma(fltflt{ a, 0.0f }, fltflt{ b, 0.0f }, c).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma(float a, float b, fltflt c) {
    fltflt p = fltflt_two_prod_fma(a, b);

    fltflt s = fltflt_two_sum(p.hi, c.hi);
    s.lo = detail::fadd_rn(s.lo, p.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    s.lo = detail::fadd_rn(s.lo, c.lo);

    s = fltflt_fast_two_sum(s.hi, s.lo);

    return s;
}

// fltflt_fma_approx() computes a * b + c but omits the low-low product term
// a.lo*b.lo. It still includes the first-order cross terms a.hi*b.lo and
// a.lo*b.hi. This matches the previous fast FMA behavior and is useful in hot
// paths where the second-order low-low term is known to be below the error
// budget.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma_approx(fltflt a, fltflt b, fltflt c) {
    fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
    p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
    p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);

    fltflt s = fltflt_two_sum(p.hi, c.hi);
    s.lo = detail::fadd_rn(s.lo, p.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);
    s.lo = detail::fadd_rn(s.lo, c.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);

    return s;
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma_approx(fltflt a, fltflt b, float c) {
    fltflt p = fltflt_two_prod_fma(a.hi, b.hi);
    p.lo = detail::fmaf_rn(a.hi, b.lo, p.lo);
    p.lo = detail::fmaf_rn(a.lo, b.hi, p.lo);

    fltflt s = fltflt_two_sum(p.hi, c);
    s.lo = detail::fadd_rn(s.lo, p.lo);
    s = fltflt_fast_two_sum(s.hi, s.lo);

    return s;
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma_approx(float a, fltflt b, fltflt c) {
    return fltflt_fma(a, b, c);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma_approx(fltflt a, float b, fltflt c) {
    return fltflt_fma(a, b, c);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma_approx(fltflt a, float b, float c) {
    return fltflt_fma(a, b, c);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma_approx(float a, fltflt b, float c) {
    return fltflt_fma(a, b, c);
}

static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fma_approx(float a, float b, fltflt c) {
    return fltflt_fma(a, b, c);
}

// fltflt_div() is the df64_div() function given by Thall, which he attributes to Karp.
// This function implements Algorithm 6 from Thall's paper. For the initial approximation,
// we use a round-to-nearest divide on the device.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(fltflt a, fltflt b) {
    const float xn = detail::feq(b.hi, 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt diff = fltflt_fma(detail::fneg(yn), b, a);
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add(prod, yn);
}

// This overload is an optimization of fltflt_div() for the case where b is
// a float, and thus b.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(fltflt a, float b) {
    const float xn = detail::feq(b, 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt diff = fltflt_fma(detail::fneg(yn), b, a);
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add(prod, yn);
}

// This overload is an optimization of fltflt_div() for the case where a is
// a float, and thus a.lo is zero.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_div(float a, fltflt b) {
    const float xn = detail::feq(b.hi, 0.0f) ? 0.0f : detail::fdividef_rn(1.0f, b.hi);
    const float yn = detail::fmul_rn(a, xn);
    const fltflt diff = fltflt_fma(detail::fneg(yn), b, a);
    const fltflt prod = fltflt_two_prod_fma(xn, diff.hi);
    return fltflt_add(prod, yn);
}

// fltflt_round_to_nearest() rounds a float-float value to the nearest integer with
// ties rounded toward even. Note that rounding "toward even" means that, in the case
// of a tie, the least significant bit of the mantissa is set to 0 (i.e., we round to
// the even significand).
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_round_to_nearest(fltflt a) {
    constexpr float FAST_PATH_THRESHOLD = 8388608.0f;
    if (detail::flt(detail::fabs_noftz(a.hi), FAST_PATH_THRESHOLD)) {
        const float candidate = detail::fround_to_nearest(a.hi);

        const float err = detail::fsub_rn(a.hi, candidate);

        if (detail::fne(detail::fabs_noftz(err), 0.5f)) {
            return fltflt{ candidate, 0.0f };
        } else {
            // We should not have errors > 0.5 ulp(a.hi). Since ulp is at most 1, the max error should
            // be 0.5 for the boundary case.
            fltflt result{ candidate, 0.0f };
            if (detail::feq(a.lo, 0.0f)) {
                // Perfect tie, round to even
                const float corrected = !detail::fis_odd_integer(candidate)
                    ? candidate
                    : detail::fadd_rn(candidate, detail::fcopysign(1.0f, err));
                result.hi = corrected;
            } else if ((detail::fgt(err, 0.0f) && detail::fgt(a.lo, 0.0f)) ||
                       (detail::flt(err, 0.0f) && detail::flt(a.lo, 0.0f))) {
                result.hi = detail::fadd_rn(candidate, detail::fcopysign(1.0f, err));
            }
            // We do not need to renormalize because we know the full integral part fits
            // exactly in hi due to the original magnitude check.
            return result;
        }
    } else { // |a.hi| >= 2^23, so a.hi is an integer
        float r_lo = detail::fround_to_nearest(a.lo);
        const float frac = detail::fsub_rn(a.lo, r_lo);

        if (detail::fgt(detail::fabs_noftz(frac), 0.5f)) {
            r_lo = detail::fadd_rn(r_lo, detail::fcopysign(1.0f, frac));
        } else if (detail::feq(detail::fabs_noftz(frac), 0.5f)) {
            // Check if hi + r_lo would be odd (sum is odd iff parities differ)
            bool hi_is_odd = detail::fis_odd_integer(a.hi);
            bool rlo_is_odd = detail::fis_odd_integer(r_lo);
            if (hi_is_odd != rlo_is_odd) {  // XOR of parities
                r_lo = detail::fadd_rn(r_lo, detail::fcopysign(1.0f, frac));
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
    if (detail::flt(detail::fabs_noftz(a.hi), 8388608.0f)) { // |a.hi| < 2^23, so a.hi is not an integer
        const float hi_trunc = detail::fround_toward_zero(a.hi);
        // If hi is exactly an integer, then lo can cause a boundary crossing
        if (detail::feq(hi_trunc, a.hi)) {
            // If hi is 1.0 and lo is -1e-9, value is 0.999... -> trunc to 0.0
            // This happens when signs are opposite.
            if ((detail::fgt(a.hi, 0.0f) && detail::flt(a.lo, 0.0f)) ||
                (detail::flt(a.hi, 0.0f) && detail::fgt(a.lo, 0.0f))) {
                // Pull toward zero by 1 unit
                return fltflt{ detail::fadd_rn(a.hi, detail::fgt(a.hi, 0.0f) ? -1.0f : 1.0f), 0.0f };
            } else {
                // Signs match or lo is 0: truncation is just hi. Fallthrough case.
            }
        }
        return fltflt{ hi_trunc, 0.0f };
    } else { // |a.hi| >= 2^23, so a.hi is an integer
        float lo_trunc = detail::fround_toward_zero(a.lo);
        if (detail::fne(lo_trunc, a.lo)) { // lo has a fractional part, so we may need a correction
            // If lo is opposite sign of hi,
            // the fractional part nudges us across an integer boundary.
            if ((detail::fgt(a.hi, 0.0f) && detail::flt(a.lo, 0.0f)) ||
                (detail::flt(a.hi, 0.0f) && detail::fgt(a.lo, 0.0f))) {
                // If hi=pos, lo=neg (e.g., 10, -0.5), we need 9.
                // If hi=neg, lo=pos (e.g., -10, 0.5), we need -9.
                const float adj = detail::fgt(a.hi, 0.0f) ? -1.0f : 1.0f;
                lo_trunc = detail::fadd_rn(lo_trunc, adj);
            }
        }
        return fltflt_fast_two_sum(a.hi, lo_trunc);
    }
}

// fltflt_floor() returns an integer truncated toward negative infinity. This is
// the largest integer that is not larger than the value a.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_floor(fltflt a) {
    if (detail::flt(detail::fabs_noftz(a.hi), 8388608.0f)) { // |a.hi| < 2^23, so a.hi might not be an integer
        const float hi_floor = detail::fround_down(a.hi);
        // If hi was exactly an integer and lo is negative,
        // the actual value is just below hi, so floor should be hi - 1
        if (detail::feq(hi_floor, a.hi) && detail::flt(a.lo, 0.0f)) {
            return fltflt{ detail::fsub_rn(a.hi, 1.0f), 0.0f };
        }
        return fltflt{ hi_floor, 0.0f };
    } else { // |a.hi| >= 2^23, so a.hi is already an integer
        const float lo_floor = detail::fround_down(a.lo);
        // Renormalize the result
        return fltflt_fast_two_sum(a.hi, lo_floor);
    }
}

// fltflt_sqrt() is the df64_sqrt() function given by Thall, which he attributes to Karp.
// This function implements Algorithm 7 from Thall's paper. It uses the
// two_prod_fma() function for the hi components followed by subtraction of the square
// of the result and re-normalization to a non-overlapping expansion.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_sqrt(fltflt a) {
    const float xn = detail::feq(a.hi, 0.0f) ? 0.0f : detail::fltflt_rsqrt(a.hi);
    const float yn = detail::fmul_rn(a.hi, xn);
    const fltflt ynsqr = fltflt_two_prod_fma(yn, yn);
    const fltflt diff = fltflt_sub(a, ynsqr);
    fltflt prod = fltflt_two_prod_fma(xn, detail::fmul_rn(0.5f, diff.hi));
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
    const float xn = detail::feq(a.hi, 0.0f) ? 0.0f : detail::fltflt_rsqrt(a.hi);
    const float yn = detail::fmul_rn(a.hi, xn);
    const float residual = detail::fadd_rn(
        detail::fmaf_rn(detail::fneg(yn), yn, a.hi), a.lo);
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
    if (detail::flt(a.hi, 0.0f)) {
        return fltflt{ detail::fneg(a.hi), detail::fneg(a.lo) };
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

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator-(fltflt a) {
    return fltflt{ detail::fneg(a.hi), detail::fneg(a.lo) };
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, fltflt b) { return fltflt_mul(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(fltflt a, float b) { return fltflt_mul(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator*(float a, fltflt b) { return fltflt_mul(b, a); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, fltflt b) { return fltflt_div(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(fltflt a, float b) { return fltflt_div(a, b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt operator/(float a, fltflt b) { return fltflt_div(a, b); }

// Dispatch once around each composite comparison. In non-FTZ builds this keeps the
// complete expression visible to the optimizer instead of hiding each scalar
// comparison behind a separately dispatched wrapper.
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(fltflt a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::feq(a.hi, b.hi) && detail::feq(a.lo, b.lo);
    }
#endif
    return a.hi == b.hi && a.lo == b.lo;
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(fltflt a, float b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::feq(a.hi, b) && detail::feq(a.lo, 0.0f);
    }
#endif
    return a.hi == b && a.lo == 0.0f;
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator==(float a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::feq(b.hi, a) && detail::feq(b.lo, 0.0f);
    }
#endif
    return b.hi == a && b.lo == 0.0f;
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(fltflt a, fltflt b) { return !(a == b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(fltflt a, float b) { return !(a == b); }
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator!=(float a, fltflt b) { return !(a == b); }

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(fltflt a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::flt(a.hi, b.hi) || (detail::feq(a.hi, b.hi) && detail::flt(a.lo, b.lo));
    }
#endif
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(fltflt a, float b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::flt(a.hi, b) || (detail::feq(a.hi, b) && detail::flt(a.lo, 0.0f));
    }
#endif
    return a.hi < b || (a.hi == b && a.lo < 0.0f);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<(float a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::flt(a, b.hi) || (detail::feq(a, b.hi) && detail::fgt(b.lo, 0.0f));
    }
#endif
    return a < b.hi || (a == b.hi && b.lo > 0.0f);
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(fltflt a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::fgt(a.hi, b.hi) || (detail::feq(a.hi, b.hi) && detail::fgt(a.lo, b.lo));
    }
#endif
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(fltflt a, float b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::fgt(a.hi, b) || (detail::feq(a.hi, b) && detail::fgt(a.lo, 0.0f));
    }
#endif
    return a.hi > b || (a.hi == b && a.lo > 0.0f);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>(float a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::fgt(a, b.hi) || (detail::feq(a, b.hi) && detail::flt(b.lo, 0.0f));
    }
#endif
    return a > b.hi || (a == b.hi && b.lo < 0.0f);
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(fltflt a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::flt(a.hi, b.hi) || (detail::feq(a.hi, b.hi) && detail::fle(a.lo, b.lo));
    }
#endif
    return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(fltflt a, float b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::flt(a.hi, b) || (detail::feq(a.hi, b) && detail::fle(a.lo, 0.0f));
    }
#endif
    return a.hi < b || (a.hi == b && a.lo <= 0.0f);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator<=(float a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::flt(a, b.hi) || (detail::feq(a, b.hi) && detail::fge(b.lo, 0.0f));
    }
#endif
    return a < b.hi || (a == b.hi && b.lo >= 0.0f);
}

__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(fltflt a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::fgt(a.hi, b.hi) || (detail::feq(a.hi, b.hi) && detail::fge(a.lo, b.lo));
    }
#endif
    return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(fltflt a, float b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::fgt(a.hi, b) || (detail::feq(a.hi, b) && detail::fge(a.lo, 0.0f));
    }
#endif
    return a.hi > b || (a.hi == b && a.lo >= 0.0f);
}
__MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ bool operator>=(float a, fltflt b) {
#if defined(__CUDA_ARCH__)
    if (MATX_FLTFLT_USE_PTX) {
        return detail::fgt(a, b.hi) || (detail::feq(a, b.hi) && detail::fle(b.lo, 0.0f));
    }
#endif
    return a > b.hi || (a == b.hi && b.lo <= 0.0f);
}

// fltflt_fmod() computes the floating-point remainder of division. In other words,
// fltflt_fmod(a, b) = a - n * b where n = trunc(a/b) and trunc() truncates to an integer
// toward zero. If b is zero, returns {NaN, NaN}.
//
// Note: For very large quotients (|a/b| > 10^6), precision is limited by the fltflt type's
// absolute precision at large magnitudes (~|value| * 2^-44). The current implementation does
// not use range reduction or similar techniques to improve precision in these cases, but a
// future implementation may do so.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fmod(fltflt a, fltflt b) {
    if (detail::feq(b.hi, 0.0f) && detail::feq(b.lo, 0.0f)) {
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

    return fltflt{ detail::fmul_rn(sign, result.hi), detail::fmul_rn(sign, result.lo) };
}

// fltflt_fmod() overload where b is a float.
static __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ fltflt fltflt_fmod(fltflt a, float b) {
    if (detail::feq(b, 0.0f)) {
        return fltflt{cuda::std::numeric_limits<float>::quiet_NaN(),
                      cuda::std::numeric_limits<float>::quiet_NaN()};
    }

    float sign = 1.0f;
    if (a < 0.0f) {
        sign = -1.0f;
        a = -a;
    }
    b = detail::fabs_noftz(b);

    const fltflt q = fltflt_div(a, b);
    const fltflt trunc_q = fltflt_round_toward_zero(q);
    fltflt result = -fltflt_fma(trunc_q, b, -a);

    while (result >= b) {
        result = fltflt_sub(result, b);
    }
    while (result < 0.0f) {
        result = fltflt_add(result, b);
    }

    return fltflt{ detail::fmul_rn(sign, result.hi), detail::fmul_rn(sign, result.lo) };
}

} // namespace matx

// cuda::std::numeric_limits specialization for fltflt (double-single extended precision).
// fltflt has the same exponent range as float but approximately 2x the mantissa precision
// (~48 significant binary digits from two non-overlapping fp32 components).
namespace cuda { namespace std {

template <>
class numeric_limits<matx::fltflt> {
  using _FloatLimits = numeric_limits<float>;
public:
  static constexpr bool is_specialized    = true;
  static constexpr bool is_signed         = true;
  static constexpr bool is_integer        = false;
  static constexpr bool is_exact          = false;
  static constexpr int  digits            = 48;        // ~2 * 24 mantissa bits (including implicit)
  static constexpr int  digits10          = 14;        // floor(48 * log10(2))
  static constexpr int  max_digits10      = 16;        // ceil(48 * log10(2) + 1)
  static constexpr int  radix             = 2;
  static constexpr int  min_exponent      = _FloatLimits::min_exponent;    // same dynamic range as float
  static constexpr int  min_exponent10    = _FloatLimits::min_exponent10;
  static constexpr int  max_exponent      = _FloatLimits::max_exponent;
  static constexpr int  max_exponent10    = _FloatLimits::max_exponent10;
  static constexpr bool has_infinity      = true;
  static constexpr bool has_quiet_NaN     = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss   = false;
  static constexpr bool is_iec559         = false;     // not an IEEE 754 format
  static constexpr bool is_bounded        = true;
  static constexpr bool is_modulo         = false;
  static constexpr bool traps             = false;
  static constexpr bool tinyness_before   = false;
  static constexpr float_round_style round_style = round_to_nearest;

  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt min() noexcept          { return matx::fltflt(_FloatLimits::min()); }
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt max() noexcept          { return matx::fltflt(_FloatLimits::max()); }
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt lowest() noexcept       { return matx::fltflt(-_FloatLimits::max()); }
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt epsilon() noexcept      { return matx::fltflt(7.105427357601002e-15f, 0.0f); } // ~2^-47
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt round_error() noexcept  { return matx::fltflt(0.5f); }
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt infinity() noexcept     { return matx::fltflt(_FloatLimits::infinity()); }
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt quiet_NaN() noexcept    { return matx::fltflt(_FloatLimits::quiet_NaN(), _FloatLimits::quiet_NaN()); }
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt signaling_NaN() noexcept { return matx::fltflt(_FloatLimits::signaling_NaN(), _FloatLimits::signaling_NaN()); }
  __MATX_HOST__ __MATX_DEVICE__ static constexpr matx::fltflt denorm_min() noexcept   { return matx::fltflt(_FloatLimits::denorm_min()); }
};

}} // namespace cuda::std

#if defined(__CUDA_ARCH__)
#undef MATX_FLTFLT_USE_PTX
#endif
