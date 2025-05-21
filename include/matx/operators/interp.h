////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#include <cusparse.h>

#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx {

  /**
   * @brief Interpolation method enumeration
   *
   * Specifies the algorithm to use when performing interpolation between sample points.
   */
  enum class InterpMethod {
    LINEAR,  ///< Linear interpolation between adjacent points
    NEAREST, ///< Uses the value at the nearest sample point
    NEXT,    ///< Uses the value at the next sample point
    PREV,    ///< Uses the value at the previous sample point
    SPLINE   ///< Cubic spline interpolation, using not-a-knot boundary conditions
  };

  namespace detail {
    template <class O, class OpX, class OpV>
    class InterpSplineTridiagonalFillOp : public BaseOp<InterpSplineTridiagonalFillOp<O, OpX, OpV>> {
      // this is a custom operator that fills a tridiagonal system
      // for cubic spline interpolation

      using matxop = bool;
    private:
      O dl_, d_, du_, b_;
      OpX x_;
      OpV v_;
      using x_val_type = typename OpX::value_type;
      using v_val_type = typename OpV::value_type;


    public:
      InterpSplineTridiagonalFillOp(const O& dl, const O& d, const O& du, const O& b, const OpX& x, const OpV& v)
          : dl_(dl), d_(d), du_(du), b_(b), x_(x), v_(v)  {}

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {

        cuda::std::array idx{indices...};
        index_t idxInterp = idx[Rank() - 1];

        cuda::std::array idx0{idx};
        cuda::std::array idx1{idx};
        cuda::std::array idx2{idx};

        if (idxInterp == 0) { // left boundary condition
          idx0[Rank() - 1] = idxInterp + 0;
          idx1[Rank() - 1] = idxInterp + 1;
          idx2[Rank() - 1] = idxInterp + 2;

          x_val_type x0 = get_value(x_, idx0);
          x_val_type x1 = get_value(x_, idx1);
          x_val_type x2 = get_value(x_, idx2);
          x_val_type h0 = x1 - x0;
          x_val_type h1 = x2 - x1;

          v_val_type v0 = get_value(v_, idx0);
          v_val_type v1 = get_value(v_, idx1);
          v_val_type v2 = get_value(v_, idx2);

          v_val_type delta0 = (v1 - v0) / h0;
          v_val_type delta1 = (v2 - v1) / h1;

          dl_(indices...) = static_cast<typename O::value_type>(0);
          d_(indices...) = h1;
          du_(indices...) = h1 + h0;
          b_(indices...) = ((2*h1 + 3*h0)*h1*delta0 + h0*h0*delta1) / (h1 + h0);
        }
        else if (idxInterp == x_.Size(0) - 1) { // right boundary condition
          idx0[Rank() - 1] = idxInterp - 2;
          idx1[Rank() - 1] = idxInterp - 1;
          idx2[Rank() - 1] = idxInterp;

          x_val_type x0 = get_value(x_, idx0);
          x_val_type x1 = get_value(x_, idx1);
          x_val_type x2 = get_value(x_, idx2);
          x_val_type h0 = x1 - x0;
          x_val_type h1 = x2 - x1;

          v_val_type v0 = get_value(v_, idx0);
          v_val_type v1 = get_value(v_, idx1);
          v_val_type v2 = get_value(v_, idx2);

          v_val_type delta0 = (v1 - v0) / h0;
          v_val_type delta1 = (v2 - v1) / h1;


          dl_(indices...) = h0 + h1;
          d_(indices...) = h0;
          du_(indices...) = static_cast<typename O::value_type>(0);
          b_(indices...) = ((2*h0 + 3*h1)*h0*delta1 + h1*h1*delta0) / (h0 + h1);
        }
        else { // interior points
          idx0[Rank() - 1] = idxInterp - 1;
          idx1[Rank() - 1] = idxInterp;
          idx2[Rank() - 1] = idxInterp + 1;

          x_val_type x0 = get_value(x_, idx0);
          x_val_type x1 = get_value(x_, idx1);
          x_val_type x2 = get_value(x_, idx2);
          x_val_type h0 = x1 - x0;
          x_val_type h1 = x2 - x1;

          v_val_type v0 = get_value(v_, idx0);
          v_val_type v1 = get_value(v_, idx1);
          v_val_type v2 = get_value(v_, idx2);

          v_val_type delta0 = (v1 - v0) / h0;
          v_val_type delta1 = (v2 - v1) / h1;

          dl_(indices...) = h1;
          d_(indices...) = 2*(h0 + h1);
          du_(indices...) = h0;
          b_(indices...) = 3 * (delta1 * h0 + delta0 * h1);
        }
      }

      __host__ __device__ inline index_t Size(uint32_t i) const  { return d_.Size(i); }
      static inline constexpr __host__ __device__ int32_t Rank() { return O::Rank(); }
    };



  template <typename OpX, typename OpV, typename OpXQ>
  class Interp1Op : public BaseOp<Interp1Op<OpX, OpV, OpXQ>> {
    public:
      using matxop = bool;
      using domain_type = typename OpX::value_type;
      using value_type = typename OpV::value_type;

    private:
      typename detail::base_type_t<OpX> x_;    // Sample points
      typename detail::base_type_t<OpV> v_;    // Values at sample points
      typename detail::base_type_t<OpXQ> xq_;  // Query points
      InterpMethod method_;                    // Interpolation method

      mutable detail::tensor_impl_t<value_type, OpV::Rank()> m_; // Derivatives at sample points (spline only)
      mutable value_type *ptr_m_ = nullptr;

      constexpr static int32_t RANK = OpXQ::Rank();
      constexpr static int32_t AXIS = RANK - 1;

      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto searchsorted(const cuda::std::array<index_t, RANK> idx, const domain_type x_query) const
      {
        // Binary search to find the interval containing the query point

        // if x_query < x(0), idx_low = n, idx_high = 0
        // if x_query > x(n-1), idx_low = n-1, idx_high = n
        // else x(idx_low) <= x_query <= x(idx_high)
        cuda::std::array idx_low{idx};
        cuda::std::array idx_high{idx};
        cuda::std::array idx_mid{idx};

        idx_low[AXIS] = 0;
        idx_high[AXIS] = x_.Size(x_.Rank() - 1) - 1;

        domain_type x_low, x_high, x_mid;

        x_low = get_value(x_, idx_low);
        if (x_query < x_low) {
          idx_low[AXIS] = x_.Size(x_.Rank() - 1);
          idx_high[AXIS] = 0;
          return cuda::std::make_tuple(idx_low, idx_high);
        } else if (x_query == x_low) {
          return cuda::std::make_tuple(idx_low, idx_low);
        }
        x_high = get_value(x_, idx_high);
        if (x_query > x_high) {
          idx_low[AXIS] = x_.Size(x_.Rank() - 1) - 1;
          idx_high[AXIS] = x_.Size(x_.Rank() - 1);
          return cuda::std::make_tuple(idx_low, idx_high);
        } else if (x_query == x_high) {
          return cuda::std::make_tuple(idx_high, idx_high);
        }

        // Find the interval containing the query point
        while (idx_high[AXIS] - idx_low[AXIS] > 1) {
          idx_mid[AXIS] = (idx_low[AXIS] + idx_high[AXIS]) / 2;
          x_mid = get_value(x_, idx_mid);
          if (x_query == x_mid) {
            return cuda::std::make_tuple(idx_mid, idx_mid);
          } else if (x_query < x_mid) {
            idx_high[AXIS] = idx_mid[AXIS];
            x_high = x_mid;
          } else {
            idx_low[AXIS] = idx_mid[AXIS];
            x_low = x_mid;
          }
        }
        return cuda::std::make_tuple(idx_low, idx_high);
      }

      // Linear interpolation implementation
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
      value_type interpolate_linear(const domain_type x_query, cuda::std::array<index_t, RANK> idx_low, cuda::std::array<index_t, RANK> idx_high) const {
        value_type v;

        if (idx_high[AXIS] == 0 || idx_low[AXIS] == idx_high[AXIS]) { // x_query <= x(0) or x_query == x(idx_low) == x(idx_high)
          v = get_value(v_, idx_high);
        } else if (idx_low[AXIS] == x_.Size(0) - 1) { // x_query > x(n-1)
          v = get_value(v_, idx_low);
        } else {
          domain_type x_low = get_value(x_, idx_low);
          domain_type x_high = get_value(x_, idx_high);
          value_type v_low = get_value(v_, idx_low);
          value_type v_high = get_value(v_, idx_high);
          v = v_low + (x_query - x_low) * (v_high - v_low) / (x_high - x_low);
        }
        return v;
      }

      // Nearest neighbor interpolation implementation
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
      value_type interpolate_nearest(const domain_type x_query, cuda::std::array<index_t, RANK> idx_low, cuda::std::array<index_t, RANK> idx_high) const {
        value_type v;
        if (idx_low[AXIS] == x_.Size(0)) { // x_query < x(0)
          v = get_value(v_, idx_high);
        } else if (idx_high[AXIS] == x_.Size(0)) { // x_query > x(n-1)
          v = get_value(v_, idx_low);
        } else {
          domain_type x_low = get_value(x_, idx_low);
          domain_type x_high = get_value(x_, idx_high);
          if (x_query - x_low < x_high - x_query) {
            v = get_value(v_, idx_low);
          } else {
            v = get_value(v_, idx_high);
          }
        }
        return v;
      }


      // Next value interpolation implementation
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
      value_type interpolate_next(const domain_type x_query, cuda::std::array<index_t, RANK> idx_low, cuda::std::array<index_t, RANK> idx_high) const {
        value_type v;
        if (idx_high[AXIS] == x_.Size(0)) { // x_query > x(n-1)
          v = get_value(v_, idx_low);
        } else {
          v = get_value(v_, idx_high);
        }
        return v;
      }

      // Previous value interpolation implementation
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
      value_type interpolate_prev(const domain_type x_query, cuda::std::array<index_t, RANK> idx_low, cuda::std::array<index_t, RANK> idx_high) const {
        value_type v;
        if (idx_low[AXIS] == x_.Size(0)) { // x_query < x(0)
          v = get_value(v_, idx_high);
        } else {
          v = get_value(v_, idx_low);
        }
        return v;
      }

      // Spline interpolation implementation
      // Hermite cubic interpolation
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
      value_type interpolate_spline(const domain_type x_query, cuda::std::array<index_t, RANK> idx_low, cuda::std::array<index_t, RANK> idx_high) const {
        if (idx_high[AXIS] == idx_low[AXIS]) {
          value_type v = get_value(v_, idx_low);
          return v;
        } else if (idx_low[AXIS] == x_.Size(0)) { // x_query < x(0)
          idx_low[AXIS] = 0;
          idx_high[AXIS] = 1;
        } else if (idx_high[AXIS] == x_.Size(0)) { // x_query > x(n-1)
          idx_high[AXIS] = x_.Size(0) - 1;
          idx_low[AXIS] = x_.Size(0) - 2;
        }

        // sample points
        domain_type x_low = get_value(x_, idx_low);
        domain_type x_high = get_value(x_, idx_high);

        // values at the sample points
        value_type v_low = get_value(v_, idx_low);
        value_type v_high = get_value(v_, idx_high);
        value_type v_diff = v_high - v_low;

        // derivatives at the sample points
        value_type m_low = get_value(m_, idx_low);
        value_type m_high = get_value(m_, idx_high);

        value_type h = x_high - x_low;
        value_type h_low = x_query - x_low;
        value_type h_high = x_high - x_query;

        value_type t = h_low / h;
        value_type s = h_high / h;

        value_type v = s * v_low \
          + t * v_high \
          + (h * (m_low * s - m_high * t) + v_diff * (t - s)) * t * s;

        return v;
      }

      // Dispatch to appropriate interpolation method based on enum
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__
      value_type interpolate(const domain_type x_query, cuda::std::array<index_t, RANK> idx_low, cuda::std::array<index_t, RANK> idx_high) const {
        switch (method_) {
          case InterpMethod::LINEAR:
            return interpolate_linear(x_query, idx_low, idx_high);
          case InterpMethod::NEAREST:
            return interpolate_nearest(x_query, idx_low, idx_high);
          case InterpMethod::NEXT:
            return interpolate_next(x_query, idx_low, idx_high);
          case InterpMethod::PREV:
            return interpolate_prev(x_query, idx_low, idx_high);
          case InterpMethod::SPLINE:
            return interpolate_spline(x_query, idx_low, idx_high);
          default:
            // Default to linear interpolation
            return interpolate_linear(x_query, idx_low, idx_high);
        }
      }


    public:
      __MATX_INLINE__ std::string str() const { return "interp1()"; }

      __MATX_INLINE__ Interp1Op(const OpX &x, const OpV &v, const OpXQ &xq, InterpMethod method = InterpMethod::LINEAR) :
        x_(x),
        v_(v),
        xq_(xq),
        method_(method)
      {
        if (x_.Size(x_.Rank() - 1) != v_.Size(v_.Rank() - 1)) {
          MATX_THROW(matxInvalidSize, "interp1: sample points and values must have the same size in the last dimension");
        }
        for (int ri = 2; ri <= x_.Rank(); ri++) {
          if (xq_.Size(xq_.Rank() - ri) != x_.Size(x_.Rank() - ri)) {
            MATX_THROW(matxInvalidSize, "interp1: query and sample points must have compatible dimensions");
          }
        }
        for (int ri = 2; ri <= v_.Rank(); ri++) {
          if (xq_.Size(xq_.Rank() - ri) != v_.Size(v_.Rank() - ri)) {
            MATX_THROW(matxInvalidSize, "interp1: query points and sample values must have compatible dimensions");
          }
        }

      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
      {
        return OpXQ::Rank();
      }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        return xq_.Size(dim);
      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const {

        // Allocate temporary storage for spline coefficients
        if (method_ == InterpMethod::SPLINE) {
          static_assert(is_cuda_executor_v<Executor>, "cubic spline interpolation only supports the CUDA executor currently");

          index_t _batch_count = 1;
          for (int i = 0; i < v_.Rank() - 1; i++) {
            _batch_count *= v_.Size(i);
          }
          index_t _n = v_.Size(v_.Rank() - 1);
          if (_batch_count > std::numeric_limits<int>::max() || _n > std::numeric_limits<int>::max()) {
            const std::string err_msg = "Spline interpolation is not supported for tensors with more than 2^" + std::to_string(std::numeric_limits<int>::digits) + " items";
            MATX_THROW(matxInvalidSize, err_msg.c_str());
          }
          int batch_count = static_cast<int>(_batch_count);
          int n = static_cast<int>(_n); // number of sample points

          cuda::std::array m_shape = v_.Shape();
          detail::AllocateTempTensor(m_, std::forward<Executor>(ex), m_shape, &ptr_m_);

          detail::tensor_impl_t<value_type, OpV::Rank()> d_tensor, dl_tensor, du_tensor; // Derivatives at sample points (spline only)
          value_type *ptr_dl_ = nullptr;
          value_type *ptr_d_ = nullptr;
          value_type *ptr_du_ = nullptr;

          detail::AllocateTempTensor(dl_tensor, std::forward<Executor>(ex), m_shape, &ptr_dl_);
          detail::AllocateTempTensor(d_tensor, std::forward<Executor>(ex), m_shape, &ptr_d_);
          detail::AllocateTempTensor(du_tensor, std::forward<Executor>(ex), m_shape, &ptr_du_);

          // Fill tridiagonal system via custom operator
          InterpSplineTridiagonalFillOp(dl_tensor,d_tensor, du_tensor, m_, x_, v_).run(std::forward<Executor>(ex));

          // Solve tridiagonal system using cuSPARSE
          cudaStream_t stream = ex.getStream();
          cusparseHandle_t handle = nullptr;
          cusparseStatus_t cusparse_status = cusparseCreate(&handle);
          MATX_ASSERT(cusparse_status == CUSPARSE_STATUS_SUCCESS, matxCudaError);
          cusparse_status = cusparseSetStream(handle, stream);
          MATX_ASSERT(cusparse_status == CUSPARSE_STATUS_SUCCESS, matxCudaError);

          size_t workspace_size = 0;
          void* workspace = nullptr;
          if constexpr (std::is_same_v<value_type, float>) {
            cusparse_status = cusparseSgtsv2StridedBatch_bufferSizeExt(
              handle,             // cuSPARSE handle
              n,                  // n
              ptr_dl_,            // sub-diagonal
              ptr_d_,             // main-diagonal
              ptr_du_,            // super-diagonal
              ptr_m_,             // right-hand side and solution
              batch_count,        // batch_count
              n,                  // batch_stride
              &workspace_size);   // workspace size
          } else if constexpr (std::is_same_v<value_type, double>) {
            cusparse_status = cusparseDgtsv2StridedBatch_bufferSizeExt(
              handle,             // cuSPARSE handle
              n,                  // n
              ptr_dl_,            // sub-diagonal
              ptr_d_,             // main-diagonal
              ptr_du_,            // super-diagonal
              ptr_m_,             // right-hand side and solution
              batch_count,        // batch_count
              n,                  // batch_stride
              &workspace_size);   // workspace size
          }
          MATX_ASSERT(cusparse_status == CUSPARSE_STATUS_SUCCESS, matxCudaError);
          cudaError_t err = cudaMallocAsync(&workspace, workspace_size, stream);
          MATX_ASSERT(err == cudaSuccess, matxCudaError);

          if constexpr (std::is_same_v<value_type, float>) {
            cusparse_status = cusparseSgtsv2StridedBatch(
              handle,       // cuSPARSE handle
              n,            // Size of the system
              ptr_dl_,      // Sub-diagonal
              ptr_d_,       // Main diagonal
              ptr_du_,      // Super-diagonal
              ptr_m_,       // Right-hand side and solution
              batch_count,  // batch_count
              n,            // batch_stride
              workspace);   // Workspace buffer
          } else if constexpr (std::is_same_v<value_type, double>) {
            cusparse_status = cusparseDgtsv2StridedBatch(
              handle,       // cuSPARSE handle
              n,            // Size of the system
              ptr_dl_,      // Sub-diagonal
              ptr_d_,       // Main diagonal
              ptr_du_,      // Super-diagonal
              ptr_m_,       // Right-hand side and solution
              batch_count,  // batch_count
              n,            // batch_stride
              workspace);   // Workspace buffer
          }
          MATX_ASSERT(cusparse_status == CUSPARSE_STATUS_SUCCESS, matxCudaError);
          // cleanup
          err = cudaFreeAsync(workspace, stream);
          MATX_ASSERT(err == cudaSuccess, matxCudaError);
          cusparse_status = cusparseDestroy(handle);
          MATX_ASSERT(cusparse_status == CUSPARSE_STATUS_SUCCESS, matxCudaError);
          matxFree(ptr_d_);
          matxFree(ptr_dl_);
          matxFree(ptr_du_);
        }

      }

      template <typename ShapeType, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape,
                                  [[maybe_unused]] Executor &&ex) const noexcept {
        if (method_ == InterpMethod::SPLINE) {
          matxFree(ptr_m_);
        }
      }


      template <typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
      {
        cuda::std::array idx{indices...};
        auto x_query = xq_(indices...);
        auto [idx_low, idx_high] = searchsorted(idx, x_query);

        return interpolate(x_query, idx_low, idx_high);
      }

    };
  } // namespace detail


/**
 * 1D interpolation of samples at query points. 
 * 
 * Interpolation is performed along the last dimension. All other dimensions must be of 
 * compatible size.
 *
 * @tparam OpX
 *   Type of sample points
 * @tparam OpV
 *   Type of sample values
 * @tparam OpXQ
 *   Type of query points
 * @param x
 *   Sample points. Last dimension must be sorted in ascending order.
 * @param v
 *   Sample values. Must have compatible dimensions with x.
 * @param xq
 *   Query points where to interpolate. All dimensions except the last must be of compatible size with x and v (e.g. x and v can be vectors, and xq can be a matrix).
 * @param method
 *   Interpolation method (LINEAR, NEAREST, NEXT, PREV, SPLINE)
 * @returns Operator that interpolates values at query points, with the same dimensions as xq.
 */
template <typename OpX, typename OpV, typename OpXQ>
auto interp1(const OpX &x, const OpV &v, const OpXQ &xq, InterpMethod method = InterpMethod::LINEAR) {
  static_assert(OpX::Rank() >= 1, "interp: sample points must be at least 1D");
  static_assert(OpV::Rank() >= OpX::Rank(), "interp: sample values must have at least the same rank as sample points");
  static_assert(OpXQ::Rank() >= OpV::Rank(), "interp: query points must have at least the same rank as sample values");
  return detail::Interp1Op<OpX, OpV, OpXQ>(x, v, xq, method);
}
} // namespace matx
