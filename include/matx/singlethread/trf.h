////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2024, NVIDIA Corporation
// All rights reserved.
//
// Original TRF python code https://github.com/scipy/scipy/blob/bfaef273d1ec251ea3d5ef52853fadeb7f39fc5e/scipy/optimize/_lsq/
// Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
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

#include "matx.h"
#include "svdpi.h"
#include "linalg.h"

namespace matx {
  namespace st {

    /**
     * Single threaded implementation of Trust Region Reflective algorithm to
     * solve a nonlinear least squares problem
     *
     * This CRTP base class implements the TRF iterative method for solving a nonlinear
     * least squares problem with bounds. The user must derive a class
     * from this CRTP base class which defines a function f().
     *
     * For more details see
     *   1. Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
     *      and Conjugate Gradient Method for Large-Scale Bound-Constrained
     *      Minimization Problems," SIAM Journal on Scientific Computing,
     *      Vol. 21, Number 1, pp 1-23, 1999.
     *
     *   2. More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
     *      and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
     *
     *   3. TRF python implementation https://github.com/scipy/scipy/blob/main/scipy/optimize/_lsq
     *
     * Note: This implementation does not implement all features of the python version
     *
     * @tparam OPT_FUNC
     *   CRTP derived class.  The derived class must implement a function f() with signature
     *   static void f(const float (&x)[NP], const float (&n)[NX], float& y, float (&dy)[NP])
     *   Note that the classic Gauss-Newton algorithm does not support constraints/bounds
     *   on the parameter estimates, however he derived class may optionally overload
     *   apply_bounds() to define custom parameter update rules after each iteration.
     *
     * @tparam NP Number of parameters to solve for function f()
     *
     * @tparam NX Number of independent variable inputs to function f()
     *
     * @tparam NF Number of observations of the function f()
     *
     * @tparam VERBOSE 0:no debug prints, 1:print solution, 2:print iteration details.
     *   Warning: No thread/block information is included in the debug prints, it is
     *            recommmended to only use VERBOSE>0 with a single block/single thread
     */
    template<typename OPT_FUNC, int NP, int NX, int NF, int VERBOSE=0>
    class trf_base
    {
      __device__ inline void print_header_nonlinear()
      {
               //       0            1      8.1661e-02          0.0000e+00          1.0000e-01        9.20e-02        0.00e+00
        printf("Iteration   Total nfev            Cost      Cost reduction           Step norm      Optimality           Alpha\n");
      }

      __device__ inline void print_iteration_nonlinear(int iteration, int nfev, float cost, float cost_reduction, float step_norm, float optimality, float alpha)
      {
        printf("%9d %12d %15.4e %19.4e %19.4e %15.2e %15.2e\n",
            iteration,
            nfev,
            cost,
            cost_reduction,
            step_norm,
            optimality,
            alpha
        );
      }

      __device__ float inline trf_invsqrtscaled_norm(const float x[NP], const float v[NP])
      {
        float norm = 0.f;
        for (int k=0; k<NP; k++)
        {
          float inv_sqrt_v_k = __frsqrt_rn(v[k]);
          float scaled_x = x[k] * inv_sqrt_v_k;
          norm += scaled_x * scaled_x;
        }
        return __fsqrt_rn(norm);
      }

      __device__ float inline trf_scaled_norm(const float g[NP], const float v[NP])
      {
        float norm = 0.f;
        for (int k=0; k<NP; k++)
        {
          float scaled_g = g[k] * v[k];
          norm += scaled_g * scaled_g;
        }
        return __fsqrt_rn(norm);
      }

      __device__ void inline CL_scaling_vector(const float x[NP], const float g[NP], const float lb[NP], const float ub[NP], float v[NP], float dv[NP])
      {
        for (int k=0; k<NP; k++)
        {
          v[k] = 1.f;
          dv[k] = 0.f;

          if (g[k] < 0) // and check ub is finite
          {
            v[k] = ub[k] - x[k];
            dv[k] = -1.f;
          }
          else if (g[k] > 0) // and check lb is finite
          {
            v[k] = x[k] - lb[k];
            dv[k] = 1.f;
          }
        }
      }

      __device__ void inline phi_and_derivative(float alpha, const float (&suf)[NP], const float (&s)[NP], float Delta, float& phi, float& phi_prime)
      {
        float suf_div_denom[NP];
        float denom[NP];
        for (int k=0; k<NP; k++)
        {
          denom[k] = s[k]*s[k] + alpha;
          suf_div_denom[k] = suf[k] / denom[k];
        }
        float p_norm = norm(suf_div_denom);
        phi = p_norm - Delta;
        phi_prime = 0.f;
        for (int k=0; k<NP; k++)
        {
          phi_prime -= suf[k]*suf[k] / (denom[k]*denom[k]*denom[k]) / p_norm;
        }
      }

      __device__ void inline solve_lsq_trust_region(int n, int m, const float (&uf)[NP], const float (&s)[NP], const float (&VT)[NP][NP], float Delta, float& alpha, float (&p)[NP], int& n_iter)
      {
        const int max_iter = 10;
        const float rtol = 0.01f;
        const float EPS = 1e-5;
        bool full_rank = false;

        if (m >= n)
        {
          float threshold = EPS * 3.f * s[0];
          full_rank = s[2] > threshold;
        }

        float negV[NP][NP];
        #pragma unroll
        for (int k=0; k<NP; k++)
        {
          #pragma unroll
          for (int n=0; n<NP; n++)
          {
            negV[n][k] = -VT[k][n];
          }
        }

        float uf_div_s[NP];
        #pragma unroll
        for (int k=0; k<NP; k++)
        {
          uf_div_s[k] = uf[k] / s[k];
        }

        if (full_rank)
        {
          matmul_ATxB(p, negV, uf_div_s);
          float norm_p = norm(p);
          if (norm_p <= Delta)
          {
            alpha = 0.f;
            n_iter = 0;
            return;
          }
        }

        float suf[NP];
        for (int k=0; k<NP; k++)
        {
          suf[k] = s[k] * uf[k];
        }
        float norm_suf = norm(suf);
        float alpha_upper = norm_suf / Delta;
        float alpha_lower;
        float phi;
        float phi_prime;

        if (full_rank)
        {
          phi_and_derivative(0.0f, suf, s, Delta, phi, phi_prime);
          alpha_lower = -phi / phi_prime;
        }
        else
        {
          alpha_lower = 0.f;
        }

        if ((full_rank == false) && (alpha == 0.f))
        {
          alpha = std::max(0.001f * alpha_upper, sqrtf(alpha_lower * alpha_upper));
        }

        for (int it=0; it<max_iter; it++)
        {
          if ((alpha < alpha_lower) || (alpha > alpha_upper))
          {
            alpha = std::max(0.001f * alpha_upper, sqrtf(alpha_lower * alpha_upper));
          }

          phi_and_derivative(alpha, suf, s, Delta, phi, phi_prime);

          float ratio = phi / phi_prime;
          alpha_lower = std::max(alpha_lower, alpha - ratio);
          alpha -= (phi + Delta) * ratio / Delta;

          if (fabs(phi) < (rtol * Delta))
          {
            break;
          }
          n_iter++;
        }

        float suf_div_ssquared_plus_alpha[NP];
        for (int k=0; k<NP; k++)
        {
         suf_div_ssquared_plus_alpha[k] = suf[k] / (s[k]*s[k] + alpha);
        }

        matmul_AxB(p, negV, suf_div_ssquared_plus_alpha);

        float norm_p = norm(p);
        for (int k=0; k<NP; k++)
        {
          p[k] *= Delta / norm_p;
        }

        n_iter++;
      }

      __device__ inline bool in_bounds(const float (&x)[NP], const float (&lb)[NP], const float (&ub)[NP])
      {
        for (int k=0; k<NP; k++)
        {
          if (x[k] < lb[k]) return false;
          if (x[k] > ub[k]) return false;
        }
        return true;
      }

      // The function computes a positive scalar t, such that x + s * t is on
      //    the bound.
      __device__ inline void step_size_to_bound(const float (&x)[NP], const float (&s)[NP], const float (&lb)[NP], const float (&ub)[NP], float& min_step, int (&hits)[NP])
      {
        float steps[NP];
        for (int k=0; k<NP; k++)
        {
          float stepa = (lb[k] - x[k]) / s[k];
          float stepb = (ub[k] - x[k]) / s[k];
          steps[k] = std::max(stepa, stepb);
        }

        min_step = std::min(steps[0], steps[1]);
        min_step = std::min(steps[1], min_step);

        for (int k=0; k<NP; k++)
        {
          hits[k] = 0;
          if (steps[k] == min_step)
          {
            if (s[k] > 0)
            {
              hits[k] = 1;
            }
            else if (s[k] < 0)
            {
              hits[k] = -1;
            }
          }
        }
      }

      __device__ inline float evaluate_quadratic(const float (&J)[NF][NP], const float (&g)[NP], const float (&s)[NP], const float (&diag)[NP])
      {
        // The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s

        // s.ndim == 1
        float Js[NF];
        matmul_AxB(Js, J, s);
        float q;
        dot_AB(q, Js, Js);
        float sdiag[NP];
        for (int k=0; k<NP; k++)
        {
          sdiag[k] = s[k] * diag[k];
        }
        float qadj;
        dot_AB(qadj, sdiag, s);
        q += qadj;

        float l;
        dot_AB(l, s, g);

        return 0.5f * q + l;
      }

      __device__ void inline build_quadratic_1d(const float (&J)[NF][NP], const float (&g)[NP], const float (&s)[NP], const float (&diag)[NP], float& a, float& b)
      {
        float v[NF];
        matmul_AxB(v, J, s);
        dot_AB(a, v, v);
        float s_diag[NP];
        for (int k=0; k<NP; k++)
        {
          s_diag[k] = s[k] * diag[k];
        }
        float s0diags0_dot;
        dot_AB(s0diags0_dot, s_diag, s);
        a += s0diags0_dot;
        a *= 0.5f;

        dot_AB(b, g, s);
      }

      __device__ void inline build_quadratic_1d(const float (&J)[NF][NP], const float (&g)[NP], const float (&s)[NP], const float (&s0)[NP], const float (&diag)[NP], float& a, float& b, float& c)
      {
        float v[NF];
        matmul_AxB(v, J, s);
        dot_AB(a, v, v);
        float s_diag[NP];
        for (int k=0; k<NP; k++)
        {
          s_diag[k] = s[k] * diag[k];
        }
        float s0diags0_dot;
        dot_AB(s0diags0_dot, s_diag, s);
        a += s0diags0_dot;
        a *= 0.5f;

        dot_AB(b, g, s);

        float u[NF];
        matmul_AxB(u, J, s0);
        float uv_dot;
        dot_AB(uv_dot, u, v);
        b += uv_dot;
        float uu_dot;
        dot_AB(uu_dot, u, u);
        float gs0_dot;
        dot_AB(gs0_dot, g, s0);
        c = 0.5f * uu_dot + gs0_dot;
        b += s0diags0_dot;
        c += 0.5f * s0diags0_dot;
      }

      __device__ void inline minimize_quadratic_1d(float a, float b, float c, float lb, float ub, float &t, float &y)
      {
        float tvec[3] {lb, ub, cuda::std::numeric_limits<float>::infinity()};

        if (a != 0)
        {
          float extremum = -0.5f * b/a;
          if ((lb < extremum) && (extremum < ub))
          {
            tvec[2] = extremum;
          }
        }

        float min_y = cuda::std::numeric_limits<float>::infinity();
        int min_index = 0;
        for (int k=0; k<3; k++)
        {
          float cur_y = tvec[k] * (a * tvec[k] + b) + c;
          if (cur_y < min_y)
          {
            min_y = cur_y;
            min_index = k;
          }
        }
        y = min_y;
        t = tvec[min_index];
      }

      __device__ inline void intersect_trust_region(const float (&x)[NP], const float (&s)[NP], float Delta, float& t_neg, float& t_pos)
      {
        float a;
        dot_AB(a, s, s);

        float b;
        dot_AB(b, x, s);

        float c;
        dot_AB(c, x, x);
        c -= Delta*Delta;

        float d = sqrtf(b*b - a*c);
        float copysign_d_b = d;
        if (b < 0)
        {
          copysign_d_b *= -1;
        }

        float q = -(b + copysign_d_b);
        float t1 = q / a;
        float t2 = c / q;

        if (t1 < t2)
        {
          t_neg = t1;
          t_pos = t2;
        }
        else
        {
          t_neg = t2;
          t_pos = t1;
        }
      }

      __device__ inline void select_step(
          const float (&x)[NP],
          const float (&J_h)[NF][NP],
          const float (&diag_h)[NP],
          const float (&g_h)[NP],
          const float (&in_p)[NP],
          const float (&in_p_h)[NP],
          const float (&d)[NP],
          float Delta,
          const float (&lb)[NP],
          const float (&ub)[NP],
          float theta,
          float (&step)[NP],
          float (&step_h)[NP],
          float &predicted_reduction,
          bool &fatal_error)
      {
        fatal_error = false;
        float x_plus_p[NP];
        for (int k=0; k<NP; k++)
        {
          x_plus_p[k] = x[k] + in_p[k];
        }

        if (in_bounds(x_plus_p, lb, ub))
        {
          float p_value = evaluate_quadratic(J_h, g_h, in_p_h, diag_h);
          for (int k=0; k<NP; k++)
          {
            step[k] = in_p[k];
            step_h[k] = in_p_h[k];
          }
          predicted_reduction = -p_value;
          return;
        }

        float p_stride;
        int hits[NP];
        step_size_to_bound(x, in_p, lb, ub, p_stride, hits);

        // Compute the reflected direction.
        // and restrict trust-region step, such that it hits the bound.
        float updated_p[NP];
        float updated_p_h[NP];
        float r_h[NP];
        float r[NP];
        float x_on_bound[NP];
        for (int k=0; k<NP; k++)
        {
          r_h[k] = in_p_h[k];
          if (hits[k])
          {
            r_h[k] *= -1;
          }
          r[k] = d[k] * r_h[k];

          updated_p[k] = in_p[k] * p_stride;
          updated_p_h[k] = in_p_h[k] * p_stride;
          x_on_bound[k] = x[k] + updated_p[k];
        }

        // Reflected direction will cross first either feasible region or trust
        // region boundary.
        float to_tr;
        float to_bound;
        int hits_unused[NP];
        float tmp_unused;
        intersect_trust_region(updated_p_h, r_h, Delta, tmp_unused, to_tr);
        step_size_to_bound(x_on_bound, r, lb, ub, to_bound, hits_unused);

        // Find lower and upper bounds on a step size along the reflected
        // direction, considering the strict feasibility requirement. There is no
        // single correct way to do that, the chosen approach seems to work best
        // on test problems.
        float r_stride = cuda::std::min(to_bound, to_tr);
        float r_stride_l;
        float r_stride_u;
        if (r_stride > 0)
        {
          r_stride_l = (1.f - theta) * p_stride / r_stride;
          if (r_stride == to_bound)
          {
            r_stride_u = theta * to_bound;
          }
          else
          {
            r_stride_u = to_tr;
          }
        }
        else
        {
          r_stride_l = 0;
          r_stride_u = -1;
        }

        // Check if reflection step is available.
        float r_value;
        if (r_stride_l <= r_stride_u)
        {
          float a, b, c;
          build_quadratic_1d(J_h, g_h, r_h, updated_p_h, diag_h, a, b, c);
          float r_stride;
          minimize_quadratic_1d(a, b, c, r_stride_l, r_stride_u, r_stride, r_value);
          for (int k=0; k<NP; k++)
          {
            r_h[k] *= r_stride;
            r_h[k] += updated_p_h[k];
            r[k] = r_h[k] * d[k];
          }
        }
        else
        {
          r_value = cuda::std::numeric_limits<float>::infinity();
        }

        // Now correct p_h to make it strictly interior.
        for (int k=0; k<NP; k++)
        {
          updated_p[k] *= theta;
          updated_p_h[k] *= theta;
        }
        float p_value;
        p_value = evaluate_quadratic(J_h, g_h, updated_p_h, diag_h);

        float ag_h[NP];
        float ag[NP];
        for (int k=0; k<NP; k++)
        {
          ag_h[k] = -g_h[k];
          ag[k] = d[k] * ag_h[k];
        }

        float norm_ag_h = norm(ag_h);
        to_tr = Delta / norm_ag_h;
        step_size_to_bound(x, ag, lb, ub, to_bound, hits_unused);

        float ag_stride;
        if (to_bound < to_tr)
        {
          ag_stride = theta * to_bound;
        }
        else
        {
          ag_stride = to_tr;
        }

        float ag_value;
        {
          float a, b;
          [[maybe_unused]] float c_unused;
          [[maybe_unused]] float s0_unused[NP];
          build_quadratic_1d(J_h, g_h, ag_h, diag_h, a, b);
          float ag_stride_out;
          minimize_quadratic_1d(a, b, 0, 0, ag_stride, ag_stride_out, ag_value);
          ag_stride = ag_stride_out;
          for (int k=0; k<NP; k++)
          {
            ag_h[k] *= ag_stride;
            ag[k] *= ag_stride;
          }
        }

        if ((p_value < r_value) && (p_value < ag_value))
        {
          //return p, p_h, -p_value
          for (int k=0; k<NP; k++)
          {
            step[k] = updated_p[k];
            step_h[k] = updated_p_h[k];
          }
          predicted_reduction = -p_value;
        }
        else if ((r_value < p_value) && (r_value < ag_value))
        {
          //return r, r_h, -r_value
          for (int k=0; k<NP; k++)
          {
            step[k] = r[k];
            step_h[k] = r_h[k];
          }
          predicted_reduction = -r_value;

        }
        else
        {
          //return ag, ag_h, -ag_value
          for (int k=0; k<NP; k++)
          {
            step[k] = ag[k];
            step_h[k] = ag_h[k];
          }
          predicted_reduction = -ag_value;
        }
      }

      __device__ inline void update_tr_radius(
          float Delta,
          float actual_reduction,
          float predicted_reduction,
          float step_norm,
          bool bound_hit,
          float& Delta_new,
          float& ratio)
      {
        if (predicted_reduction > 0)
        {
          ratio = actual_reduction / predicted_reduction;
        }
        else if (predicted_reduction == actual_reduction)
        {
          ratio = 1.f;
        }
        else
        {
          ratio = 0;
        }

        if (ratio < 0.25f)
        {
          Delta_new = 0.25 * step_norm;
        }
        else if ((ratio > 0.75f) && (bound_hit))
        {
          Delta_new = Delta * 2;
        }
        else
        {
          Delta_new = Delta;
        }
      }

      __device__ inline int check_termination(float dF, float F, float dx_norm, float x_norm, float ratio, float ftol, float xtol)
      {
        bool ftol_satisfied = (dF < (ftol * F)) && (ratio > 0.25);
        bool xtol_satisfied = dx_norm < (xtol * (xtol + x_norm));

        if (ftol_satisfied && xtol_satisfied)
        {
          return 4;
        }
        else if (ftol_satisfied)
        {
          return 2;
        }
        else if (xtol_satisfied)
        {
          return 3;
        }

        return 0;
      }

    public:
      /**
       * Iteratively solves a nonlinear least squares problem using the TRF method
       *
       * trf_bounds() implements the Trust Region Reflective iterative method for
       * solving a nonlinear least squares problem.
       * trf_bounds() requires a CRTP derived function OPT_FUNC::f() to calculate
       * scalar value and NP partial differentials of the function being optimized.
       *
       * @param[in,out] x
       *   NP-length parameter estimate vector.  The vector should be set to an initial value
       *   by the caller.  The vector is updated with the solver solution.
       *
       * @param[in] observations
       *   NF-length observation vector
       *
       * @param[in] n
       *   NF-length list of NX independent variables used in OPT_FUNC::f() corresponding to the
       *   observations vector
       *
       * @param[in] ub
       *   NP-length upper bound vector
       *
       * @param[in] lb
       *   NP-length lower bound vector
       *
       * @param[in] svdpi_init
       *   NP+NF length random vector used in SVD power iteration portion of the algorithm
       */
      __device__ void inline trf_bounds(float (&x)[NP], const float (&observations)[NF], const float (&n)[NF][NX], const float (&lb)[NP], const float (&ub)[NP], const float (&svdpi_init)[NP+NF])
      {

        const float ftol = 1e-8;
        const float gtol = 1e-8;
        const float xtol = 1e-8;

        int nfev = 1;
        int njev = 1;
        const int max_nfev = 300;

        float f[NF]; // array of residuals
        float J[NF][NP]; // Jacobian matrix
        float g[NP] {0.f}; // gradient
        float cost = 0.f;

        // Compute initial gradient, residuals, cost
        // gradient g = transpose(J) * r
        for (int k=0; k<NF; k++)
        {
          float y_est, dy[NP];
          OPT_FUNC::f(x, n[k], y_est, dy);
          f[k] = y_est - observations[k];
          cost += 0.5 * f[k]*f[k];
          for (int l=0; l<NP; l++)
          {
            J[k][l] = dy[l];
            g[l] += dy[l]*f[k];
          }
        }

        float v[NP]; // scaling vector
        float dv[NP]; // derivative of scaling vector with respect to x
        CL_scaling_vector(x, g, lb, ub, v, dv);

        float Delta = trf_invsqrtscaled_norm(x, v);
        if (Delta == 0)
        {
          Delta = 1.f;
        }

        float alpha = 0.f; // Levenberg-Marquartd parameter

        int termination_status = 0;
        int iteration = 0;
        float step_norm = 0.1f;
        float actual_reduction = 0.f;

        if (VERBOSE == 2)
        {
          print_header_nonlinear();
        }

        while (1)
        {
          CL_scaling_vector(x, g, lb, ub, v, dv);

          float g_norm = trf_scaled_norm(g, v);

          if (g_norm < gtol)
          {
            termination_status = 1;
          }

          // print iteration details
          if (VERBOSE == 2)
          {
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction, step_norm, g_norm, alpha);
          }

          if ((termination_status > 0) || (nfev >= max_nfev))
          {
            break; // out of while loop
          }

          float d[NP];
          float diag_h[NP];
          float g_h[NP];
          for (int k=0; k<NP; k++)
          {
            d[k] = __fsqrt_rn(v[k]);
            diag_h[k] = g[k] * dv[k];

            // "hat" gradient
            g_h[k] = d[k] * g[k];
          }

          // Solve subproblem.
          float f_augmented[NF+NP] {0.f};
          for (int k=0; k<NF; k++)
          {
            f_augmented[k] = f[k];
          }
          float J_augmented[NF+NP][NP] {0.f};
          float J_h[NF][NP];
          for (int m=0; m<NF; m++)
          {
            for (int n=0; n<NP; n++)
            {
              J_augmented[m][n] = J[m][n] * d[n];
              J_h[m][n] = J_augmented[m][n];
            }
          }
          for (int m=0; m<NP; m++)
          {
            J_augmented[NF+m][m] = sqrtf(diag_h[m]);
          }
          float U[NF+NP][NP];
          float s[NP];
          float VT[NP][NP];
          matx::st::svdpi(J_augmented, U, s, VT, svdpi_init, NP, 100);

          float uf[NP];
          matmul_ATxB(uf,U,f_augmented);

          float theta = cuda::std::max(0.995f, 1.f - g_norm);
          float norm_x = norm(x);

          actual_reduction = -1.f;
          float p_h[NP];
          float p[NP];
          int n_iter;
          float x_new[NP];
          float f_new[NF];
          float cost_new;
          float J_new[NF][NP];
          float g_new[NF];
          while ((actual_reduction < 0.f) && (nfev < max_nfev))
          {
            solve_lsq_trust_region(NP, NF, uf, s, VT, Delta, alpha, p_h, n_iter);

            float step[NP];
            float step_h[NP];
            float predicted_reduction;
            for (int k=0; k<NP; k++)
            {
              p[k] = d[k] * p_h[k]; // Trust-region solution in the original space
            }
            bool fatal_error;
            select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta, step, step_h, predicted_reduction, fatal_error);
            if (fatal_error) return;

            // make_strictly_feasible(x + step, lb, ub, x_new);
            for (int k=0; k<NP; k++)
            {
              x_new[k] = std::max(std::min(x[k] + step[k], ub[k]), lb[k]);
            }

            cost_new = 0.f;
            for (int k=0; k<NP; k++)
            {
              g_new[k] = 0.f;
            }
            for (int k=0; k<NF; k++)
            {
              float y_est, dy[NP];
              OPT_FUNC::f(x_new, n[k], y_est, dy);
              f_new[k] = y_est - observations[k];
              cost_new += 0.5 * f_new[k]*f_new[k];
              for (int l=0; l<NP; l++)
              {
                J_new[k][l] = dy[l];
                g_new[l] += dy[l]*f_new[k];

              }
            }
            nfev++;

            float step_h_norm = norm(step_h);

            bool all_finite = true;
            for (int k=0; k<NF; k++)
            {
              if (cuda::std::isnan(f_new[k]))
              {
                all_finite = false;
                break;
              }
            }
            if (all_finite == false)
            {
              Delta = 0.25 * step_h_norm;
              continue;
            }

            actual_reduction = cost - cost_new;

            float Delta_new;
            float ratio;
            update_tr_radius(Delta, actual_reduction, predicted_reduction, step_h_norm, step_h_norm > (0.95f * Delta), Delta_new, ratio);

            step_norm = norm(step);

            termination_status = check_termination(actual_reduction, cost, step_norm, norm_x, ratio, ftol, xtol);
            if (termination_status > 0)
            {
              break; // out of while loop
            }

            alpha *= Delta / Delta_new;
            Delta = Delta_new;
          }

          if (actual_reduction > 0)
          {
            // Update parameter estimates and gradient
            for (int k=0; k<NP; k++)
            {
              x[k] = x_new[k];
              g[k] = g_new[k];
            }
            norm_x = norm(x);

            // Update residuals, Jacobian, and gradient
            for (int k=0; k<NF; k++)
            {
              f[k] = f_new[k];
              for (int l=0; l<NP; l++)
              {
                J[k][l] = J_new[k][l];
              }
            }
            cost = cost_new;
            njev++;
          }
          else
          {
            step_norm = 0;
            actual_reduction = 0;
          }

          iteration++;
        }

        if (VERBOSE > 0)
        {
          printf("Final x:\n"); matprint(x);
        }
      }
    };
  }; // namespacd st
}; // namespace matx