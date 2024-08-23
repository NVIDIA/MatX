////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2024, NVIDIA Corporation
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

// Single threaded algorithms are useful building blocks for running many
// small problems, one per thread, with a fused kernel.
//
// This program demonstrates some uses of some single threaded algorithms
//   - SVD (power iteration method)
//   - Gauss Newton nonlinear least squares solver
//   - Trust Region Reflective nonlinear least squares solver

#include "matx.h"
#include "matx/singlethread/trf.h"
#include "matx/singlethread/svdpi.h"
#include "matx/singlethread/gn.h"

#define CHECK_CUDA(result)                              \
    if((cudaError_t)result != cudaSuccess)              \
    {                                                   \
        printf("\n\nCUDA Runtime Error: %s:%d:%s\n\n",  \
                __FILE__,                               \
                __LINE__,                               \
                cudaGetErrorString(result));            \
        exit(1);                                        \
    }


constexpr int NP = 3;
constexpr int NX = 2;
constexpr int NF = 81;
constexpr int VERBOSE = 2;

// Calculate the optimization function and its 3 partial derivatives
// The function is
// y = x[2] * sin((n[0] - x[0]) * V / N * pi) * np.sin((n[1] - x[1]) * V / N * pi) /
//              (sin((n[0] - x[0]) * pi / N) * sin((n[1] - x[1]) * pi / N) * (N^2))
// Note: Care must be taken to avoid 0/0 when n[0]=x[0] or n[1]=x[1]
__device__ inline void opt_func(
      const float (&x)[NP],
      const float (&n)[NX],
      float& y,
      float (&dy)[NP])
{
  constexpr float N = 64;
  constexpr float V = N / 2.f + 1.f;
  const float PI = static_cast<float>(M_PI);

  float dx0 = n[0] - x[0];
  float dx1 = n[1] - x[1];

  float g1 = sin(dx0 * PI / N);
  float f1 = g1 ? sin(dx0 * V / N * PI) : V;
  if (g1 == 0) g1 = 1.f;
  float f1prime = -cos(dx0 * V / N * PI) * V / N * PI;
  float g1prime = -cos(dx0 * PI / N) * PI / N;

  float g2 = sin(dx1 * PI / N);
  float f2 = g2 ? sin(dx1 * V / N * PI) : V;
  if (g2 == 0) g2 = 1.f;
  float f2prime = -cos(dx1 * V / N * PI) * V / N * PI;
  float g2prime = -cos(dx1 * PI / N) * PI / N;
  y = x[2] * f1 * f2 / (g1 * g2 * N * N);
  dy[0] = x[2] * f2 * (f1prime*g1 - f1*g1prime) / (g1*g1 * g2 * N * N);
  dy[1] = x[2] * f1 * (f2prime*g2 - f2*g2prime) / (g2*g2 * g1 * N * N);
  dy[2] = f1 * f2 / (g1 * g2 * N * N);
}


class trf_derived : public matx::st::trf_base<trf_derived,NP,NX,NF,VERBOSE>
{
public:
  static __device__ void f(
      const float (&x)[NP],
      const float (&n)[NX],
      float& y,
      float (&dy)[NP])
  {
    opt_func(x, n, y, dy);
  }
};

class gn_derived : public matx::st::gn_base<gn_derived,NP,NX,NF,VERBOSE>
{
public:
  static __device__ void f(
      const float (&x)[NP],
      const float (&n)[NX],
      float& y,
      float (&dy)[NP])
  {
    opt_func(x, n, y, dy);
  }

  __device__ inline void apply_bounds(float (&x)[NP])
  {
    constexpr float lb[NP] {-1.f, -1.f, 0.f};
    constexpr float ub[NP] {1.f, 1.f, 1.f};

    for (int k=0; k<NP; k++)
    {
      x[k] = cuda::std::min(cuda::std::max(x[k], lb[k]), ub[k]);
    }
  }
};

template<typename T_1D, typename T_2D>
__global__ void launch_trf_bounds(T_1D t_x, T_1D t_observations, T_2D t_n, T_1D t_svdpi)
{
  float (&x)[NP] = *reinterpret_cast<float(*)[NP]>(t_x.Data());
  float (&observations)[NF] = *reinterpret_cast<float(*)[NF]>(t_observations.Data());
  float (&n)[NF][NX] = *reinterpret_cast<float(*)[NF][NX]>(t_n.Data());
  float (&svdpi_init)[NF+NP] = *reinterpret_cast<float(*)[NF+NP]>(t_svdpi.Data());

  const float lb[NP] {-1.f, -1.f, 0.f};
  const float ub[NP] {1.f, 1.f, 1.f};

  trf_derived trf;
  trf.trf_bounds(x, observations, n, lb, ub, svdpi_init);
}

template<typename T_1D, typename T_2D>
__global__ void launch_gn(T_1D t_x, T_1D t_observations, T_2D t_n)
{
  float (&x)[NP] = *reinterpret_cast<float(*)[NP]>(t_x.Data());
  float (&observations)[NF] = *reinterpret_cast<float(*)[NF]>(t_observations.Data());
  float (&n)[NF][NX] = *reinterpret_cast<float(*)[NF][NX]>(t_n.Data());

  gn_derived gn;
  gn.solve(x, observations, n);
}

int main(void)
{
  MATX_ENTER_HANDLER();
  int mismatches = 0;

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  printf("\n----------------------------------------------------------------------\n");
  printf("SVD Power Iteration Results:\n");

  auto A = matx::make_tensor<float>({3,3});
  A.SetVals({
    {-0.9236981 , -1.87814045,  1.59373179},
    {1.50496255,  0.17626143, -0.70076442},
    {1.54462882,  0.77728911,  1.1411755}
  });

  auto U = matx::make_tensor<float>({3,3});
  auto VT = matx::make_tensor<float>({3,3});
  auto S = matx::make_tensor<float>({3});

  auto t_svd_x0 = matx::make_tensor<float>({3});
  (t_svd_x0 = matx::random<float>({3}, matx::NORMAL)).run(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  {
    float (&Amat)[3][3] = *reinterpret_cast<float(*)[3][3]>(A.Data());
    float (&Umat)[3][3] = *reinterpret_cast<float(*)[3][3]>(U.Data());
    float (&Svec)[3] = *reinterpret_cast<float(*)[3]>(S.Data());
    float (&VTmat)[3][3] = *reinterpret_cast<float(*)[3][3]>(VT.Data());
    float (&x0vec)[3] = *reinterpret_cast<float(*)[3]>(t_svd_x0.Data());
    matx::st::svdpi(Amat, Umat, Svec, VTmat, x0vec, 3, 100);
  }

  auto US = matx::make_tensor<float>({3,3});
  (US = U * S).run(stream);

  auto Ahat = matx::make_tensor<float>({3,3});
  (Ahat = matx::matmul(US, VT)).run(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  printf("U\n"); matx::print(U);
  printf("S\n"); matx::print(S);
  printf("VT\n"); matx::print(VT);
  printf("Ahat\n"); matx::print(Ahat);
  for (int m=0; m<3; m++)
  {
    for (int n=0; n<3; n++)
    {
      const float EPS = 1e-5f;
      float delta = fabs(Ahat(m,n) - A(m,n));
      if (delta > EPS)
      {
        printf("Mismatch on Ahat at position %d,%d\n",m,n);
        mismatches++;
      }
    }
  }

  printf("\n----------------------------------------------------------------------\n");
  printf("TRF Nonlinear Solver Results:\n");

  auto t_x_golden = matx::make_tensor<float>({NP});
  t_x_golden.SetVals({0.57102037, -0.14885496,  0.58603605});

  auto t_observations = matx::make_tensor<float>({NF});
  t_observations.SetVals({
        -0.00507214121, -0.00550589416, 0.00834280962, -0.0191427465, -0.00878878605, 0.000353395344, 0.028137237, -0.00357987104, -2.60734814e-06,
        -0.0109178063, 0.0195318878, -0.000491019838, 0.010030876, -0.00613712444, 0.0167583988, -0.00142493236, 0.011332446, 0.000318457392,
        0.00973992052, -0.0177684134, 0.0108894162, 0.0273508687, -0.00793719719, 0.0131566006, -0.0148512754, -0.0066955701, 0.00318006582,
        0.0268852783, -0.00242866299, -0.00960003583, -0.027159622, -0.0277969892, -0.0456425489, -0.0450366133, 0.00304451233, 0.00906408233,
        0.00508507958, -0.0185903388, 0.0303496681, 0.159540409, 0.274478441, 0.16017877, 0.0450214611, 0.012920898, 0.0381575004,
        -0.00744066275, -0.0270813114, -0.0267409821, 0.0333024225, 0.105146204, -0.0404875266, -0.0470864699, 0.00208025889, -0.0186689355,
        -0.00349110552, -0.0188402975, 0.00597009396, 0.0605763849, 0.0958322919, 0.0658311629, -0.00490061378, 0.00185717984, -0.0120444051,
        0.00914534504, 0.0185889794, 0.0122738978, 0.0146749764, 0.0137274759, 0.0622846276, 0.0136236457, -0.0187234101, 0.00162107397,
        0.0189116464, -0.0133973833, -0.0150557874, 0.0171880876, -0.0137607785, -0.017539094, 0.0229248365, 0.00258490441, 0.0297245443,
  });

  auto t_x = matx::make_tensor<float>({NP});
  auto t_n = matx::make_tensor<float>({NF,NX});
  auto t_svdpi_init = matx::make_tensor<float>({NP+NF});

  // Set initial parameters
  t_x.SetVals({0.001f, 0.001f, 0.1f});
  for (int k=0; k<NF; k++)
  {
    t_n(k,0) = static_cast<float>((k / 9) - 4);
    t_n(k,1) = static_cast<float>((k % 9) - 4);
  }
  (t_svdpi_init = matx::random<float>({NP+NF}, matx::NORMAL)).run(stream);

  // Run TRF optimizer
  launch_trf_bounds<<<1,1>>>(t_x, t_observations, t_n, t_svdpi_init);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  printf("t_x:\n"); matx::print(t_x);

  for (int k=0; k<NP; k++)
  {
    const float EPS = 1e-2f;
    float delta = fabs(t_x(k)- t_x_golden(k));
    if (delta > EPS)
    {
      printf("Mismatch at position %d\n",k);
      mismatches++;
    }
  }


  printf("\n----------------------------------------------------------------------\n");
  printf("GN Nonlinear Solver Results:\n");

  // Set initial parameters
  t_x.SetVals({0.001f, 0.001f, 0.1f});

  // Run GN optimizer
  launch_gn<<<1,1>>>(t_x, t_observations, t_n);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  printf("t_x:\n"); matx::print(t_x);

  for (int k=0; k<NP; k++)
  {
    const float EPS = 1e-2f;
    float delta = fabs(t_x(k)- t_x_golden(k));
    if (delta > EPS)
    {
      printf("Mismatch at position %d\n",k);
      mismatches++;
    }
  }

  if (mismatches == 0)
  {
    printf("Test vector matching PASS.\n");
  }
  else
  {
    printf("Test vector matching had %d errors.\n", mismatches);
  }

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
  return 0;
}