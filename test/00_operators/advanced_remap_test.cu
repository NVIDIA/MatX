#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

TEST(OperatorTestsAdvanced, AdvancedRemapOp)
{
  typedef cuda::std::complex<float> complex;
  cudaExecutor exec{};
  MATX_ENTER_HANDLER();

  int I = 4;
  int J = 4;
  int K = 14;
  int L = 133;

  int F = 4096;
  int P = 288;

  int M = 2;

  auto idx = matx::make_tensor<int, 1>({M});

  idx(0) = 1;
  idx(1) = 3;

  auto A = matx::make_tensor<complex, 4>({I, J, K, L});
  //collapsed tensor
  auto B = matx::make_tensor<complex, 2>({I * M * K, L});

  auto index = [&] (int i, int j, int k, int l) {
    return i * J * K * L +
      j * K * L +
      k * L +
      l;
  };
  for (int i = 0; i < I ; i++) {
    for (int j = 0; j < J ; j++) {
      for (int k = 0; k < K ; k++) {
        for (int l = 0; l < L ; l++) {
          float val = (float)index(i,j,k,l);
          A(i,j,k,l) = complex(val, val/100);
        }
      }
    }
  }

  (B = 0).run(exec);

  auto rop = remap<1>(A, idx);
  auto lop = lcollapse<3>(rop);

  ASSERT_EQ(lop.Rank() , 2);
  ASSERT_EQ(lop.Size(1) , A.Size(3));
  ASSERT_EQ(lop.Size(0) , I * M * K);

  (B = lop).run(exec);

  exec.sync(); 

  for (int i = 0; i < I; i++) {
    for (int m = 0; m < M; m++) {
      for (int k = 0; k < K; k++) {
        for (int l = 0; l < L; l++) {
          int j = idx(m);
          int fidx = i * M * K + m * K  + k;
          float val = (float)index(i,j,k,l);
          complex expected_val = complex(val,val/100);
          complex a_val = A(i,j,k,l);
          complex b_val = B(fidx, l);	  
          complex lop_val = lop(fidx, l);
          complex rop_val = rop(i, m, k, l);

          ASSERT_EQ(a_val, expected_val);
          ASSERT_EQ(rop_val, expected_val);
          ASSERT_EQ(lop_val, expected_val);
          ASSERT_EQ(b_val, expected_val);

          ASSERT_EQ(B(fidx, l) , lop(fidx, l));
        }
      }
    }
  }


  // convolution test
  auto O1 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});
  auto O2 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});
  auto O3 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});
  auto O4 = matx::make_tensor<complex, 4>({I, J, K, F + P + L - 1});

  auto C = matx::make_tensor<complex, 3>({I, K, F + P});
  //collapsed tensor
  auto D = matx::make_tensor<complex, 2>({I * M * K, F + P});
  
  auto indexc = [&] (int i, int j, int k) {
    return i * C.Size(1) * C.Size(2) +
      j * C.Size(2) +
      k;
  };
  
  for (int i = 0; i < I ; i++) {
    for (int j = 0; j < J ; j++) {
      for (int k = 0; k < K ; k++) {
        float val = (float) indexc(i,j,k);
        C(i,j,k) = complex(val, val/100);
      }
    }
  }

  exec.sync();

  auto o1op = lcollapse<3>(remap<1>(O1, idx));
  auto o2op = lcollapse<3>(remap<1>(O2, idx));
  auto o3op = lcollapse<3>(remap<1>(O3, idx));
  auto o4op = lcollapse<3>(remap<1>(O4, idx));

  auto cop = C.Clone<4>({matxKeepDim, M, matxKeepDim, matxKeepDim});
  auto rcop = lcollapse<3>(remap<1>(cop, idx));

  (O1 = 1).run(exec);
  (O2 = 2).run(exec);
  (O3 = 3).run(exec);
  (O4 = 4).run(exec);
  
  (B = lop).run(exec);
  (D = rcop).run(exec);

  // two operators as input
  (o1op = conv1d(lop, rcop, matx::matxConvCorrMode_t::MATX_C_MODE_FULL)).run(exec);

  // one tensor and one operators as input
  (o2op = conv1d(B, rcop, matx::matxConvCorrMode_t::MATX_C_MODE_FULL)).run(exec);
  
  // one tensor and one operators as input
  (o3op = conv1d(lop, D, matx::matxConvCorrMode_t::MATX_C_MODE_FULL)).run(exec);
  
  //two tensors as input
  (o4op = conv1d(B, D, matx::matxConvCorrMode_t::MATX_C_MODE_FULL)).run(exec);

  exec.sync();

  for (int i = 0; i < o1op.Size(0); i++) {
    for (int l = 0; l < o1op.Size(1); l++) {
      ASSERT_EQ(o1op(i,l), o2op(i,l));
      ASSERT_EQ(o2op(i,l), o3op(i,l));
      ASSERT_EQ(o3op(i,l), o4op(i,l));
    }
  }

  MATX_EXIT_HANDLER();
}