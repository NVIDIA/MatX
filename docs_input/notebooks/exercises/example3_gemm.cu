#include <matx.h>

using namespace matx;

int main() {
  auto A = make_tensor<float>({8, 4});
  auto B = make_tensor<float>({4, 8});
  auto C = make_tensor<float>({8, 8});

  randomGenerator_t<float> randData(C.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>({8, 4}, NORMAL);
  auto randTensor2 = randData.GetTensorView<2>({4, 8}, NORMAL);
  (A = randTensor1).run();  
  (B = randTensor2).run();  

  // TODO: Perform a GEMM of C = A*B
  
  printf("A:\n");
  print(A);
  printf("B:\n");
  print(B);  
  printf("C:\n");
  print(C);    

  return 0;
}
