#include <matx.h>

using namespace matx;

int main() {
  tensor_t<float, 2> A({8, 4});
  tensor_t<float, 2> B({4, 8});
  tensor_t<float, 2> C({8,8});

  randomGenerator_t<float> randData(C.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>({8, 4}, NORMAL);
  auto randTensor2 = randData.GetTensorView<2>({4, 8}, NORMAL);
  (A = randTensor1).run();  
  (B = randTensor2).run();  

  // TODO: Perform a GEMM of C = A*B
  
  printf("A:\n");
  A.Print();
  printf("B:\n");
  B.Print();  
  printf("C:\n");
  C.Print();    

  return 0;
}
