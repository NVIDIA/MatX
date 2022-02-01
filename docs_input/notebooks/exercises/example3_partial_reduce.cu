#include <matx.h>

using namespace matx;

int main() {

  auto A = make_tensor<float>({4, 5});
  auto MD0 = make_tensor<float>({4});
  auto AD0 = make_tensor<float>({4});

  randomGenerator_t<float> randData(A.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>(shape, NORMAL);
  (A = randTensor1).run();    
  
  // Initialize max and average to 0
  (MD1 = 0).run();
  (AD1 = 0).run();

  // TODO: Reduce all rows of A by max where each reduction is a separate value in the vector MD1



  printf("A:\n");
  A.Print();
  printf("Max:\n");
  MD1.Print();
  printf("Sum:\n");
  AD1.Print();

  return 0;
}
