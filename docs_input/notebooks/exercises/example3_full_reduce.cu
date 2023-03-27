#include <matx.h>

using namespace matx;

int main() {

  auto A = make_tensor<float>({4, 5});
  auto MD0 = make_tensor<float>();
  auto AD0 = make_tensor<float>();

  randomGenerator_t<float> randData(A.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>({4,5}, NORMAL);
  (A = randTensor1).run();    
  
  // Initialize max and average to 0
  (MD0 = 0).run();
  (AD0 = 0).run();

  // TODO: Perform a max and sum reduction of A into MD0 and AD0, respectively.


  printf("A:\n");
  print(A);
  printf("Max: %f\n", MD0());
  printf("Sum: %f\n", AD0());  

  return 0;
}
