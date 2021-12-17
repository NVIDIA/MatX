#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({4,5});
  tensor_t<float, 2> A(shape);
  tensor_t<float, 0> MD0;
  tensor_t<float, 0> AD0;

  randomGenerator_t<float> randData(A.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>(shape, NORMAL);
  (A = randTensor1).run();    
  
  // Initialize max and average to 0
  (MD0 = 0).run();
  (AD0 = 0).run();

  // TODO: Perform a max and sum reduction of A into MD0 and AD0, respectively.


  printf("A:\n");
  A.Print();
  printf("Max: %f\n", MD0());
  printf("Sum: %f\n", AD0());  

  return 0;
}
