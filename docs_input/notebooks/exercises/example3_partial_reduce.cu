#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({4,5});
  tensor_t<float, 2> A(shape);
  tensor_t<float, 1> MD1({4});
  tensor_t<float, 1> AD1({4});

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
