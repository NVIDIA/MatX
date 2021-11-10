#include <matx.h>

using namespace matx;

int main() {

  tensor_t<float, 2> C({8,8});
  auto filt = ones(tensorShape_t<2>({2, 2}));
  tensor_t<float, 2> Co({8 + filt.Size(0) - 1, 8 + filt.Size(1) - 1});


  randomGenerator_t<float> randData(C.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>({8,8}, NORMAL);
  (C = randTensor1).run();  

  printf("Initial C tensor:\n");
  C.Print();

  // TODO: Perform a 2D direct convolution on C with filter filt

  

  printf("After conv2d:\n");
  Co.Print();

  return 0;
}
