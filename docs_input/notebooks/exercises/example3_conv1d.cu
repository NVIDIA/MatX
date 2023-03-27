#include <matx.h>

using namespace matx;

int main() {
  auto C = make_tensor<float>({16});
  auto filt = make_tensor<float>({3});
  auto Co = make_tensor<float>({16 + filt.Lsize() - 1});

  filt.SetVals({1.0/3, 1.0/3, 1.0/3});

  randomGenerator_t<float> randData(C.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<1>({16}, NORMAL);
  (C = randTensor1).run();  

  printf("Initial C tensor:\n");
  print(C);

  // TODO: Perform a 1D direct convolution on C with filter filt
  

  printf("After conv1d:\n");
  print(Co);
  return 0;
}
