#include <matx.h>

using namespace matx;

int main() {
  auto C = make_tensor<float>({8,8});
  auto filt = make_tensor<float>({3});
  auto Co = make_tensor<float>({16 + filt.Lsize() - 1});

  auto filt = ones<float>({2, 2});
  auto Co = make_tensor<float>({8 + filt.Size(0) - 1, 8 + filt.Size(1) - 1});

  randomGenerator_t<float> randData(C.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>({8,8}, NORMAL);
  (C = randTensor1).run();  

  printf("Initial C tensor:\n");
  print(C);

  // TODO: Perform a 2D direct convolution on C with filter filt

  

  printf("After conv2d:\n");
  print(Co);

  return 0;
}
