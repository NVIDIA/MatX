#include <matx.h>

using namespace matx;

int main() {

  auto C = make_tensor<cuda::std::complex<float>>({2, 4});

  randomGenerator_t < cuda::std::complex<float> randData(C.TotalSize(), 0);
  auto randTensor1 = randData.GetTensorView<2>({2, 4}, NORMAL);
  (C = randTensor1).run();

  printf("Initial C tensor:\n");
  C.Print();

  // TODO: Perform an in-place FFT on C across rows

  printf("After FFT:\n");
  C.Print();

  // TODO: Perform an in-place IFFT on C across rows.

  printf("After IFFT and normalization:\n");
  C.Print();

  return 0;
}
