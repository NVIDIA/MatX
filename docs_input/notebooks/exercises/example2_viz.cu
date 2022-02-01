#include <matx.h>
#include "matx_viz.h"

using namespace matx;

int main() {
  auto B = make_tensor<float>({10});

  // TODO: Set tensor B such that it forms a Hamming window
  (B = hamming_x(shape)).run();

  viz::line(B, "Hamming Window", "Sample", "Amplitude", "hamming.html");

  return 0;
}
