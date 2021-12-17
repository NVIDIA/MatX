#include <matx.h>
#include "matx_viz.h"

using namespace matx;

int main() {
  auto gil = pybind11::scoped_interpreter{}; 

  tensorShape_t<1> shape({10});
  tensor_t<float, 1> B(shape);

  // TODO: Set tensor B such that it forms a Hamming window
  (B = hamming_x(shape)).run();

  viz::line(B, "Hamming Window", "Sample", "Amplitude", "hamming.html");

  return 0;
}
