#include <matx.h>

using namespace matx;

int main() {

  auto t2 = make_tensor<int>({5, 4});

  // Initialize the tensor linearly
  t2.SetVals({{1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19, 20}});

  // TODO: Create a slide of the view t2 starting at the second element and
  // ending at the third element (inclusive) in both dimensions
  auto t2s = ...;

  print(t2s);

  return 0;
}
