#include <matx.h>

using namespace matx;

int main() {

  tensor_t<int, 2> t2({5, 4});

  // Initialize the tensor linearly
  t2 = {{1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19, 20}};

  t2.PrefetchDevice(0);

  // TODO: Permute the view t2 such that the two dimensions are swapped
  auto t2p = ...;

  t2p.Print();

  return 0;
}
