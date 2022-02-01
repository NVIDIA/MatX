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

  t2.PrefetchDevice(0);

  // TODO: reduce tensor t2 to a 1D tensor by pulling the second column and all
  // rows
  auto t1 = ...;

  t1.Print();

  return 0;
}
