#include "matx.h"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>

using namespace matx;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  {
    using TestType = cuda::std::complex<float>;
    int size = 32;
    auto exec = cudaExecutor{};

    tensor_t<TestType, 1> av{{size}};
    tensor_t<TestType, 1> av2{{size}};
    tensor_t<TestType, 1> avo{{size}};

    tensor_t<TestType, 2> av2d{{4096,size}};
    tensor_t<TestType, 2> av22d{{4096, size}};
    tensor_t<TestType, 2> avo2d{{4096, size}};    
    printf("avo data %p\n", avo.Data());

    //(av = random<TestType>(av.Shape(), UNIFORM)).run(exec);
    av.SetVals({{3.9905e-01f,8.8052e-01f},
      {5.1668e-01f,9.3966e-01f},
      {2.4930e-02f,7.1293e-01f},
      {9.4008e-01f,6.7758e-02f},
      {9.4585e-01f,5.5899e-01f},
      {7.9673e-01f,6.6475e-01f},
      {4.1501e-01f,5.7781e-01f},
      {8.2026e-01f,5.8734e-01f},
      {2.2904e-01f,8.9237e-01f},
      {9.0959e-01f,6.5702e-01f},
      {1.1834e-01f,8.5005e-01f},
      {7.5222e-02f,2.6438e-01f},
      {4.0922e-01f,3.7453e-02f},
      {9.6007e-01f,8.8839e-01f},
      {9.6007e-01f,8.8839e-01f},
      {2.0930e-01f,3.7034e-01f},
      {1.9395e-01f,3.8684e-01f},
      {8.9094e-01f,5.4259e-01f},
      {4.3867e-01f,3.0159e-01f},
      {3.5698e-01f,2.5036e-01f},
      {5.4537e-01f,2.0659e-01f},
      {8.2992e-01f,6.5006e-01f},
      {2.0994e-01f,6.4923e-01f},
      {7.6842e-01f,7.4013e-01f},
      {4.2899e-01f,7.3793e-01f},
      {2.1167e-01f,6.2957e-03f},
      {6.6055e-01f,9.9871e-01f},
      {1.6536e-01f,2.4758e-01f},
      {4.2499e-01f,4.1189e-01f},
      {9.9267e-01f,8.8892e-01f},
      {6.9642e-01f,7.2175e-01f},
      {2.4719e-01f,3.5031e-01f},
      {7.0281e-01f,9.4431e-01f}});

      av2.SetVals({{.9905e-01f,8.8052e-01f},
        {.1668e-01f,9.3966e-01f},
        {2.4930e-02f,7.1293e-01f},
        {9.4008e-01f,6.7758e-02f},
        {.4585e-01f,5.5899e-01f},
        {7.9673e-01f,6.6475e-01f},
        {4.1501e-01f,.7781e-01f},
        {8.2026e-01f,5.8734e-01f},
        {2.2904e-01f,8.9237e-01f},
        {.0959e-01f,6.5702e-01f},
        {1.1834e-01f,8.5005e-01f},
        {7.5222e-02f,.6438e-01f},
        {4.0922e-01f,3.7453e-02f},
        {9.6007e-01f,8.8839e-01f},
        {.6007e-01f,8.8839e-01f},
        {2.0930e-01f,3.7034e-01f},
        {1.9395e-01f,.8684e-01f},
        {8.9094e-01f,5.4259e-01f},
        {4.3867e-01f,.0159e-01f},
        {3.5698e-01f,2.5036e-01f},
        {.4537e-01f,2.0659e-01f},
        {8.2992e-01f,.5006e-01f},
        {2.0994e-01f,6.4923e-01f},
        {7.6842e-01f,7.4013e-01f},
        {4.2899e-01f,7.3793e-01f},
        {2.1167e-01f,.2957e-03f},
        {6.6055e-01f,9.9871e-01f},
        {1.6536e-01f,2.4758e-01f},
        {4.2499e-01f,4.1189e-01f},
        {9.9267e-01f,8.8892e-01f},
        {6.9642e-01f,7.2175e-01f},
        {2.4719e-01f,3.5031e-01f},
        {7.0281e-01f,9.4431e-01f}});     
      
    print(av);
    print(av2);

    (av2d = av).run(exec);
    (av22d = av2).run(exec);

    //(avo = 5.0f * ifft(fft(av) * fft(av2))).run(exec);
    for (int i = 0; i < 10; i++) 
    (avo2d = 5.0f * ifft(fft(av2d) * fft(av22d))).run(exec);

    print(avo2d, 1, 0);
  }

  MATX_EXIT_HANDLER();
}