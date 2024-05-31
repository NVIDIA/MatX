////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include <matx.h>

using namespace matx;

/**
 * MatX training assignment 3. This training goes through tensor operations that
 * were learned in the 03_transformations notebook. Uncomment each verification
 * block as you go to ensure your solutions are correct.
 */

int main() {
  using complex = cuda::std::complex<float>;
  auto A = make_tensor<complex>({2, 3});
  auto B = make_tensor<complex>({2, 3});

  cudaExecutor exec{};

  /****************************************************************************************************
   * Use the random number generator with a seed of 12345 to generate
   * normally-distributed numbers in the tensor A. Next, take the FFT across
   * columns of A (a 2-element FFT), and store the results in-place back in A.
   * An example of random number generation can be found in the second tutorial
   * or in the quick start guide here:
   *
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/quickstart.html#random-numbers
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/api/random.html
   ****************************************************************************************************/

  /*** End editing ***/

  // Verify init is correct
  B.SetVals({{{0.5927, -0.3677}, {-2.6895, 1.8154}, {-0.0129, 0.9246}},
       {{0.5646, 0.8638}, {1.6400, 0.3494}, {-0.5709, 0.5919}}});
  A.print();
  B.print();
  exec.sync();
  for (int row = 0; row < A.Size(0); row++) {
    for (int col = 0; col < A.Size(1); col++) {
      if (fabs(A(row, col).real() - B(row, col).real()) > 0.001) {
        printf(
            "Mismatch in real part of FFT view! actual = %f, expected = %f\n",
            A(row, col).real(), B(row, col).real());
        exit(-1);
      }
      if (fabs(A(row, col).imag() - B(row, col).imag()) > 0.001) {
        printf(
            "Mismatch in imag part of FFT view! actual = %f, expected = %f\n",
            A(row, col).imag(), B(row, col).imag());
        exit(-1);
      }
    }
  }

  printf("FFT verification passed!\n");

  /****************************************************************************************************
   * Create a 3D tensor of floats using a normal distribution and with shape
   * 10x5x15. Reduce the entire tensor down to a single float containing the max
   * value. Scale the original tensor by this max value and do another max
   * reduction. The final reduction should be 1.0.
   *
   * Hint: the reduction function is named rmax and takes the output, input, and
   * stream as parameters
   * https://devtech-compute.gitlab-master-pages.nvidia.com/matx/api/reduce.html
   ****************************************************************************************************/
  // Create and initialize 3D tensor

  // Create scalar tensor for reduction
  tensor_t<float, 0> redv;

  /*** End editing ***/

  // Verify init is correct
  exec.sync();
  if (fabs(redv() - 1.0) > 0.001) {
    printf("Mismatch on final reduction. Expected=1.0, actual = %f\n", redv());
    exit(-1);
  }

  printf("Reduction verification passed!\n");

  return 0;
}
