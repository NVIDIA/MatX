#include <matx.h>

using namespace matx;

int main() {

  tensorShape_t<2> shape({2, 3});
  using complex = cuda::std::complex<float>;
  tensor_t<complex, 2> A(shape);
  tensor_t<complex, 2> B(shape);

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
  randomGenerator_t<complex> rgen(A.TotalSize(), 12345);
  auto rt = rgen.GetTensorView<2>(A.Shape(), NORMAL);
  (A = rt).run();

  auto At = A.Permute({1, 0});
  fft(At, At);
  /*** End editing ***/

  // Verify init is correct
  B.SetVals({{{0.5927, -0.3677}, {-2.6895, 1.8154}, {-0.0129, 0.9246}},
       {{0.5646, 0.8638}, {1.6400, 0.3494}, {-0.5709, 0.5919}}});
  print(A);
  print(B);
  cudaStreamSynchronize(0);
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
  tensor_t<float, 3> dv({10, 5, 15});
  randomGenerator_t<float> rgen2(dv.TotalSize(), 0);
  auto rt2 = rgen2.GetTensorView<3>(dv.Shape(), NORMAL);

  (dv = rt2).run();

  tensor_t<float, 0> redv;
  rmax(redv, dv, 0);
  (dv = dv / redv).run();
  rmax(redv, dv, 0);
  /*** End editing ***/

  cudaStreamSynchronize(0);
  // Verify init is correct
  if (fabs(redv() - 1.0) > 0.001) {
    printf("Mismatch on final reduction. Expected=1.0, actual = %f\n", redv());
    exit(-1);
  }

  printf("Reduction verification passed!\n");

  return 0;
}
