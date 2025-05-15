#include "matx.h"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>

using namespace matx;

/**
 * FFT Convolution
 *
 * This example shows how to perform an FFT convolution using the MatX library.
 * The example shows the convolution theorem of:
 *
 * \f(h*x \leftrightarrow H \cdot X$  \f)
 *
 * Namely, a convolution in the time domain is a point-wise multiplication in
 * the frequency domain. In this example we start with two signals in the time
 * domain, convert them to frequency domain, perform the multiply, then convert
 * them back to the time domain. This should give very close results to
 * performing a direct convolution in the time domain, so the results are
 * compared to a direct convolution. They will not match identically since the
 * types and order of operations are different, but they will match within a
 * close margin.
 *
 * FFT convolution is frequently used in signal processing when a signal or
 * filter is larger than a threshold, since it will outperform direct
 * convolution past this threshold. Another benefit of FFT convolution is the
 * number of operations is the same, regardless of the filter size. This allows
 * a user to FFT a very long filter one time, and that buffer can be used many
 * times for any incoming samples.
 *
 * For smaller signal sizes, the FFT convolution typically performs worse since
 * there is some buffer and 3 FFT operations (2 for FFT of signal and filter,
 * and 1 IFFT after the multiply) that causes the setup time to dominate.
 *
 * Note that the conv1d() operator has a mode to perform FFT-based convolution
 * automatically.
 *
 */
int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  
  [[maybe_unused]] int batches = 2;
  using complex = cuda::std::complex<float>;
  //  using TestType = float;
  //    using ComplexType = detail::complex_from_scalar_t<TestType>;
  //    auto c0 = make_tensor<ComplexType>({});  
  // auto t1 = make_tensor<complex>({8});
  // //auto t0 = make_tensor<complex>({});

  // t1.SetVals({{10,-2}, {20,4}, {30,1}, {40,-8}, {50,6}, {60,-3}, {70,6}, {80,3}});

  // (c0 = at(fft(t1), 0)).run();
  // print(c0);
  // print(fft(t1));

  {
    using TestType = cuda::std::complex<float>;

    for (index_t fft_dim = 64; fft_dim <= 256; fft_dim *= 2) {
    

      tensor_t<TestType, 1> av{{fft_dim}};
      tensor_t<TestType, 1> avo{{fft_dim}};
      (av = random<TestType>(av.Shape(), UNIFORM)).run();
print(av);
      (avo = fft(av) * static_cast<typename TestType::value_type>(5.0f)).run();


      print(avo);
    }

  }

  // {
  //   using inner_type = typename inner_op_type_t<TestType>::type;
  //   using complex_type = detail::complex_from_scalar_t<inner_type>;

  //   [[maybe_unused]] const inner_type thresh = static_cast<inner_type>(1.0e-6);
  
  //   // Verify that fftshift1D/ifftshift1D work with nested transforms.
  //   // These tests are limited to complex-to-complex transforms where we have matched
  //   // dimensions and types for the inputs/outputs. Adding tests that include real-to-complex
  //   // or complex-to-real fft compositions is TBD.

  //     const int N1 = 3;
  //     const int N2 = 4;
  
  //     auto t3 = make_tensor<complex_type>({N1});
  //     auto t4 = make_tensor<complex_type>({N2});
  //     auto T3 = make_tensor<complex_type>({N1});
  //     auto T4 = make_tensor<complex_type>({N2});
  
  //     const cuda::std::array<complex_type, N1> t3_vals = {{ { 1.0, 0.0 }, { 2.0, 0.0 }, { 3.0, 0.0 } }};
  //     const cuda::std::array<complex_type, N2> t4_vals = {{ { 1.0, 0.0 }, { 2.0, 0.0 }, { 3.0, 0.0 }, { 4.0, 0.0 } }};
  
  //     for (int i = 0; i < N1; i++) { t3(i) = t3_vals[i]; };
  //     for (int i = 0; i < N2; i++) { t4(i) = t4_vals[i]; };
  
  
  //     (T3 = fftshift1D(fft(t3))).run();    
  //     print
  // }
  // print(b);
  // print(c);
   // (d = fft(a) + fft(b) + fft(c)).run();
  //
  // for (int i = 0; i < 10; i++) {

  //gg.assign_ids(counter);

//(d = a + fft(b*c)).run();

  // }


  //(d = a + b*c).run();
  //(d = fft(a)).run();
  //print(d);
  // using complex = cuda::std::complex<float>;
  // cudaExecutor exec{};

  // index_t signal_size = 1ULL << 16;
  // index_t filter_size = 16;
  // index_t batches = 8;
  // index_t filtered_size = signal_size + filter_size - 1;
  // float separate_ms;
  // float fused_ms;
  // constexpr int iterations = 100;
  // cudaStream_t stream;
  // cudaStreamCreate(&stream);
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // // Create time domain buffers
  // auto sig_time  = make_tensor<complex>({batches, signal_size});
  // auto filt_time = make_tensor<complex>({batches, filter_size});
  // auto time_out  = make_tensor<complex>({batches, filtered_size});

  // // Frequency domain buffers
  // auto sig_freq  = make_tensor<complex>({batches, filtered_size});
  // auto filt_freq = make_tensor<complex>({batches, filtered_size});

  // for (index_t b = 0; b < batches; b++) {
  //   // Fill the time domain signals with data
  //   for (index_t i = 0; i < signal_size; i++) {
  //     sig_time(b,i) = {-1.0f * (2.0f * static_cast<float>(i % 2) + 1.0f) *
  //                           (static_cast<float>(i % 10) / 10.0f) +
  //                       0.1f,
  //                   -1.0f * (static_cast<float>(i % 2) == 0.0f) *
  //                           (static_cast<float>(i % 10) / 5.0f) -
  //                       0.1f};
  //   }
  //   for (index_t i = 0; i < filter_size; i++) {
  //     filt_time(b,i) = {static_cast<float>(i) / static_cast<float>(filter_size),
  //                     static_cast<float>(-i) / static_cast<float>(filter_size) +
  //                         0.5f};
  //   }
  // }

  // // Perform the FFT in-place on both signal and filter
  // for (int i = 0; i < iterations; i++) {
  //   if (i == 1) {
  //     cudaEventRecord(start, stream);
  //   }
  //   (sig_freq = fft(sig_time, filtered_size)).run(exec);
  //   (filt_freq = fft(filt_time, filtered_size)).run(exec);

  //   (sig_freq = sig_freq * filt_freq).run(exec);

  //   // IFFT in-place
  //   (sig_freq = ifft(sig_freq)).run(exec);

  // }

  // cudaEventRecord(stop, stream);
  // exec.sync();
  // cudaEventElapsedTime(&separate_ms, start, stop);

  // for (int i = 0; i < iterations; i++) {
  //   if (i == 1) {
  //     cudaEventRecord(start, stream);
  //   }
  //   (sig_freq = ifft(fft(sig_time, filtered_size) * fft(filt_time, filtered_size))).run(exec);
  // }

  // cudaEventRecord(stop, stream);
  // exec.sync();
  // cudaEventElapsedTime(&fused_ms, start, stop);

  // printf("FFT runtimes for separate = %.2f ms, fused = %.2f ms\n", separate_ms/(iterations-1), fused_ms/(iterations-1));

  // // Now the sig_freq view contains the full convolution result. Verify against
  // // a direct convolution. The conv1d function only accepts a 1D filter, so we
  // // create a sliced view here.
  // auto filt1 = slice<1>(filt_time, {0,0}, {matxDropDim, matxEnd});
  // (time_out = conv1d(sig_time, filt1, matxConvCorrMode_t::MATX_C_MODE_FULL)).run(exec);

  // exec.sync();

  // // Compare signals
  // for (index_t b = 0; b < batches; b++) {
  //   for (index_t i = 0; i < filtered_size; i++) {
  //     if (fabs(time_out(b,i).real() - sig_freq(b,i).real()) > 0.001 ||
  //         fabs(time_out(b,i).imag() - sig_freq(b,i).imag()) > 0.001) {
  //       std::cout <<
  //           "Verification failed at item " << i << ". Direct=" << time_out(b,i).real() << " " << time_out(b,i).imag() << ", FFT=" <<
  //           sig_freq(b,i).real() << " " <<
  //           sig_freq(b,i).imag() << "\n";
  //       return -1;
  //     }
  //   }
  // }

  // std::cout << "Verification successful" << std::endl;

  MATX_EXIT_HANDLER();
}