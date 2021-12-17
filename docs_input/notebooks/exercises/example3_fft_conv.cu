#include <matx.h>

using namespace matx;

int main() {

  using complex = cuda::std::complex<float>;

  index_t signal_size = 16;
  index_t filter_size = 3;
  index_t filtered_size = signal_size + filter_size - 1;

  // Create time domain buffers
  tensor_t<complex, 1> sig_time({signal_size});
  tensor_t<complex, 1> filt_time({filter_size});
  tensor_t<complex, 1> time_out({filtered_size});

  // Frequency domain buffers
  tensor_t<complex, 1> sig_freq({filtered_size});
  tensor_t<complex, 1> filt_freq({filtered_size});

  // Fill the time domain signals with data
  for (index_t i = 0; i < signal_size; i++) {
      sig_time(i) = {-1*(2*(i%2)+1) * ((i%10) / 10.0f) + 0.1f, -1*((i%2) == 0) * ((i%10) / 5.0f) - 0.1f};
  }
  for (index_t i = 0; i < filter_size; i++) {
      filt_time(i) = {(float)i/filter_size, (float)-i/filter_size + 0.5f};
  }

  // Prefetch the data we just created
  sig_time.PrefetchDevice(0);
  filt_time.PrefetchDevice(0);

  // TODO: Perform FFT convolution
  // Perform the FFT in-place on both signal and filter, do an element-wise multiply of the two, then IFFT that output


  // TODO: Perform a time-domain convolution
  

  cudaStreamSynchronize(0);

  // Compare signals
  for (index_t i = 0; i < filtered_size; i++) {
      if (  fabs(time_out(i).real() - sig_freq(i).real()) > 0.001 || 
            fabs(time_out(i).imag() - sig_freq(i).imag()) > 0.001) {
          printf("Verification failed at item %lld. Direct=%f%+.2fj, FFT=%f%+.2fj\n", i,
            time_out(i).real(), time_out(i).imag(), sig_freq(i).real(), sig_freq(i).imag());
          return -1;
      }
  }

  std::cout << "Verification successful" << std::endl;

  return 0;
}
