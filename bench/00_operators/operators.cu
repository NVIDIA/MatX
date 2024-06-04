#include <nvbench/nvbench.cuh>
#include "matx.h"

using namespace matx;

using vec_add_types = nvbench::type_list<float, double, cuda::std::complex<float>, cuda::std::complex<double>>;

/* Vector adding benchmarks */
template <typename ValueType>
void vector_add(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Get current parameters:
  cudaExecutor exec{0};
  const int x_len = static_cast<int>(state.get_int64("Vector size"));

  state.add_element_count(x_len, "NumElements");
  state.add_global_memory_reads<ValueType>(2*x_len, "DataSize");
  state.add_global_memory_writes<ValueType>(x_len);  


  tensor_t<ValueType, 1> xv{{x_len}};
  tensor_t<ValueType, 1> xv2{{x_len}};

  (xv = xv + xv2).run(exec);
  exec.sync();

  state.exec( 
    [&xv, &xv2](nvbench::launch &launch) {
      (xv = xv + xv2).run((cudaStream_t)launch.get_stream());
    });

}

NVBENCH_BENCH_TYPES(vector_add, NVBENCH_TYPE_AXES(vec_add_types))
  .add_int64_power_of_two_axis("Vector size", nvbench::range(22, 28, 1));


using permute_types = nvbench::type_list<float, double, cuda::std::complex<float>, cuda::std::complex<double>>;
template <typename ValueType>
void permute(nvbench::state &state, nvbench::type_list<ValueType>)
{
  cudaExecutor exec{0};
  auto x = make_tensor<ValueType>({1000,200,6,300});
  auto y = make_tensor<ValueType>({300,1000,6,200});

  state.add_element_count(x.TotalSize(), "NumElements");
  state.add_global_memory_reads<ValueType>(x.TotalSize(), "DataSize");
  state.add_global_memory_writes<ValueType>(x.TotalSize());    

  x.PrefetchDevice(0);
  exec.sync();

  state.exec( 
    [&x, &y](nvbench::launch &launch) {
      (y = x.Permute({3,0,2,1})).run((cudaStream_t)launch.get_stream());
    });
}


NVBENCH_BENCH_TYPES(permute, NVBENCH_TYPE_AXES(permute_types));

using random_types = nvbench::type_list<float, double, cuda::std::complex<float>, cuda::std::complex<double>>;
template <typename ValueType>
void random(nvbench::state &state, nvbench::type_list<ValueType>)
{
  cudaExecutor exec{0};
  auto x = make_tensor<ValueType>({1966800});
  auto y = make_tensor<ValueType>({1966800});
  x.PrefetchDevice(0);
  y.PrefetchDevice(0);

  (y = random<float>(x.Shape(), NORMAL)).run(exec);

  state.add_element_count(x.TotalSize(), "NumElements");
  state.add_global_memory_writes<ValueType>(x.TotalSize());    

  exec.sync();

  state.exec( 
    [&x, &y](nvbench::launch &launch) {
      (x = y).run((cudaStream_t)launch.get_stream());
    });
}

NVBENCH_BENCH_TYPES(random, NVBENCH_TYPE_AXES(random_types));

template<typename T> T factorial(int N) {
  T prod = 1;
  for(int i=2; i<=N; i++) {
    prod = prod * static_cast<T>(i);
  }
  return prod;
}

using sphericalharmonics_types = nvbench::type_list<float, double>;
template <typename ValueType>
void sphericalharmonics(nvbench::state &state, nvbench::type_list<ValueType>)
{
  int l = 5;
  int m = 4;
  int n = 600;
  ValueType dx = static_cast<ValueType>(M_PI/n);
  
  cudaExecutor exec{};
  auto col = range<0>({n+1},ValueType(0), ValueType(dx));
  auto az = range<0>({2*n+1}, ValueType(0), ValueType(dx));

  auto [phi, theta] = meshgrid(az, col);

  auto Plm = lcollapse<3>(legendre(l, m, cos(theta)));

  ValueType a = static_cast<ValueType>(2*l+1)*factorial<ValueType>(l-m);
  ValueType b = static_cast<ValueType>(4*M_PI)*factorial<ValueType>(l+m);
  ValueType C = cuda::std::sqrt(a/b);

  auto Ylm = C * Plm * exp(cuda::std::complex<ValueType>(0,1)*(static_cast<ValueType>(m)*phi));
  auto [ Xm, Ym, Zm ] = sph2cart(phi, ValueType(M_PI)/2 - theta, abs(real(Ylm)));

  // Work around C++17 restriction, structured bindings cannot be captured
  auto XXm = Xm;
  auto YYm = Ym;
  auto ZZm = Zm;

  // Output location
  auto X = make_tensor<ValueType>(Xm.Shape());
  auto Y = make_tensor<ValueType>(Ym.Shape());
  auto Z = make_tensor<ValueType>(Zm.Shape());

  exec.sync();
  
  state.add_element_count(n+1, "Elements");

  state.exec( 
    [&](nvbench::launch &launch) {
      (X=XXm, Y=YYm, Z=ZZm).run((cudaStream_t)launch.get_stream());
    });
}

NVBENCH_BENCH_TYPES(sphericalharmonics, NVBENCH_TYPE_AXES(sphericalharmonics_types));
