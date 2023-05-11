#/bin/sh

nvcc --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_80,code=[compute_80,sm_80] --threads 0 -std=c++17 -DINDEX_64_BIT -I/matx_libs/matx-main/examples -I/matx_libs/matx-main/include/matx/kernels -I/matx_libs/matx-main/include/matx -I/usr/include/python3.8 -DMATX_ROOT=\"/repro/matx\" --expt-relaxed-constexpr -DENABLE_CUTLASS=0 -o /tmp/$1 exercises/$1.cu -lcufft -lcublas -lcublasLt /usr/lib/x86_64-linux-gnu/libpython3.8.so && 
/tmp/$1