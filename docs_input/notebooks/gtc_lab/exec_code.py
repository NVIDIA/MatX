import subprocess
import sys
import os 

with open('/tmp/user_code.cu', 'r') as file:
    user_code = file.read()


output_code = f"""
#include <matx.h>

int main() {{
    {user_code}
}}
"""

with open("/tmp/output.cu", "w") as f:
    f.write(output_code)
with open("/tmp/output.cpp", "w") as f:
    f.write(output_code)    

current_dir = os.getcwd()

MATX_ROOT = '/opt/xeus/cling/tools/Jupyter/kernel/MatX'
MATX_ROOT = '/MatX'


gcc_cmd = f'g++ -std=c++17 -DMATX_DISABLE_CUB_CACHE -DMATX_ENABLE_FILEIO -DMATX_ENABLE_PYBIND11 -DMATX_EN_OMP -DMATX_EN_X86_FFTW -DMATX_NVTX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_DISABLE_ABI_NAMESPACE '\
           f'-I"{MATX_ROOT}/build/_deps/cccl-src/libcudacxx/lib/cmake/libcudacxx/../../../include" -I/usr/local/cuda/include -I{MATX_ROOT}/include -I{MATX_ROOT}/include/matx/kernels -I"{MATX_ROOT}/build/_deps/cccl-src/thrust/thrust/cmake/../.." -I"{MATX_ROOT}/build/_deps/cccl-src/cub/cub/cmake/../.." '\
           f'-isystem "{MATX_ROOT}/build/_deps/pybind11-src/include" -isystem /usr/include/python3.10 -isystem /usr/local/cuda/include'\
           f'-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -fopenmp -DMATX_ROOT=\"\" -fvisibility=hidden -lcuda -lcufft -lcublas -lpybind11 -o /tmp/output /tmp/output.cpp'

nvcc_cmd = f'nvcc -forward-unknown-to-host-compiler -Ofc -std=c++17 -DMATX_DISABLE_CUB_CACHE -DMATX_ENABLE_FILEIO -DMATX_ENABLE_PYBIND11 -DMATX_EN_OMP -DMATX_EN_X86_FFTW -DMATX_NVTX_FLAGS -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_DISABLE_ABI_NAMESPACE --generate-code=arch=compute_80,code=[sm_80] '\
            f'-I{MATX_ROOT}/include -I{MATX_ROOT}/include/matx/kernels -I"{MATX_ROOT}/build/_deps/cccl-src/lib/cmake/thrust/../../../thrust" -I"{MATX_ROOT}/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include" -I"{MATX_ROOT}/build/_deps/cccl-src/lib/cmake/cub/../../../cub" -isystem "{MATX_ROOT}/build/_deps/pybind11-src/include" -isystem /usr/include/python3.10 -isystem "/usr/include/x86_64-linux-gnu/openblas64-openmp" -isystem "{MATX_ROOT}/build/_deps/cutensor-src/include" -isystem "{MATX_ROOT}/build/_deps/cutensornet-src/include" -isystem "{MATX_ROOT}/build/_deps/cudss-src/include" -isystem /usr/local/cuda/include'\
            f'-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -fopenmp -DMATX_ROOT=\"\" -fvisibility=hidden -lcuda -lcufft -lcublas -o /tmp/output /tmp/output.cu'


print(nvcc_cmd)
print('Compiling...')
compile_process = subprocess.run(nvcc_cmd.split(), capture_output=True, text=True)

# Check for compilation errors
if compile_process.returncode != 0:
    print("Compilation failed:")
    print(compile_process.stderr)
    sys.exit(1)

# Run the compiled executable and capture its output
run_process = subprocess.run([f"/tmp/output"], capture_output=True, text=True)

# Print the output of running the executable
print(run_process.stdout)