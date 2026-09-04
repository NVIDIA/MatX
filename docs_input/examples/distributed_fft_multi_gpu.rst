Single-Node Multi-GPU FFT
#########################

``distributed_fft_multi_gpu`` demonstrates one process using several GPUs on a
single node. It creates a block-distributed rank-1 tensor and runs a MatX
``fft`` expression through the CUDA Toolkit's cuFFT Xt/Mg backend. MPI, NCCL,
cuFFTMp, and NVSHMEM are not used by this example.

Build the example with::

  cmake -S . -B build \
        -DMATX_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target distributed_fft_multi_gpu

The first argument selects how many visible GPUs to use. The second optional
argument selects the global transform length::

  ./build/examples/distributed_fft_multi_gpu 4 1048576

The GPU count defaults to two and must be at least two. The program uses device
ordinals starting at zero, so the requested number cannot exceed the GPUs
visible to the process. The transform length must be supported by cuFFT's
multi-GPU backend.

The input is a unit impulse at global index zero. The example verifies that
every distributed output element is one and reports the maximum error.
