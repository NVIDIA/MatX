.. _fusion:

Operator Fusion
###############

MatX supports operator fusion for all element-wise operators, and CUDA JIT kernel fusion for math functions with a 
supporting MathDx function. JIT kernel fusion is considered *experimental* currently and may contain bugs that don't 
occur with JIT enabled.

Element-wise Operator Fusion
============================

When writing a simple arithmetic expression like the following:

.. code-block:: cpp

    (A = B * (cos(C) / D)).run();

Using the typical order of operations rules, we evaluate the expression in parentheses first (``(cos(C) / D)``),
followed by the multiply, then the assignment. Written using standard C++ operator overloading, we would have a 
cosine, division, multiplication, and assignment overload. Each operator performs their respective task, then returns
the value computed. That returned value is stored somewhere (either out to memory or possible in a register), then
the next operator uses that output as input into its own computation. Finally, the assignment writes the value, 
usually out to memory.

While there's nothing wrong with the approach above, it can lead to significant performance penalties, especially
on hardware where the penalty for going to main memory is orders of magnitude higher than other types of memory. In
the worst case, the expression above would have 4 loads and stores (cosine, division, multiplication, assignment).

To avoid this overhead, MatX uses a technique called *lazy evaluation* to reduce the total number of loads and stores.
It does this by overloading each operator so that instead of performing the operation, such as multiplication, instead
it returns an object that *represents* multiplication when it's needed. The entire expression is generates a single
type in C++ representing the equation above, and when we ask for element ``(0,0)`` of ``A`` above, the value is computed
on-the-fly without storing any values. This also implies that you can store an entire expression into a variable and
nothing will be exectuted: 

.. code-block:: cpp

    auto op = (B * (cos(C) / D));

In the example above ``op`` can be further combined with other expressions, which can increase code readability without
loss of performance.

MatX's operator fusion can extend beyond the simple expressions above. Since transforms are also usable inside operator
expressions, this opens the possibility to selectively fuse more complex expressions:

.. code-block:: cpp

    (A = B * fft(C)).run();

The type system can see that we have a multiply where the right-hand side is an FFT transform and the left side is another
operator. This allows MatX to potentially fuse the output of the FFT with a multiply of B at compile-time. In general, the 
more information it can deduce during compilation and runtime, the better the performance will be.

CUDA JIT Kernel Fusion
======================

.. note::

    CUDA JIT kernel fusion is considered an experimental feature. There may be bugs that don't occur with JIT disabled, and new features are being added over time.

MatX supports CUDA JIT kernel fusion that compiles the entire expression into a single kernel. Currently this is enabled
for all standard MatX element-wise operators, small random generators via cuRANDDx, FFT operations via cuFFTDx, GEMM
operations via cuBLASDx, and selected solver operations via cuSolverDx. cuFFTDx supports 1D FFT fusion and single-block
complex-to-complex 2D ``fft2``/``ifft2`` fusion for supported power-of-two square transforms. To enable fusion with
MathDx, the following option must be enabled: ``-DMATX_EN_MATHDX=ON``. MathDx support also enables the NVRTC-based JIT
support used by ``CUDAJITExecutor``.
Once enabled, the ``CUDAJITExecutor`` can be used to perform JIT compilation in supported situations. If the expression
cannot be JIT compiled, the ``CUDAJITExecutor`` may throw an error.

While JIT compilation can provide a large performance boost, there are two overheads that occur when using JIT compilation:

* The first pass to JIT the code takes time. The first time a ``run()`` statement is executed on a new operator, MatX identifies this and performs JIT compilation. Depending on the complexity of the operator, this could be anywhere from milliseconds to seconds to complete. Once finished, MatX will cache the compiled kernel so that subsequent runs of the same operator will not require JIT compilation.
* A lookup is done to find kernels that have already been compiled. This is a small overhead and may not be noticeable.

For MathDx-backed operations, MatX uses the ``libmathdx`` runtime interface to query launch resources such as shared
memory usage, block dimensions, workspace size, generated symbol names, and LTOIR. This is required because MatX
operator sizes may be known only at runtime. MatX caches the generated code and launch metadata for a matching operator
signature so repeated runs can avoid repeating the runtime descriptor queries.

As mentioned above, there is no difference in syntax between MatX statements that perform JIT compilation and those that do not. The executor 
is the only change, just as it would be with a host executor. For example, in the following code:

.. code-block:: cpp

    (A = B * fft(C)).run(CUDAExecutor{});
    (A = B * fft(C)).run(CUDAJITExecutor{});

The first statement will execute the FFT as a separate kernel into a temporary buffer, then the multiply will be executed. This results 
in a minimum of 2 kernels (one for MatX and at least one for cuFFT). The second statement will execute the FFT and multiply in a single kernel if 
possible.

Some operators cannot be JIT compiled. For example, if the FFT above is a size not compatible with the cuFFTDx library or if MathDx is disabled 
the expression will not be JIT compiled. To determine if an operator can be JIT compiled, use the ``matx::jit_supported(op)`` function: 

.. code-block:: cpp

    auto my_op = (fft(b) + c);
    if (matx::jit_supported(my_op)) {
      printf("FFT is supported by JIT\n");
    } else {
      printf("FFT is not supported by JIT\n");
    }

Even if the MathDx library supports a particular operation, other operators in the expression may prevent JIT compilation. For 
example: 

.. code-block:: cpp

    auto my_op = (fftshift1D(fft(b)));

In this case the MathDx library requires at least 2 elements per thread for the FFT, but the ``fftshift1D`` operator requires 
only 1 element per thread. Therefore, the entire expression cannot be JIT-compiled with ``CUDAJITExecutor``. Use a normal
``CUDAExecutor`` to run the same expression through the non-JIT path. Some of these restrictions may be relaxed in newer
versions of MatX or the MathDx library.

MathDx Compatibility
====================

.. list-table:: MathDx library compatibility for CUDA JIT fusion
   :header-rows: 1
   :widths: 20 14 66

   * - Library
     - Supported
     - Notes
   * - cuBLASDx
     - Yes
     - Enabled via ``-DMATX_EN_MATHDX=ON`` for compatible ``matmul``/GEMM fusion paths. Fusion with other MathDx
       operators requires a compatible runtime block dimension. For example, a ``matmul`` expression can fuse with a
       cuSolverDx ``inv`` when their block-dimension ranges intersect.
   * - cuFFTDx
     - Yes
     - Enabled via ``-DMATX_EN_MATHDX=ON`` for compatible 1D FFT fusion paths and supported single-block 2D C2C FFT
       fusion paths. cuFFTDx has stricter launch-shape requirements than the generic element-wise JIT path, and does not
       generally compose with cuBLASDx/cuSolverDx operations that require a different block/grid model.
   * - cuSolverDx
     - Partial
     - Enabled via ``-DMATX_EN_MATHDX=ON`` for ``chol``, ``inv``, and selected projection outputs from
       ``lu``, ``qr``, ``qr_solver``, ``qr_econ``, and ``eig``. The current cuSolverDx JIT path supports rank
       2 through 4 matrices with ``float``, ``double``, ``complex<float>``, and ``complex<double>`` values.
       ``chol``, ``inv``, ``qr``, and ``eig`` require square matrices in the JIT path. ``qr_econ(A).Q`` JIT fusion
       is currently limited to non-wide matrices where ``m >= n``; use ``qr_econ(A).R`` or ``qr_solver`` projections
       for other rectangular QR forms. Multi-output ``mtie`` assignments still use the normal
       non-JIT solver path, while projection expressions such as ``auto op = lu(A); (Y = op.LU).run(CUDAJITExecutor{})``
       may be fused. ``svd`` is not JIT-fused because the runtime ``libcusolverdx`` code-generation API does not
       expose an SVD routine.
   * - cuRANDDx
     - Partial
     - Enabled via ``-DMATX_EN_MATHDX=ON`` for floating-point and complex ``random()`` expressions whose random
       output has at most 1024 elements. The cuRANDDx path generates Philox values in the fused kernel and avoids the
       temporary value buffer used by generic CUDA expressions. ``randomi()`` and larger random tensors are not
       JIT-fused.

Current Limitations
===================

``CUDAJITExecutor`` only runs expressions that MatX can lower completely into one JIT kernel. All operators in the
expression must support CUDA JIT, and MathDx-backed operators must have compatible runtime launch requirements such as
block dimensions and shared-memory usage. Unsupported shapes, types, ranks, or incompatible MathDx launch requirements
are reported before launch instead of silently falling back inside ``CUDAJITExecutor``.

For multi-output solvers, CUDA JIT fusion is exposed through lazy projection members rather than through the
``mtie`` assignment form. Only the projection outputs referenced by the expression are generated by the JIT path.
For example, using ``qr_econ(A).R`` does not run the ``ungqr`` step needed only to form ``Q``, and using eigenvalues
without eigenvectors selects the no-vectors cuSolverDx job.

The normal non-JIT CUDA, host, and library-backed execution paths remain available for operations that are not currently
supported by CUDA JIT fusion.
