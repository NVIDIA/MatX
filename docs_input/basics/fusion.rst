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
for all standard MatX element-wise operators and FFT operations via MathDx. To enable fusion with MathDx, 
the following options must be enabled: ``-DMATX_EN_MATHDX=ON``. Once enabled, the ``CUDAJITExecutor`` can be used perform JIT compilation
in supported situations. If the expression cannot be JIT compiled, the JITExecutor will fall back to the normal non-JIT path.

While JIT compilation can provide a large performance boost, there are two overheads that occur when using JIT compilation:
- The first pass to JIT the code takes time. The first time a ``run()`` statement is executed on a new operator, MatX identifies this and performs JIT compilation. Depending on the complexity of the operator, this could be anywhere from milliseconds to seconds to complete. Once finished, MatX will cache the compiled kernel so that subsequent runs of the same operator will not require JIT compilation.
- A lookup is done to find kernels that have already been compiled. This is a small overhead and may not be noticeable.

As mentioned above, there is no difference in syntax between MatX statements that perform JIT compilation and those that do not. The executor 
is the only change, just as it would be with a host executor. For example, in the following code:

.. code-block:: cpp

    (A = B * fft(C)).run(CUDAExecutor{});
    (A = B * fft(C)).run(CUDAJITExecutor{});

When MathDx is disabled, the the first statement will execute the FFT into a temporary buffer, then the multiply will be executed. This results 
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
only 1 element per thread. Therefore, the entire expression cannot be JIT-compiled and will fall back to the non-JIT path. Some of 
these restrictions may be relaxed in newer versions of MatX or the MathDx library.


