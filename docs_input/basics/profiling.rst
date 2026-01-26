.. _profiling:

Profiling
#########

Profiling is a way to measure the performance of a program and to identify bottlenecks in your MatX application. Since 
the method for profiling depends on the executor, each executor implements its own profiling mechanism. For example, 
the CUDA executor can use events encapsulating the kernels it's profiling. The profiling is done through the executor 
object rather than the `run` statement so that multiple `run`\s can be profiled together. Profiling is disabled by default
in the executor due to the overhead of the profiling mechanism. To enable profiling, pass `true` as the second parameter 
to the executor constructor:

.. code-block:: cpp

    cudaExecutor exec{stream, true}; // Enable profiling

Profiling is done by calling the `start_timer()` method of the executor:

.. code-block:: cpp

    exec.start_timer();

To stop the profiler, `stop_timer()` is called:

.. code-block:: cpp

    exec.stop_timer();

Depending on the executor, `stop_timer()` may need to block for the operation to conplete on an asynchronous executor.

Once `stop_timer()` returns, the execution time between the timers can be retrieved by calling `get_time_ms()`:

.. code-block:: cpp

    auto time = exec.get_time_ms();

In the above example `time` contains the runtime of everything executed between the `start_timer()` and `stop_timer()` calls. For
a CUDA executor this is the time between the beginning of the first kernel and the end of the last. For a CPU executor this is the CPU 
time between the two calls.

.. note::
   Profiling does not work a multi-threaded host executor currently

For a full example of profiling, see the `spectrogram` example.