.. _logging_basics:

Logging
#######

MatX provides a flexible, zero-overhead logging system for debugging and diagnostics. The logging system uses C++20's ``std::format`` and ``std::source_location`` to provide rich, formatted log messages with automatic file and line information.

Features
========

- **Zero overhead when disabled**: Logging is disabled by default with minimal runtime cost
- **Multiple severity levels**: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
- **ISO 8601 timestamps**: All log messages include precise timestamps
- **Automatic source location**: File name, line number, and optional function name are automatically captured
- **Flexible output**: Logs can be directed to stdout, stderr, or a file
- **Thread-safe**: Logging operations are protected by mutex
- **Type-safe formatting**: Uses C++20 ``std::format`` for compile-time format string checking

Basic Usage
===========

Include MatX and use the logging macros:

.. code-block:: cpp

   #include <matx.h>
   
   using namespace matx;
   
   void my_function() {
     int value = 42;
     MATX_LOG_INFO("Processing value: {}", value);
     
     double result = 3.14159;
     MATX_LOG_DEBUG("Result computed: {:.2f}", result);
     
     MATX_LOG_WARN("Warning: operation may be slow");
   }

Logging Macros
==============

MatX provides convenience macros for each severity level:

.. code-block:: cpp

   MATX_LOG_TRACE("Detailed trace information: {}", data);
   MATX_LOG_DEBUG("Debug information: x={}, y={}", x, y);
   MATX_LOG_INFO("Informational message");
   MATX_LOG_WARN("Warning message: threshold={}", threshold);
   MATX_LOG_ERROR("Error occurred: {}", error_code);
   MATX_LOG_FATAL("Fatal error: cannot continue");

You can also use the base macro with explicit severity:

.. code-block:: cpp

   MATX_LOG(matx::detail::LogLevel::DEBUG, "Custom message: {}", value);

Configuration
=============

Logging is controlled via environment variables, which must be set before the program starts.

MATX_LOG_LEVEL
--------------

Controls which messages are displayed based on severity:

.. code-block:: bash

   # Show all messages (most verbose)
   export MATX_LOG_LEVEL=TRACE
   
   # Show DEBUG and above
   export MATX_LOG_LEVEL=DEBUG
   
   # Show INFO and above (recommended for production)
   export MATX_LOG_LEVEL=INFO
   
   # Show only warnings and errors
   export MATX_LOG_LEVEL=WARN
   
   # Show only errors
   export MATX_LOG_LEVEL=ERROR
   
   # Disable all logging (default)
   export MATX_LOG_LEVEL=OFF

Numeric levels are also supported: 0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=FATAL, 6=OFF.

Severity Levels
~~~~~~~~~~~~~~~

+----------+-------+-------------------------------------------------------+
| Level    | Value | Use Case                                              |
+==========+=======+=======================================================+
| TRACE    | 0     | Very detailed information for diagnosing issues       |
+----------+-------+-------------------------------------------------------+
| DEBUG    | 1     | Detailed information for debugging                    |
+----------+-------+-------------------------------------------------------+
| INFO     | 2     | Informational messages about normal operation         |
+----------+-------+-------------------------------------------------------+
| WARN     | 3     | Warning messages for potentially problematic issues   |
+----------+-------+-------------------------------------------------------+
| ERROR    | 4     | Error messages for recoverable errors                 |
+----------+-------+-------------------------------------------------------+
| FATAL    | 5     | Critical errors that may cause termination            |
+----------+-------+-------------------------------------------------------+
| OFF      | 6     | Disable all logging                                   |
+----------+-------+-------------------------------------------------------+

MATX_LOG_DEST
-------------

Controls where log messages are written:

.. code-block:: bash

   # Write to stdout (default)
   export MATX_LOG_DEST=stdout
   
   # Write to stderr
   export MATX_LOG_DEST=stderr
   
   # Write to a file (will be created or appended)
   export MATX_LOG_DEST=/path/to/logfile.log

If writing to a file fails, logging automatically falls back to stdout with a warning.

MATX_LOG_FUNC
-------------

Controls whether function names are included in log output. Some function names are very large due to heavy templating:

.. code-block:: bash

   # Show function names in log output
   export MATX_LOG_FUNC=1
   # or
   export MATX_LOG_FUNC=true
   # or
   export MATX_LOG_FUNC=ON
   
   # Hide function names (default)
   export MATX_LOG_FUNC=0
   # or
   unset MATX_LOG_FUNC

This is useful for reducing log verbosity while still maintaining file and line information.

Log Format
==========

Log messages follow a structured format that varies based on configuration:

Without Function Names (Default)
---------------------------------

.. code-block:: text

   YYYY-MM-DDTHH:MM:SS.mmm [LEVEL] filename.ext:line - message

Example:

.. code-block:: text

   2025-10-21T14:32:45.123 [DEBUG] fft.h:277 - DYN_SHM_SIZE: 8192
   2025-10-21T14:32:45.124 [INFO] my_app.cu:42 - Processing complete

With Function Names (MATX_LOG_FUNC=1)
--------------------------------------

.. code-block:: text

   YYYY-MM-DDTHH:MM:SS.mmm [LEVEL] filename.ext:line (function_name) - message

Example:

.. code-block:: text

   2025-10-21T14:32:45.123 [DEBUG] fft.h:277 (get_capability) - DYN_SHM_SIZE: 8192
   2025-10-21T14:32:45.124 [INFO] my_app.cu:42 (main) - Processing complete

Format String Syntax
====================

MatX logging uses C++20 ``std::format`` syntax, which is similar to Python's format strings and Rust's formatting:

Basic Formatting
----------------

.. code-block:: cpp

   // Basic substitution
   MATX_LOG_INFO("Value: {}", 42);
   // Output: Value: 42
   
   // Multiple arguments
   int x = 10, y = 20;
   MATX_LOG_INFO("Coordinates: x={}, y={}", x, y);
   // Output: Coordinates: x=10, y=20
   
   // Positional arguments
   MATX_LOG_DEBUG("{0} + {1} = {2}", 5, 3, 8);
   // Output: 5 + 3 = 8

Number Formatting
-----------------

.. code-block:: cpp

   // Floating-point precision
   MATX_LOG_DEBUG("Pi: {:.2f}", 3.14159);
   // Output: Pi: 3.14
   
   MATX_LOG_DEBUG("Scientific: {:.2e}", 12345.6);
   // Output: Scientific: 1.23e+04
   
   // Hexadecimal
   MATX_LOG_DEBUG("Address: 0x{:08x}", 0xDEADBEEF);
   // Output: Address: 0xdeadbeef
   
   // Binary
   MATX_LOG_DEBUG("Flags: {:08b}", 42);
   // Output: Flags: 00101010
   
   // With thousands separator
   MATX_LOG_INFO("Large number: {:L}", 1234567);
   // Output: Large number: 1,234,567

Alignment and Padding
---------------------

.. code-block:: cpp

   // Left-aligned (default for strings)
   MATX_LOG_INFO("Left: {:<10}", "text");
   // Output: Left: text      
   
   // Right-aligned (default for numbers)
   MATX_LOG_INFO("Right: {:>10}", "text");
   // Output: Right:       text
   
   // Center-aligned
   MATX_LOG_INFO("Center: {:^10}", "text");
   // Output: Center:   text   
   
   // Zero-padding for numbers
   MATX_LOG_DEBUG("Index: {:04d}", 42);
   // Output: Index: 0042

Arrays and Containers
---------------------

For arrays, you'll need to format elements individually:

.. code-block:: cpp

   // Logging array elements
   cuda::std::array<int, 3> dims = {128, 256, 512};
   MATX_LOG_DEBUG("Dimensions: [{}, {}, {}]", dims[0], dims[1], dims[2]);
   // Output: Dimensions: [128, 256, 512]

Practical Examples
==================

FFT Operations
--------------

Enable debug logging for FFT operations:

.. code-block:: bash

   export MATX_LOG_LEVEL=DEBUG
   ./my_fft_application

You'll see detailed capability information:

.. code-block:: text

   2025-10-21T14:32:45.123 [DEBUG] fft.h:277 - DYN_SHM_SIZE: 8192
   2025-10-21T14:32:45.124 [DEBUG] fft.h:293 - SUPPORTS_JIT: true
   2025-10-21T14:32:45.125 [DEBUG] fft.h:318 - ELEMENTS_PER_THREAD (JIT supported): [8,16]
   2025-10-21T14:32:45.126 [DEBUG] fft.h:339 - GROUPS_PER_BLOCK: [4,4]

CUDA Executor
-------------

Debug kernel launch parameters:

.. code-block:: bash

   export MATX_LOG_LEVEL=DEBUG
   export MATX_LOG_FUNC=1
   ./my_cuda_app

Output includes function context:

.. code-block:: text

   2025-10-21T14:32:45.200 [DEBUG] get_grid_dims.h:283 (GetGridDims) - Blocks 32x1x1 Threads 256x1x1 groups_per_block=1
   2025-10-21T14:32:45.201 [DEBUG] nvrtc_helper.h:551 (LaunchKernel) - Launching kernel with grid=(32, 1, 1), block=(256, 1, 1), dynamic_shmem_size=0 bytes

Cache Operations
----------------

Debug cache hits and misses:

.. code-block:: bash

   export MATX_LOG_LEVEL=DEBUG
   ./my_app

.. code-block:: text

   2025-10-21T14:32:45.100 [DEBUG] cache.h:329 - Cache HIT (memory) for: kernel_xyz.cubin
   2025-10-21T14:32:45.101 [DEBUG] cache.h:343 - Cache MISS (disk) for: kernel_abc.cubin
   2025-10-21T14:32:45.102 [DEBUG] cache.h:357 - Cache HIT (disk) for: kernel_def.cubin, size: 12345 bytes

Performance Considerations
==========================

Overhead When Disabled
----------------------

When logging is disabled (default state), the overhead is minimal:

.. code-block:: cpp

   // This has negligible overhead when logging is OFF
   for (int i = 0; i < 1000000; i++) {
     MATX_LOG_TRACE("Iteration {}", i);  // Only a single boolean check
   }

The compiler optimizes away the format string and argument evaluation when the log level is disabled.

Overhead When Enabled
---------------------

When logging is enabled:

- **String formatting**: Performed using ``std::format``
- **I/O operations**: Writing to file/stream has normal I/O costs
- **Thread synchronization**: Mutex lock/unlock for thread safety

Recommendations:

1. Use appropriate log levels for your use case
2. Avoid TRACE logging in production unless debugging
3. Use DEBUG for development and troubleshooting
4. Keep INFO logging for important operational events
5. Consider log file rotation for long-running applications

Best Practices
==============

Choose Appropriate Levels
--------------------------

.. code-block:: cpp

   // TRACE: Very detailed, typically temporary debug code
   MATX_LOG_TRACE("Entering function with params: a={}, b={}", a, b);
   
   // DEBUG: Development debugging information
   MATX_LOG_DEBUG("Cache size: {} entries, {} MB", count, size_mb);
   
   // INFO: Important operational events
   MATX_LOG_INFO("System initialized with {} GPUs", gpu_count);
   
   // WARN: Unexpected but handled situations
   MATX_LOG_WARN("Cache full, evicting {} old entries", evict_count);
   
   // ERROR: Errors that don't stop execution
   MATX_LOG_ERROR("Failed to allocate {} bytes, retrying", requested);
   
   // FATAL: Critical errors
   MATX_LOG_FATAL("CUDA device not found, cannot continue");

Provide Context
---------------

Good logging includes relevant context:

.. code-block:: cpp

   // Bad: Not enough context
   MATX_LOG_ERROR("Operation failed");
   
   // Good: Includes context
   MATX_LOG_ERROR("FFT operation failed: size={}, type={}, error_code={}", 
                  fft_size, type_str, error);
   
   // Good: Shows state
   MATX_LOG_DEBUG("Memory usage: allocated={} MB, free={} MB, total={} MB",
                  allocated, free, total);

Be Concise
----------

Keep messages clear and to the point:

.. code-block:: cpp

   // Too verbose
   MATX_LOG_DEBUG("Now about to execute the FFT transform operation on the input tensor with {} elements", n);
   
   // Better
   MATX_LOG_DEBUG("Executing FFT: {} elements", n);

Avoid Sensitive Data
--------------------

Don't log sensitive information:

.. code-block:: cpp

   // Bad: Logging sensitive data
   MATX_LOG_INFO("User password: {}", password);  // Never do this!
   
   // Good: Log safe information
   MATX_LOG_INFO("User authenticated: id={}", user_id);

Conditional Verbose Logging
----------------------------

Use TRACE for very detailed logging that you enable only when needed:

.. code-block:: cpp

   void process_large_dataset() {
     for (size_t i = 0; i < data.size(); i++) {
       // This won't impact performance unless TRACE is enabled
       MATX_LOG_TRACE("Processing element {}: value={}", i, data[i]);
       
       // Do actual work
       process(data[i]);
     }
     
     // Always log completion
     MATX_LOG_INFO("Processed {} elements", data.size());
   }

Limitations
===========

Current Constraints
-------------------

- **Host-only**: Logging is only available in host code, not inside CUDA kernels
- **C++20 required**: Requires a C++20-compatible compiler with ``std::format`` support
- **Synchronous**: Log writes are synchronous and serialized across threads
- **Static configuration**: Environment variables are read once at program startup

Device Code
-----------

For logging from device code, use CUDA's ``printf``:

.. code-block:: cpp

   __global__ void my_kernel() {
     if (threadIdx.x == 0 && blockIdx.x == 0) {
       printf("Kernel executing: threads=%d, blocks=%d\n", 
              blockDim.x, gridDim.x);
     }
   }

Runtime Configuration
---------------------

The logger reads environment variables once during initialization. To change logging configuration:

.. code-block:: cpp

   // Set environment variable before MatX initialization
   setenv("MATX_LOG_LEVEL", "DEBUG", 1);
   
   // Or use the reinitialize method (mainly for testing)
   matx::detail::Logger::instance().reinitialize();

Complete Example
================

Here's a complete example demonstrating logging in a MatX application:

.. code-block:: cpp

   #include <matx.h>
   
   using namespace matx;
   
   int main() {
     MATX_LOG_INFO("MatX application starting");
     
     // Create tensors
     auto t1 = make_tensor<float>({1024, 1024});
     auto t2 = make_tensor<float>({1024, 1024});
     MATX_LOG_DEBUG("Tensors created: shape=[{}, {}]", t1.Size(0), t1.Size(1));
     
     // Initialize
     (t1 = ones(t1.Shape())).run();
     (t2 = ones(t2.Shape())).run();
     MATX_LOG_DEBUG("Tensors initialized");
     
     // Perform FFT
     MATX_LOG_INFO("Starting FFT operation");
     auto t1_fft = fft(t1);
     (t2 = t1_fft).run();
     MATX_LOG_INFO("FFT operation complete");
     
     // Check results
     float result;
     (result = sum(abs(t2))).run();
     MATX_LOG_INFO("FFT result magnitude: {:.2e}", result);
     
     if (result < 1e-6) {
       MATX_LOG_WARN("Result magnitude is very small: {}", result);
     }
     
     MATX_LOG_INFO("Application complete");
     return 0;
   }

Running with different log levels:

.. code-block:: bash

   # Minimal output
   export MATX_LOG_LEVEL=INFO
   ./my_app
   # Output:
   # 2025-10-21T14:32:45.100 [INFO] my_app.cu:7 - MatX application starting
   # 2025-10-21T14:32:45.120 [INFO] my_app.cu:17 - Starting FFT operation
   # 2025-10-21T14:32:45.145 [INFO] my_app.cu:19 - FFT operation complete
   # 2025-10-21T14:32:45.150 [INFO] my_app.cu:23 - FFT result magnitude: 1.05e+06
   # 2025-10-21T14:32:45.151 [INFO] my_app.cu:29 - Application complete
   
   # Detailed output with function names
   export MATX_LOG_LEVEL=DEBUG
   export MATX_LOG_FUNC=1
   ./my_app
   # Includes all DEBUG messages plus function names
