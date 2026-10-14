.. _debugging:

Debugging
#########

MatX employs several tools for debugging and improving the correctness of the code. 

Logging
--------

MatX provides a logging system that can be used to log messages to the console. This is useful for debugging your code and can be used to trace the execution of your code.

See :ref:`logging_basics` for more information on the logging system.

Compile Time
------------

At compile time MatX uses `static_assert` calls where possible to provide helpful error messages. Static assertions have a limitation that 
they cannot display a formatted string, so the value of the invalid parameters are not displayed. Common compile time errors include:

- Invalid rank
- Invalid type
- Invalid tensor shapes (for static tensor sizes)

Runtime
-------

At runtime MatX uses C++ exceptions to throw errors. These errors are typically based on expected vs actual outcomes. Several macros are used 
to raise these errors:

- MATX_ASSERT (boolean assertion)
- MATX_ASSERT_STR (boolean assertion with a formatted string)
- MATX_ASSERT_STR_EXP (boolean assertion with a formatted string and an expected value)

These macros are also listed in order of usefulness with the `MATX_ASSERT_STR_EXP` macro providing the most information to the user. Common 
runtime errors include:

- Invalid sizes
- Invalid indexing
- Errors returned from CUDA APIs 


Null Pointer Checking
---------------------

Tensors in MatX may be left unitialized on declaration. This is common when a tensor is used as a class member and is not initialized in the constructor. For example: 

.. code-block:: cpp

    class MyClass {
        public:
            MyClass() {
            }
        private:
            tensor_t<float> t; // Uninitialized
    };

Typically `make_tensor` is used at a later time to declare the shape allocate the memory backing the tensor. Detecting an unitialized tensor on the device 
has a non-zero performance penalty and is disabled by default. To detect an unitialized tensor on the device, build your application in debug mode with the 
`NDEBUG` flag undefined. When the `NDEBUG` flag is undefined, MatX will check for unitialized tensors on the device and assert if one is found.

Unsafe Aliased Memory Checking
------------------------------

MatX provides an imperfect unsafe aliased memory checking system that can be used to detect when an input tensor may overlap with output tensor memory, 
causing a data race. The word *unsafe* is used here because there are cases where aliasing is safe, such as a direct element-wise operation.
To have a false positive rate of 0 we would need to check every possible input and output location to see if any of them overlap. 
This would be impractical for most applications. Instead, we use several checks that can catch the most common cases of memory aliasing. Since aliasing can be 
an expensive check and it's not perfect, alias checking must be explicitly enabled with the CMake option `MATX_EN_UNSAFE_ALIAS_DETECTION` or the compiler 
define with the same name.

The types of aliasing that can be detected are:

- Safe element-wise aliasing: (a = a + a) // No aliasing since it's a direct element-wise operation
- Safe element-wise aliasing: (slice(a, {0}, {5}) = slice(a, {0}, {5}) - slice(a, {0}, {5})) // No aliasing since it's a direct element-wise operation
- Unsafe element-wise aliasing: (slice(a, {0}, {5}) = slice(a, {3}, {8}) - slice(a, {0}, {5})) // Unsafe since inputs and outputs overlap to different locations
- Unsafe matrix multiplication: (c = matmul(c, d)) // Unsafe since matmul doesn't allow aliasing on input and output memory
- Safe FFT: (c = fft(c)) // No aliasing since FFT allows aliasing
- Safe strided element-wise aliasing: (slice(a, {0}, {6}, {2}) = slice(a, {0}, {6}, {2}) + slice(a, {0}, {6}, {2})) // Matching tensor layouts with distinct element addresses are safe

Concrete tensor views with identical data pointers, element sizes, shapes, and strides are accepted for in-place element-wise operations when their strides prove
that elements do not overlap. This includes slices that select a fixed index and drop the last dimension of a contiguous tensor, and IFFT normalization on those
views. Operators that reorder input relative to output, or tensors with different layouts, are still checked for unsafe overlap.
A non-identity ``permute`` of an expression that reads the destination (for example, ``a = permute(a + a, {1, 0})``) is unsafe;
an identity permutation preserves the element-wise alias exemption. Adjacent permutations are composed, so inverse permutations that restore the original
mapping also preserve this exemption. Permutations separated by another operator are still checked conservatively.
Layouts whose elements cannot be proven distinct (including zero or negative strides on non-singleton dimensions) remain conservatively rejected.
Writable operator destinations without concrete tensor layout metadata retain the contiguity requirement for this exemption.
