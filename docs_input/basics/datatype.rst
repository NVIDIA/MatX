.. _datatypes_api:

Data Types
##########

MatX attempts to use the default C++ type system rules wherever possible; the inner types
of operators should behave identically to normal C++ rules. For example, assigning a tensor
of type double to a tensor of type float will have the same implications as assigning the simple
data types to each other. For example:

.. code-block:: cpp

    auto t1 = make_tensor<float>();
    auto t2 = make_tensor<double>();
    (t1 = t2).run();

Follows the same rules as:

.. code-block:: cpp

  float t1;
  double t2;
  t1 = t2;

MatX tests the following types in unit tests:

#. Integer
   * `int8_t`
   * `uint8_t`
   * `int16_t`
   * `uint16_t`
   * `int32_t`
   * `uint32_t`
   * `int64_t`
   * `uint64_t`
#. Floating Point 
   * `matxFp16` (`__half`)
   * `matxBf16` (`__nv_bfloat16`)
   * `float`
   * `double`
#. Complex
   * `matxfp16Complex`
   * `matxBf16Complex`
   * `cuda::std::complex<float>`
   * `cuda::std::complex<double>`

Since MatX attempts to have parity for all functionality on both host and device, all types above
are useable in both scenarios. While most types above are common C++ types, there are notable exceptions:

- Native half precision types (`__half`/`__nv_bfloat16`) are swapped for `matxFp16` and `matxBf16`. This is done because the native types do not provide the full set of operator overloads on both the host and device. The same concept applies to the complex versions `matxfp16Complex` and `matxBf16Complex`.
- Complex `float` and `double` use the `cuda::std` versions rather than `std::` since `std::complex` does not work in device code. libcudacxx is included with the CUDA toolkit.


User-defined types
------------------

While the above types are tested, any type supporting the standard C++ operator overloading semantics should work, depending
on the context used. For example, to sort a tensor the comparison and equality operators must be defined.