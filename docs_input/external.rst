.. _devexternal:

Interfacing With External Code and Libraries
############################################

Existing host and CUDA code can interoperate seamlessly with MatX both by using MatX primitives in existing code, 
and transferring MatX data into other libraries. Integrating MatX into existing code is a common use case that 
allows developers to incrementally port code into MatX without having to rewrite everything at once.

This guide is not intended for developers who wish to extend MatX. See :ref:`devguide` for the MatX developer guide.


Passing Existing Pointers to MatX
---------------------------------

To use MatX in existing code, the pointers (whether host or device) are passed into the `make_tensor` call as the 
first parameter:

.. code-block:: cpp

  // Existing code
  float *my_data_ptr;
  cudaMalloc((void*)&my_data_ptr, 100 * sizeof(float)); // Treated as a 10x10 float matrix in the code
  foo(my_data_ptr); // Call existing function that uses my_data_ptr

  // Work with my_data_ptr on the device

  // End of existing code. Convert to MatX tensor
  auto matx_tensor = matx::make_tensor<float>(my_data_ptr, {10, 10});

  // MatX functions

In the code above the developer has an existing device pointer that they used in their CUDA code. It's common in existing
CUDA code to see linear allocations like the one above, but the developer treats it as a higher-dimension tensor in the code. 
For this example `my_data_ptr` was allocated with linear memory holding 100 floats, but the user treats it later as a 10x10 matrix.

Since MatX needs to know the shape of the tensor when it's created, we explictly pass the `{10, 10}` shape into the 
`make_tensor` call. 

By default MatX will not take ownership of the pointer; the user is responsible for freeing the memory when they are done with it.
This is true of all `make_tensor` calls that take an existing pointer as an argument since the user typically has their own
memory management outside of MatX. The last parameter of each `make_tensor` call is a boolean named `owning` that tells MatX to 
take ownership, and defaults to *false*. By setting `owning` to *true*, MatX will free the memory when the tensor goes out of scope.
By default it uses its own allocator, but users can pass in their own PMR-compatible allocator if they wish. For more information 
see :ref:`creating`.

MatX does not know the "space" or "kind" of the memory when a pointer is passed in. This is by design since looking up what kind
of pointer is passed (device, host, managed, etc) is expensive and error-prone. In addition, what can be done with a pointer of a 
certain type is dependent on the system. For example, a Grace-Hopper system can share pointers allocated with `malloc` between the 
the GPU and CPU, but x86 cannot. It is up to the user to make sure that all memory types used in an expression are compatible.

.. code-block:: cpp

  auto host_mem      = reinterpret_cast<float*>(malloc(10 * sizeof(float)));
  auto host_tensor   = make_tensor<float>(host_mem, {10});
  auto device_tensor = make_tensor<float>({10}, MATX_DEVICE_MEMORY);
  (device_tensor = host_tensor).run();

The code above attempts to copy memory from a malloc'd pointer on the host to device memory. This may or may not work depending 
on the system. If unsure, it's usually safe to make a tensor with managed memory, copy data into it, then use it on both host or
device. 
   

Passing MatX Operators to External Code/Libraries
-------------------------------------------------

MatX operators can be passed to external code or libraries in two ways: by object or by pointer. Passing MatX operators by object is 
the preferred way when possible. Doing so maintains all of the internal information and state that is contained in the operator and 
reduces the chances of errors. 

Sometimes code cannot be modified to allow for passing by object. This is common when working with libraries that have API that 
cannot be changed easily, or if the overhead of passing by value is too large. MatX also allows developers to extract the pointer 
from a MatX operator and pass it to external code by using the `Data()` method of a tensor. Note that unlike the "pass-by-object" method, 
this method only works for tensors since general operators do not have a data pointer.

Care must be taken when passing either operators or pointers to existing code to avoid bugs:

* The data is only valid for the lifetime of the tensor. If the tensor goes out of scope, the data backing the tensor is invalid. For 
  example, if a CUDA kernel is called asynchronously with a tensor as a parameter, then the tensor goes out of scope while the kernel 
  runs, the results are undefined.
* The *kind* of the pointer must be known to the external code. For example, if the tensor was created in device memory, the external 
  code must access it only where device memory is accessible.

If the external code supports the *dlpack* standard, the tensor's `ToDLPack()` method can be used instead to get a `DLManagedTensor` object.
This method is much safer since all shape and ownership can be transferred.


Passing By Object
=================

Passing by object makes all of the object's metadata available inside of an external function. Since operator types can be very complex, it's 
always recommended to pass the operator as a template parameter rather than specifying the type of the operator. Passing by value does *not* 
copy the data (if any) backing the operator; only the metadata (shape, strides, etc) is copied.

.. code-block:: cpp

  template <typename Op>
  void foo(Op &op)
  {
    // Do something with the operator
    auto val = op(10, 1);
  }

  template <typename Op>
  __global__ void foo_kernel(Op op)
  {
    // Do something with the operator
    auto val = op(10, 1);
  }  

  // Create a MatX operator
  auto t1 = matx::make_tensor<float>({10, 10});
  auto t2 = matx::make_tensor<float>({10, 10});
  auto o1 = (t1 + t2) * 2.0f;

  foo(o1);

  typename matx::detail::base_type_t<decltype(o1)> o1_base;
  foo_kernel<<<1,1>>>(o1_base);

The first function `foo` is a host function that takes a MatX operator as a template parameter by reference, while `foo_kernel` is 
a CUDA kernel that takes the operator by value. When passing an operator to a CUDA kernel it should always be passed by value 
unless the operator's memory is accessible on the device. The template parameter allows the user to pass any operator to the 
function that adheres to the operator interface. This is a powerful concept that reduces the need for code changes if the type 
of the operator changes. For example, changing the `o1` statment to `t1 - t2` would change the type of the operator, but using 
templates allows the same code to exist in `foo` without changing the type. 

For more information about the *operator interface*. see :ref:`concepts`.

Inside of both `foo` and `foo_kernel` all functions in the *operator interface* are available. `op(10, 1)` will return the value 
at the 11th row and 2nd column of the operator (0-based). Using `operator()` inside of the operator will handle all the indexing 
logic to handle the shape and stride of the operator.

The last part to mention in the code is the declaration of `o1_base`. Some operator types in MatX, such as a `tensor_t`, cannot 
be passed directly to a CUDA kernel due to internal types that cannot be used on the device. The `base_type_t` type trait will 
convert the operator to a type that can be used on the device if needed, or it will return the same type if it's already usable 
on the device. 

Passing By Pointer
==================

In the code above `t1` and `t2` could have their pointers extracted, but `o1` could not. For that reason, passing raw pointers 
can only be used on tensors and not other operators. 

.. code-block:: cpp

  #include <matx.h>

  // Existing function
  void foo(float *data);

  // Create a MatX tensor in managed memory
  auto t1 = matx::make_tensor<float>({10, 10});

  // MatX processing code

  // Existing code
  foo(t1.Data());

The above example shows an existing function `foo` taking in a pointer from the MatX tensor `t1`. Since only a pointer is available, all 
metadata available in the operator (shape, strides, etc) is not available inside of the function, and the user must ensure the correctness 
of usage with the pointer.

