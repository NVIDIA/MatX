.. _tensor_api:

Tensor Type
###########

A tensor in MatX ``tensor_t``` is a memory-backed, reference-counted operator that contains metadata about the 
size, rank, and other properties. The type of memory can be anything that is accessible to where the tensor is 
being used, including device memory, managed memory, and host memory. MatX tensors are very similar to NumPy's 
`ndarray` type in that common operations like slicing and cloning can be performed on them. Since MatX tensors 
are also operators they are designed to be accepted as both inputs and outputs to almost all functions.

``tensor_t`` uses a ``std::shared_ptr`` for reference-counting the number of times the tensor is shared. This
allows the tensor to be passed around on the host by value, and when the last owner goes out of scope the 
destructor is called, optionally freeing the tensor's memory.

Tensors can be used on both the host and device. This allows custom operators and functions to utilize the same
functionality, such as ``operator()`` that's available on the host. Passing tensors to the device is preferred
over raw pointers since tensors maintain their shape and strides to ensure correct accesses with no extra overhead. 
Since ``tensor_t`` contains types that are not available on the device (``std::shared_ptr`` for example), 
when a tensor is passed to the device it is upcasted to the base class of ``tensor_impl_t``. ``tensor_impl_t``
contains only types that are available on both the host and device, and provides a minimal set of functionality
needed for device code.

For information on creating tensors, please see :ref:`creating` or :ref:`quickstart` for common usage.
For information on experimental sparse tensor support, please see :ref:`sparse_tensor_api`.

.. doxygenclass:: matx::tensor_t
    :members:
