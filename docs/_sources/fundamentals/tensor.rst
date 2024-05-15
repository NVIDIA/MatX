.. _tensor_api:

Tensors
#######

A tensor in MatX `tensor_t` is a memory-backed operator that contains metadata about the size, rank, and other
properties. The type of memory can be anything that is accessible to where the tensor is being used, including
device memory, managed memory, and host memory. MatX tensors are very similar to NumPy's `ndarray` type in that 
common operations like slicing and cloning can be performed on them. Since MatX tensors are also operators they
are designed to be accepted as both inputs and outputs to almost all functions.

For information on creating tensors, please see :ref:`creating` or :ref:`quickstart` for common usage.

.. doxygenclass:: matx::tensor_t
    :members:
