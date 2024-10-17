.. _memory:

Memory
======

MatX allows tensors to be allocated in several different spaces either corresponding to physical or logical
allocations. The space where the memory is allocated dictates where a user is allowed to access the memory.
For example, allocating device memory on some system is not accessible by the CPU and will result in a SEGFAULT
when trying to access.

The type of allocation may behave differently across systems. For example, on a Grace-Hopper system (GH200) 
standard host memory from malloc is accessible from the GPU, but may not be on other platforms. Some types
may not be available in certain environments. On WSL2 CUDA Unified Memory (UM) or managed memory is not fully
supported and may result in slow code or other issues.

The memory type is typically chosen when creating a tensor with `make_tensor`. The memory *may* be allocated
immediately, but it is not guaranteed. The memory is guaranteed to be available before it used used, however.

.. doxygenenum:: matxMemorySpace_t