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

Internal device allocations
---------------------------

MatX requests ``MATX_ASYNC_DEVICE_MEMORY`` for internal device allocations such as
transform workspaces and temporary outputs. On devices that support CUDA memory
pools, these allocations use ``cudaMallocAsync`` and ``cudaFreeAsync`` with the
associated CUDA stream.
Memory pool support is queried once per device and cached across threads. When
every visible device supports memory pools, MatX calls ``cudaMallocAsync``
directly, and CUDA allocates from the pool of the stream's device. Otherwise MatX
first finds the device that owns the allocation's stream, since that device's
pool is used even when a different device is current, and checks its support.
If the capability query fails, MatX logs a warning and uses the fallback below.

When the stream's device reports ``cudaDevAttrMemoryPoolsSupported == 0``, MatX
automatically falls back to ``cudaMalloc`` and ``cudaFree``, and allocates on the
stream's device rather than the current device. This also applies to
explicit requests for ``MATX_ASYNC_DEVICE_MEMORY`` through ``matxAlloc`` or
``make_tensor``. No build option or application change is required. The fallback
may synchronize and can be slower than stream-ordered allocation. Because
``cudaMalloc`` is not stream-ordered, fallback allocations cannot be captured into
a CUDA graph. A fallback allocation is recorded as ``MATX_DEVICE_MEMORY``, which
is also the value returned by ``GetPointerKind``.
Other CUDA allocation errors are still reported to the caller.
