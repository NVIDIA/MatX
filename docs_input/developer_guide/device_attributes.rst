.. _devdeviceattributes:

Device Attribute Cache
######################

MatX reads several CUDA device attributes on hot paths, such as choosing launch parameters on every CUDA
executor run, picking a JIT target architecture, and choosing allocation fallbacks. These attributes describe
the hardware and never change while a process runs, so MatX queries each device once and caches the results
in ``include/matx/core/device_attributes.h``. Library code should read attributes from this cache rather than
calling ``cudaDeviceGetAttribute`` or ``cudaGetDeviceProperties`` directly.

.. note::
  The cache is internal (``matx::detail``) and is not part of the public API.

Reading attributes
==================

``detail::DeviceAttributes`` holds one field per cached attribute. Four functions return it:

* ``detail::GetCurrentDeviceAttributes()`` returns the entry for the calling thread's current device.
* ``detail::GetDeviceAttributes(device)`` returns the entry for a device ordinal. Prefer it when the
  ordinal is already known, since it skips a ``cudaGetDevice`` call.
* ``detail::TryGetDeviceAttributes(device, attrs)`` and ``detail::TryGetCurrentDeviceAttributes(attrs)``
  return a ``cudaError_t`` instead of throwing. Use them only where the caller has a fallback for a failed
  query.

.. code-block:: cpp

  const auto &attrs = detail::GetCurrentDeviceAttributes();
  if (shm_bytes > attrs.max_smem_per_block_optin) {
    // ...
  }

The throwing functions raise ``matxCudaError`` for an invalid ordinal or a failed CUDA query. The returned
reference stays valid until static destruction at process exit. The existing helpers
``detail::GetComputeCapability()``, ``detail::GetComputeCapabilityMajor()``, ``detail::GetComputeCapabilityMinor()``,
``detail::IsAmpereOrAbove()``, and ``detail::IsHopperOrAbove()`` read from the cache.

``detail::TryGetComputeCapability()`` returns 0 instead of throwing when no device can be queried. Use it in code
that also runs with host executors on machines without a GPU. The MathDx helpers call it each time they need a
target architecture, rather than storing the value when an operator is built, because a JIT kernel launches on
the device that is current when it runs.

``detail::GetDeviceCount()`` returns the number of visible devices from the same cache. Use it for single-GPU
fast paths, such as the allocator skipping the stream's device lookup when only device 0 exists.

Behavior
========

* Every field is filled together on the first lookup for a device. Later lookups take no locks.
  With more than one visible device, ``GetCurrentDeviceAttributes()`` calls ``cudaGetDevice`` on every lookup
  to find the current device, which is most of its cost. With one visible device, the current device can only
  be device 0, so it skips that call.
* A failed query of a required attribute fails the lookup. The optional attributes (L2 size, memory bus width,
  FP64 rate, concurrent managed access, and memory-pool support) only steer heuristics or fallbacks, so a failed
  query of one logs a warning and stores 0, which selects the conservative path.
* Entries are keyed by device ordinal and shared by all threads. The device count is read on the first
  lookup or ``GetDeviceCount()`` call that gets one from CUDA, and is not read again.
* A failed device-count query or failed required-attribute query is not cached. After a caught
  ``matxCudaError`` or a ``TryGetDeviceAttributes`` error, the next lookup queries CUDA again.
* The cache is a function-local static, so it is destroyed at process exit. Do not read it from static
  destructors. ``ClearCaches()`` does not affect it.

Adding an attribute
===================

Only cache attributes that cannot change while the process runs. Clock rates, compute mode, and free or used
memory can change and must be queried directly. To add an attribute:

1. Add a field to ``DeviceAttributes`` with a comment naming its ``cudaDeviceAttr``.
2. Add the matching query to ``QueryDeviceAttributes``. Mark it required unless every caller behaves safely
   when the value is 0.
3. Add a comparison against ``cudaDeviceGetAttribute`` to ``test/00_misc/DeviceAttributeCacheTests.cu``.
