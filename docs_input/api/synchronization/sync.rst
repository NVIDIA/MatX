.. _sync_func:

sync
====

Wait for any code running on an executor to complete. For CUDA executors this typically synchronizes 
the stream backing the executor, while host executors wait until the calling thread completes.

.. versionadded:: 0.1.0

Examples
~~~~~~~~

.. literalinclude:: ../../../examples/cgsolve.cu
   :language: cpp
   :start-after: example-begin sync-test-1
   :end-before: example-end sync-test-1
   :dedent:
