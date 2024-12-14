import cupy as cp
import sys

# Add path . if we built as a stand-alone project
sys.path.append('.')

# Add path examples/python_integration_sample/ if we built as part of MatX examples
sys.path.append('examples/python_integration_sample/')

import matxutil

# Demonstrate dlpack consumption invalidates it for future use
def dlp_usage_error():
  a = cp.empty((3,3), dtype=cp.float32)
  dlp = a.toDlpack()
  assert(matxutil.check_dlpack_status(dlp) == 0)
  a2 = cp.from_dlpack(dlp) # causes dlp to become unused
  assert(matxutil.check_dlpack_status(dlp) != 0)
  return dlp

# Demonstrate cupy array stays in scope when returning valid dlp
def scope_okay():
  a = cp.empty((3,3), dtype=cp.float32)
  a[1,1] = 2
  dlp = a.toDlpack()
  assert(matxutil.check_dlpack_status(dlp) == 0)
  return dlp

#Do all cupy work using the "with stream" context manager
stream = cp.cuda.stream.Stream(non_blocking=True)
with stream:
   print("Demonstrate dlpack consumption invalidates it for future use:")
   dlp = dlp_usage_error()
   assert(matxutil.check_dlpack_status(dlp) != 0)
   print(f"  dlp capsule name is: {matxutil.get_capsule_name(dlp)}")
   print()

   print("Demonstrate cupy array stays in scope when returning valid dlpack:")
   dlp = scope_okay()
   assert(matxutil.check_dlpack_status(dlp) == 0)
   print(f"  dlp capsule name is: {matxutil.get_capsule_name(dlp)}")
   print()

   print("Print info about the dlpack:")
   matxutil.print_dlpack_info(dlp)
   print()

   print("Use MatX to print the tensor:")
   matxutil.print_float_2D(dlp)
   print()

   print("Print current memory usage info:")
   gpu_mempool = cp.get_default_memory_pool()
   pinned_mempool = cp.get_default_pinned_memory_pool()
   print(f"  GPU mempool used bytes {gpu_mempool.used_bytes()}")
   print(f"  Pinned mempool n_free_blocks {pinned_mempool.n_free_blocks()}")
   print()

   print("Demonstrate python to C++ to python to C++ calling chain (uses mypythonlib.py):")
   # This function calls back into python and executes a from_dlpack, consuming the dlp
   matxutil.call_python_example(dlp)
   assert(matxutil.check_dlpack_status(dlp) != 0)
   del dlp

   print("Demonstrate adding two tensors together using MatX:")
   a = cp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=cp.float32)
   b = cp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=cp.float32)
   c = cp.empty(b.shape, dtype=b.dtype)

   c_dlp = c.toDlpack()
   a_dlp = a.toDlpack()
   b_dlp = b.toDlpack()
   matxutil.add_float_2D(c_dlp, a_dlp, b_dlp, stream.ptr)
   stream.synchronize()
   print(f"Tensor a {a}")
   print(f"Tensor b {b}")
   print(f"Tensor c=a+b {c}")
