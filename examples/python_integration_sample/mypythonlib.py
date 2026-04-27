import cupy as cp

def python_print(dlp):
  # Convert the DLPack capsule to a cupy array
  a = cp.from_dlpack(dlp)
  # Print the tensor using python
  print("shape:", a.shape, "dtype:", a.dtype)
  print(a)
