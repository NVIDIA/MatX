import cupy as cp
import sys
sys.path.append('.')
import matxutil

def my_func(dlp):
  print(f"  type(dlp) before cp.from_dlpack(): {type(dlp)}")
  print(f"  dlp capsule name is: {matxutil.get_capsule_name(dlp)}")
  a = cp.from_dlpack(dlp)
  print(f"  type(dlp) after cp.from_dlpack(): {type(dlp)}")
  print(f"  dlp capsule name is: {matxutil.get_capsule_name(dlp)}")
  print(f"  type(cp.from_dlPack(dlp)): {type(a)}")
  print()
  print("Finally, print the tensor we received from MatX using python:")
  print(a)
