import cupy as cp
import sys

# Add path . if we built as a stand-alone project
sys.path.append('.')

# Add path examples/python_integration_sample/ if we built as part of MatX examples
sys.path.append('examples/python_integration_sample/')

import matxutil

a = cp.arange(9, dtype=cp.float32).reshape(3, 3)

# Convert the cupy array to a DLPack capsule
print("Printing tensor using MatX:")
a_dlp = a.__dlpack__()
# Print the tensor using MatX
matxutil.print_float_2D(a_dlp)

# calling again will throw an error, as the DLPack capsule has been consumed
try:
    matxutil.print_float_2D(a_dlp)
    assert False, "Expected print_float_2D to throw"
except Exception:
    pass

# passing an incompatible tensor type will throw an error
try:
    matxutil.print_float_2D(cp.arange(9, dtype=cp.float64).__dlpack__())
    assert False, "Expected print_float_2D to throw"
except Exception:
    pass

print("Printing tensor using Python called from MatX:")
# valid as we create a new DLPack capsule
matxutil.python_print_float_2D(a.__dlpack__())

print("Adding two tensors together using MatX on the current stream:")
b = cp.ones((3, 3), dtype=cp.float32)
c = cp.empty((3, 3), dtype=cp.float32)
matxutil.add_float_2D(c.__dlpack__(), a.__dlpack__(), b.__dlpack__(), cp.cuda.get_current_stream().ptr)
print(c) # implicit stream synchronization
