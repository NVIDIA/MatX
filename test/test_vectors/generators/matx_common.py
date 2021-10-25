import numpy as np


def tup_2_string(x):
    return '_'.join(reversed(list(map(str, x))))


def to_file(var, name):
    if (var.dtype == np.complex128):
        var.astype(np.complex64).tofile(
            f'{name}_{tup_2_string(var.shape)}_complex64.bin')
    elif (var.dtype == np.float64):
        var.astype(np.float32).tofile(
            f'{name}_{tup_2_string(var.shape)}_float32.bin')
    else:
        var.tofile(f'{name}_{tup_2_string(var.shape)}_{str(var.dtype)}.bin')


def randn_ndarray(tshape, dtype):
    if dtype in ('f8', 'f4'):
        return np.random.randn(*tshape)
    elif dtype in ('c4', 'c8'):
        return np.random.randn(*tshape) + 1j*np.random.randn(*tshape)
