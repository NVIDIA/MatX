#!/usr/bin/env python3

import math
import numpy as np
import sys
from scipy import io
from scipy import signal
from scipy.constants import c, pi
import matx_common
from typing import Dict, List


class conv_operators:
    def __init__(self, dtype: str, size: List[int]):
        np.random.seed(1234)
        self.a = matx_common.randn_ndarray((size[0],), dtype)
        self.b = matx_common.randn_ndarray((size[1],), dtype)
        self.res = {
            'a_op': self.a,
            'b_op': self.b
        }

    def conv(self):
        self.res['conv_full'] = np.convolve(self.a, self.b, 'full')
        self.res['conv_same'] = np.convolve(self.a, self.b, 'same')
        self.res['conv_valid'] = np.convolve(self.a, self.b, 'valid')
        return self.res

    def corr(self):
        self.res['corr'] = np.correlate(self.a, self.b, 'full')
        return self.res

    def corr_swap(self):
        self.res['corr_swap'] = np.correlate(self.b, self.a, 'full')
        return self.res

class conv2d_operators:
    def __init__(self, dtype: str, size: List[int]):
        np.random.seed(1234)
        self.a = matx_common.randn_ndarray((size[0],size[1]), dtype)
        self.b = matx_common.randn_ndarray((size[2],size[3]), dtype)
        self.res = {
            'a_op': self.a,
            'b_op': self.b
        }

    def conv2d(self):
        self.res['conv_full'] = signal.convolve2d(self.a, self.b, 'full')
        self.res['conv_same'] = signal.convolve2d(self.a, self.b, 'same')
        self.res['conv_valid'] = signal.convolve2d(self.a, self.b, 'valid')
        return self.res


class matmul_operators:
    def __init__(self,  dtype: str, size: List[int]):
        np.random.seed(1234)
        self.size = size
        self.dtype = dtype
        if len(size) == 3:
            self.res = {
                'a': matx_common.randn_ndarray((size[-3], size[-2]), dtype),
                'b': matx_common.randn_ndarray((size[-2], size[-1]), dtype)
            }
        else:
            self.res = {
                'a': matx_common.randn_ndarray((*size[:-3], size[-3], size[-2]), dtype),
                'b': matx_common.randn_ndarray((*size[:-3], size[-2], size[-1]), dtype)
            }   

    def run(self) -> Dict[str, np.ndarray]:
        self.res['c'] = self.res['a'] @ self.res['b']

        # Create the strided batched version
        if len(self.res['c'].shape) == 3:
            self.res['cs'] = self.res['c'][::2,:,:]
        return self.res

    def run_a_transpose(self) -> Dict[str, np.ndarray]:
        self.res['a'] = matx_common.randn_ndarray((self.size[1], self.size[0]), self.dtype)
        self.res['c'] = np.transpose(self.res['a']) @ self.res['b']
        return self.res
    def run_b_transpose(self) -> Dict[str, np.ndarray]:
        self.res['b'] = matx_common.randn_ndarray((self.size[2], self.size[1]), self.dtype)
        self.res['c'] = self.res['a'] @ np.transpose(self.res['b'])
        return self.res

    def run_transpose(self) -> Dict[str, np.ndarray]:
        self.res['c'] = np.transpose(self.res['a'] @ self.res['b'])
        return self.res        

    def run_mixed(self) -> Dict[str, np.ndarray]:
        float_to_complex_dtype = {np.float32 : np.complex64, np.float64 : np.complex128}
        a = self.res['a']
        complex_type = float_to_complex_dtype[a.dtype.type]
        complex_a = a.astype(complex_type)
        self.res['c'] = complex_a @ self.res['b']
        return self.res

class cov_operators:
    def __init__(self, dtype: str, size: List[int]):
        np.random.seed(1234)
        self.size = size
        self.res = {
            'a': matx_common.randn_ndarray((size[0], size[0]), dtype)
        }

    def cov(self) -> Dict[str, np.ndarray]:
        # Python uses rows instead of columns for samples. Transpose here to match MATLAB
        c_cov = np.cov(self.res['a'], rowvar=False)

        # When computing covariance, Python uses E[XX'] whereas MATLAB and MatX use E[X'X]. Conjugate the
        # answer here to make them match
        c_cov = np.conj(c_cov)
        self.res['c_cov'] = c_cov

        return self.res

class resample_poly_operators:
    np_random_state = None
    def __init__(self, dtype: str, size: List[int]):
        if not resample_poly_operators.np_random_state:
            # We want reproducible results, but do not want to create random vectors that
            # are too similar between test cases. If we seed every time with the same value
            # and then create test cases with e.g. 1000 and 2000 samples, the first 1000
            # samples will be identical in both case. Thus, we seed only once and store the
            # state from one call to the next thereafter.
            np.random.seed(1234)
        else:
            np.random.set_state(resample_poly_operators.np_random_state)

        self.size = size
        up = size[2]
        down = size[3]
        gcd = math.gcd(up, down)
        up //= gcd
        down //= gcd
        self.res = {
            'a': matx_common.randn_ndarray((size[0],), dtype),
            'filter_random': matx_common.randn_ndarray((size[1],), dtype),
            'up': up,
            'down': down
        }

        # Create a filter compatible with scipy's resample_poly
        max_rate = max(up, down)
        f_c = 1. / max_rate
        half_len = 10 * max_rate
        if up != 1 or down != 1:
            self.res['filter_default'] = signal.firwin(2 * half_len + 1, f_c, window=('kaiser',5.0)).astype(dtype)

        resample_poly_operators.np_random_state = np.random.get_state()

    def resample(self) -> Dict[str, np.ndarray]:
        self.res['b_random'] = signal.resample_poly(self.res['a'], self.res['up'], self.res['down'], window=self.res['filter_random'])
        if 'filter_default' in self.res:
            self.res['b_default'] = signal.resample_poly(self.res['a'], self.res['up'], self.res['down'])
        return self.res

class fft_operators:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def fft_1d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft(seq, self.size[1])
        }

    def fft_1d_batched(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft(seq, self.size[2])
        }        

    def ifft_1d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.ifft(seq, self.size[1])
        }

    def rfft_1d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.rfft(seq, self.size[1])
        }

    def rfft_1d_batched(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.rfft(seq, self.size[2])
        }        

    def irfft_1d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.irfft(seq, self.size[1])
        }

    def fft_2d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray(
            (self.size[0], self.size[1]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft2(seq, (self.size[1], self.size[1]))
        }

    def ifft_2d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray(
            (self.size[0], self.size[1]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.ifft2(seq, (self.size[1], self.size[1]))
        }
