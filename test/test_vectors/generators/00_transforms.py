#!/usr/bin/env python3

import numpy as np
import sys
from scipy import io
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
        self.res['conv'] = np.convolve(self.a, self.b, 'full')
        return self.res

    def corr(self):
        self.res['corr'] = np.correlate(self.a, self.b, 'full')
        return self.res

    def corr_swap(self):
        self.res['corr_swap'] = np.correlate(self.b, self.a, 'full')
        return self.res


class matmul_operators:
    def __init__(self,  dtype: str, size: List[int]):
        np.random.seed(1234)
        self.size = size
        self.dtype = dtype
        self.res = {
            'a': matx_common.randn_ndarray((size[0], size[1]), dtype),
            'b': matx_common.randn_ndarray((size[1], size[2]), dtype)
        }

    def run(self) -> Dict[str, np.ndarray]:
        self.res['c'] = self.res['a'] @ self.res['b']
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
