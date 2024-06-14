#!/usr/bin/env python3

import math
import numpy as np
import sys
from scipy import io
from scipy import signal
from scipy.constants import c, pi
from scipy.fft import ifft
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
            'a': matx_common.randn_ndarray((size[0], size[1]), dtype)
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

class channelize_poly_operators:
    np_random_state = None
    def __init__(self, dtype: str, size: List[int]):
        if not channelize_poly_operators.np_random_state:
            # We want reproducible results, but do not want to create random vectors that
            # are too similar between test cases. If we seed every time with the same value
            # and then create test cases with e.g. 1000 and 2000 samples, the first 1000
            # samples will be identical in both case. Thus, we seed only once and store the
            # state from one call to the next thereafter.
            np.random.seed(1234)
        else:
            np.random.set_state(channelize_poly_operators.np_random_state)

        self.size = size
        self.dtype = dtype
        signal_len = size[0]
        filter_len = size[1]
        num_channels = size[2]
        # Remaining dimensions are batch dimensions
        if len(size) > 3:
            a_dims = size[3:]
            a_dims = np.append(a_dims, signal_len)
        else:
            a_dims = [signal_len]
        self.res = {
            'a': matx_common.randn_ndarray(a_dims, dtype=dtype),
            'filter_random': matx_common.randn_ndarray((filter_len,), dtype=dtype),
            'num_channels': num_channels,
        }
        # In some cases we test with real filters and complex inputs
        self.res['filter_random_real'] = np.real(self.res['filter_random'])

        channelize_poly_operators.np_random_state = np.random.get_state()

    def channelize(self) -> Dict[str, np.ndarray]:
        def idivup(a, b) -> int: return (a+b-1)//b

        h = self.res['filter_random']
        num_channels = self.res['num_channels']
        x = self.res['a']
        num_taps_per_channel = idivup(h.size, num_channels)
        if num_channels * num_taps_per_channel > h.size:
            h = np.pad(h, (0,num_channels*num_taps_per_channel-h.size))
        h = np.reshape(h, (num_channels, num_taps_per_channel), order='F')
        x_len_per_channel = idivup(x.shape[-1], num_channels)
        x_pad_len = x_len_per_channel * num_channels
        num_batches = x.size // x.shape[-1]
        out = np.zeros((num_batches, num_channels, x_len_per_channel), dtype=np.complex128)
        out_hreal = np.zeros((num_batches, num_channels, x_len_per_channel), dtype=np.complex128)
        xr = np.reshape(x, (num_batches, x.shape[-1]))
        for batch_ind in range(num_batches):
            xpad = xr[batch_ind, :]
            if x_pad_len > x.shape[-1]:
                xpad = np.pad(xpad, (0,x_pad_len-x.shape[-1]))
            # flipud because samples are inserted into the filter banks in order
            # M-1, M-2, ..., 0
            xf = np.flipud(np.reshape(xpad, (num_channels,x_len_per_channel), order='F'))
            buf = np.zeros((num_channels, num_taps_per_channel), dtype=self.dtype)

            # We scale the outputs by num_channels because we use the ifft
            # and it scales by 1/N for an N-point FFT. We use ifft instead
            # of fft because the complex exponentials in the Harris paper
            # (c.f. Equation 17) are exp(j * ...) instead of exp(-j * ...)
            # whereas scipy uses the negative version for DFTs.
            scale = num_channels
            for i in range(x_len_per_channel):
                buf[:, 1:] = buf[:, 0:num_taps_per_channel-1]
                buf[:, 0] = xf[:, i]
                for j in range(num_channels):
                    out[batch_ind, j, i] = scale * np.dot(np.squeeze(buf[j,:]), np.squeeze(h[j,:]))
                    out_hreal[batch_ind, j, i] = scale * np.dot(np.squeeze(buf[j,:]), np.squeeze(np.real(h[j,:])))
            out[batch_ind,:,:] = ifft(out[batch_ind,:,:], axis=0)
            out_hreal[batch_ind,:,:] = ifft(out_hreal[batch_ind,:,:], axis=0)
        if num_batches > 1:
            s = list(x.shape)
            s[-1] = num_channels
            s = np.append(s, x_len_per_channel)
            perm = np.arange(len(x.shape)+1)
            perm[-2] = len(x.shape)
            perm[-1] = len(x.shape)-1
            out = np.transpose(np.reshape(out, s), axes=perm)
            out_hreal = np.transpose(np.reshape(out_hreal, s), axes=perm)
        else:
            out = np.transpose(np.reshape(out, out.shape[1:]), axes=[1,0])
            out_hreal = np.transpose(np.reshape(out_hreal, out_hreal.shape[1:]), axes=[1,0])
        self.res['b_random'] = out
        self.res['b_random_hreal'] = out_hreal
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

    def fft_1d_ortho(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft(seq, self.size[1], norm="ortho")
        }

    def fft_1d_fwd(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft(seq, self.size[1], norm="forward")
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

    def ifft_1d_ortho(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.ifft(seq, self.size[1], norm="ortho")
        }

    def ifft_1d_fwd(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.ifft(seq, self.size[1], norm="forward")
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
            'a_out': np.fft.fft2(seq, (self.size[0], self.size[1]))
        }

    def fft_2d_batched(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray(
            (self.size[0], self.size[1], self.size[2]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft2(seq, (self.size[1], self.size[2]))
        }

    def fft_2d_batched_strided(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray(
            (self.size[0], self.size[1], self.size[2]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft2(seq, (self.size[0], self.size[2]), axes=(0, 2))
        }

    def ifft_2d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray(
            (self.size[0], self.size[1]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.ifft2(seq, (self.size[0], self.size[1]))
        }

    def rfft_2d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray(
            (self.size[0], self.size[1]), self.dtype)

        return {
            'a_in': seq,
            'a_out': np.fft.rfft2(seq, (self.size[0], self.size[1]))
        }

    def irfft_2d(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray(
            (self.size[0], self.size[1]), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.irfft2(seq, (self.size[0], self.size[1]))
        }


class outer_operators:
    def __init__(self,  dtype: str, size: List[int]):
        np.random.seed(1234)
        self.size = size
        self.dtype = dtype

        self.res = {
            'a': matx_common.randn_ndarray((size[-2],), dtype),
            'b': matx_common.randn_ndarray((size[-1],), dtype),
            'ba': matx_common.randn_ndarray((size[-3], size[-2]), dtype),
            'bb': matx_common.randn_ndarray((size[-3], size[-1]), dtype)
        }

    def run(self) -> Dict[str, np.ndarray]:
        self.res['c'] = np.outer(self.res['a'], self.res['b'])
        self.res['bc'] = np.ndarray((self.size[0], self.size[1], self.size[2]), dtype=self.dtype)
        for b in range(self.size[-3]):
            self.res['bc'][b] = np.outer(self.res['ba'][b], self.res['bb'][b])
            
        return self.res

class norm_operators:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def vector_l2(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'in_v': seq,
            'out_v': np.linalg.norm(seq, 2)
        }
    
    def vector_l1(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'in_v': seq,
            'out_v': np.linalg.norm(seq, 1)
        }
    
    def matrix_frob(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'in_m': seq,
            'out_m': np.linalg.norm(seq, 'fro')
        }
    
    def matrix_l1(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'in_m': seq,
            'out_m': np.linalg.norm(seq, 1)
        }       