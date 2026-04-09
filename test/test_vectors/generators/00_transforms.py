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
        # size[3] is decimation_factor; remaining dimensions are batch dimensions
        if len(size) > 4:
            a_dims = size[4:]
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

    # Phase rotation convention for the oversampled channelizer.
    # Must match CHANNELIZE_POLY1D_OVERSAMPLED_FIRST_PHASE_ROTATION in channelize_poly.cuh.
    # 0: Harris convention (no rotation at t=0).
    # 1: MATLAB dsp.Channelizer convention (one rotation at t=0, after priming with one zero).
    OVERSAMPLED_FIRST_PHASE_ROTATION = 0

    def channelize_oversampled(self) -> Dict[str, np.ndarray]:
        """General polyphase channelizer supporting arbitrary decimation factor D <= M.

        Uses size[3] as decimation_factor (D). When D == M, this produces the same
        result as channelize(). When D < M, this is the oversampled case.
        """
        def idivup(a, b) -> int: return (a+b-1)//b

        h = self.res['filter_random']
        num_channels = self.res['num_channels']
        decimation_factor = self.size[3]
        x = self.res['a']

        M = num_channels
        D = decimation_factor

        # Polyphase decompose: E[r, q] = h[q*M + r], zero-pad h to multiple of M
        num_taps_per_channel = idivup(h.size, M)
        h_padded = h
        if M * num_taps_per_channel > h.size:
            h_padded = np.pad(h, (0, M * num_taps_per_channel - h.size))
        E = np.reshape(h_padded, (M, num_taps_per_channel), order='F')

        num_output_samples = idivup(x.shape[-1], D)
        num_batches = x.size // x.shape[-1]
        out = np.zeros((num_batches, num_output_samples, M), dtype=np.complex128)
        out_hreal = np.zeros((num_batches, num_output_samples, M), dtype=np.complex128)
        xr = np.reshape(x, (num_batches, x.shape[-1]))

        E_real = np.real(E)
        n_idx = np.arange(num_output_samples)

        # Precompute per-(n, r) quantities as 2D arrays [num_output_samples, M]
        # Branch remap: r_remapped = (r + M - D) % M changes the input sample
        # mapping so newest D samples land in Harris convention branches.
        # Phase uses the original logical branch index r (not remapped).
        n_all = np.arange(num_output_samples)[:, np.newaxis]  # [nout, 1]
        r_all = np.arange(M)[np.newaxis, :]                    # [1, M]
        r_remapped = (r_all + M - D) % M                       # [1, M]
        s_all = M - 1 - r_remapped                              # [1, M] (remapped for input access)
        last_arrived = n_all * D + D - 1                        # [nout, 1] broadcast
        valid = last_arrived >= s_all                            # [nout, M]
        A = np.where(valid, last_arrived - s_all, 0)            # [nout, M]
        newest = np.where(valid, last_arrived - (A % M), 0)     # [nout, M]
        causal_count = np.where(valid, A // M + 1, 0)           # [nout, M]
        phase = np.where(valid,
            (r_all + ((n_all + channelize_poly_operators.OVERSAMPLED_FIRST_PHASE_ROTATION) * D) % M) % M,
            0).astype(int)                                       # [nout, M] (original r, NOT remapped)

        for batch_ind in range(num_batches):
            xb = xr[batch_ind, :]
            input_len = xb.shape[-1]
            filtered = np.zeros((num_output_samples, M), dtype=np.complex128)
            filtered_hreal = np.zeros((num_output_samples, M), dtype=np.complex128)

            # Loop only over taps (P iterations, typically small)
            for q in range(num_taps_per_channel):
                idx = newest - q * M                             # [nout, M]
                tap_valid = valid & (idx >= 0) & (idx < input_len) & (q < causal_count)
                if not np.any(tap_valid):
                    continue
                # Gather input samples and filter taps for all valid (n, r) pairs
                inp_vals = np.where(tap_valid, xb[np.clip(idx, 0, input_len - 1).astype(int)], 0)
                h_vals = E[phase, q]                             # [nout, M]
                h_real_vals = E_real[phase, q]
                filtered += np.where(tap_valid, h_vals * inp_vals, 0)
                filtered_hreal += np.where(tap_valid, h_real_vals * inp_vals, 0)
            # IFFT across channels, scale by M
            out[batch_ind, :, :] = ifft(filtered, n=M, axis=1) * M
            out_hreal[batch_ind, :, :] = ifft(filtered_hreal, n=M, axis=1) * M

        # Reshape output: (num_batches, num_output_samples, M) -> match MatX layout
        # MatX output is (...batch_dims..., num_output_samples, num_channels)
        if num_batches > 1:
            s = list(x.shape)
            s[-1] = num_output_samples
            s = np.append(s, M)
            out = np.reshape(out, s)
            out_hreal = np.reshape(out_hreal, s)
        else:
            out = np.reshape(out, out.shape[1:])
            out_hreal = np.reshape(out_hreal, out_hreal.shape[1:])

        self.res['b_random'] = out
        self.res['b_random_hreal'] = out_hreal
        return self.res

class harris2003_oversampled_operators:
    """Test case based on the oversampled channelizer from Harris 2003
    (receiver_40z.m): M=40, D=28, 600-tap filter designed with remez.

    Golden input and output are read from a binary file. The filter is
    designed using scipy.signal.remez with the same specification as the
    reference implementation (verified to match within machine epsilon).
    The binary file layout is:
      [real(xx), 5600 doubles][imag(xx), 5600 doubles]
      [real(yy), 40*200 doubles][imag(yy), 40*200 doubles]
    where yy is [40, 200] stored column-major (channels × output steps)
    with fftshift applied (DC centered).
    """
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype

    def channelize(self) -> Dict[str, np.ndarray]:
        import os
        import scipy.signal

        M = 40
        D = 28
        input_len = 5600
        num_output = 200

        # Design the 600-tap prototype filter using Parks-McClellan (remez)
        # with the same specification as the Harris 2003 reference.
        freqs = np.array([0, 12, 17, 56, 57, 84, 85, 112, 113, 140,
                          141, 168, 169, 196, 197, 224, 225, 252, 253, 280,
                          281, 308, 309, 336, 337, 364, 365, 392, 393, 420,
                          421, 448, 449, 476, 477, 504, 505, 532, 533, 560]) / 560.0
        gains = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        weights = [1, 6, 7, 9, 11, 13, 15, 17, 19, 21,
                   23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
        h = scipy.signal.remez(600, freqs, gains, weight=weights, fs=2)
        h = h.astype(np.float64)

        # Read golden input and output from binary test vector file
        tv_dir = os.path.join(os.path.dirname(__file__), '..', 'channelize_poly')
        tv_path = os.path.join(tv_dir, 'harris2003_oversampled_testvector.bin')
        data = np.fromfile(tv_path, dtype=np.float64)

        offset = 0
        xx_re = data[offset:offset + input_len]; offset += input_len
        xx_im = data[offset:offset + input_len]; offset += input_len
        a = (xx_re + 1j * xx_im).astype(np.complex128)

        yy_re = data[offset:offset + M * num_output]; offset += M * num_output
        yy_im = data[offset:offset + M * num_output]; offset += M * num_output
        # The reference output uses fftshift (DC centered). Apply ifftshift
        # to convert to our convention (DC at index 0).
        from scipy.fft import ifftshift
        yy = (yy_re + 1j * yy_im).reshape((M, num_output), order='F')
        b = ifftshift(yy, axes=0).T

        res = {
            'a': a,
            'filter': h,
            'b_golden': b,
        }
        return res


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
    
    def fft_1d_scaled(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],), self.dtype)
        return {
            'a_in': seq,
            'a_out': np.fft.fft(seq, self.size[1]) * 5.0
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
    
    def normalize_maxnorm(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'in_m': seq,
            'out_m': seq / np.linalg.norm(seq, ord=np.inf, axis=0, keepdims=True)
        }
    
    def normalize_lpnorm(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'in_m': seq,
            'out_m': seq / np.linalg.norm(seq, ord=2, axis=0, keepdims=True)
        }
    
    def normalize_zscore(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'in_m': seq,
            'out_m': (seq - np.mean(seq, axis=0, keepdims=True)) / np.std(seq, axis=0, ddof=1, keepdims=True)
        }
    
    def normalize_range(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        a = 0.0
        b = 1.0
        seq_min = np.min(seq, 0, keepdims=True)
        seq_max = np.max(seq, 0, keepdims=True)
        return {
            'in_m': seq,
            'out_m': a + (seq - seq_min) / (seq_max - seq_min) * (b - a)
        }
    
    def normalize_scale(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        seq_std = np.std(seq, axis=0, ddof=1, keepdims=True)
        return {
            'in_m': seq,
            'seq_std': seq_std,
            'scaled_std': np.ones(self.size[1], dtype=self.dtype)
        }
    
    def normalize_center(self) -> Dict[str, np.ndarray]:
        seq = matx_common.randn_ndarray((self.size[0],self.size[1]), self.dtype)
        return {
            'in_m': seq,
            'out_m': (seq - np.mean(seq, axis=0, keepdims=True))
        }