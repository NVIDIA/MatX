#!/usr/bin/env python3

import numpy as np
from scipy import signal
from scipy import io
from numpy import random
import math
import os
import matx_common
import cupy as cp
from typing import Dict, List


class simple_pipeline:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)
        self.num_channels = 1

    def set_channels(self, nc: int):
        self.num_channels = nc

    def run(self):
        num_pulses = self.size[0]
        num_uncompressed_range_bins = self.size[1]
        waveform_length = self.size[2]
        num_compressed_range_bins = num_uncompressed_range_bins - waveform_length + 1
        NDfft = 256
        #num_channels = 16
        res = {}

        x = matx_common.randn_ndarray(
            (num_pulses, num_uncompressed_range_bins), self.dtype)
        res['x_init'] = x.copy()

        waveform = matx_common.randn_ndarray((waveform_length,), self.dtype)
        res['waveform'] = waveform.copy()

        window = signal.hamming(waveform_length)
        waveform_windowed = waveform * window
        res['waveform_windowed'] = waveform_windowed.copy()

        waveform_windowed_norm = waveform_windowed / \
            np.linalg.norm(waveform_windowed)
        res['waveform_windowed_norm'] = waveform_windowed_norm.copy()
        Nfft = 2**math.ceil(
            math.log2(np.max([num_uncompressed_range_bins, waveform_length])))
        W = np.conj(np.fft.fft(waveform_windowed_norm, Nfft))
        res['W'] = W.copy()

        X = np.fft.fft(x, Nfft, 1)
        res['X'] = X.copy()

        for pulse in range(num_pulses):
            y = np.fft.ifft(np.multiply(X[pulse, :], W), Nfft, 0)
            x[pulse, 0:num_compressed_range_bins] = y[0:num_compressed_range_bins]
        x_compressed = x[:, 0:num_compressed_range_bins]
        if self.num_channels > 1:
            x_compressed_stack = np.stack([x_compressed] * self.num_channels)
            res['x_compressed'] = x_compressed_stack.copy()
        else:
            res['x_compressed'] = x_compressed.copy()

        x_conv2 = signal.convolve2d(
            x_compressed, np.matrix([[1], [-2], [1]]), 'valid')

        if self.num_channels > 1:
            x_conv2_stack = np.stack([x_conv2] * self.num_channels)
            res['x_conv2'] = x_conv2_stack.copy()
        else:
            res['x_conv2'] = x_conv2.copy()
        num_pulses = x_conv2.shape[0]

        window = np.transpose(np.repeat(np.expand_dims(
            signal.hamming(num_pulses), 0), num_compressed_range_bins, axis=0))

        X_window = np.fft.fft(np.multiply(x_conv2, window), NDfft, 0)

        if self.num_channels > 1:
            X_window_stack = np.stack([X_window] * self.num_channels).copy()
            res['X_window'] = X_window_stack
        else:
            res['X_window'] = X_window.copy()

        mask = np.transpose(np.asarray([[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 0, 0, 0, 1],
                                        [1, 0, 0, 0, 1],
                                        [1, 0, 0, 0, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1]]))
        norm = signal.convolve2d(np.ones(X_window.shape), mask, 'same')
        res['norm_conv2'] = norm.copy()
        Xpow = np.abs(X_window)**2
        res['Xpow'] = Xpow.copy()
        background_averages = np.divide(
            signal.convolve2d(Xpow, mask, 'same'), norm)

        if self.num_channels > 1:
            ba_stacked = np.stack([background_averages] * self.num_channels)
            res['background_averages'] = ba_stacked.copy()
        else:
            res['background_averages'] = background_averages.copy()

        Pfa = 1e-5
        alpha = np.multiply(norm, np.power(Pfa, np.divide(-1.0, norm)) - 1)
        dets = np.zeros(Xpow.shape)
        dets[np.where(Xpow > np.multiply(alpha, background_averages))] = 1
        res['alpha'] = alpha.copy()
        res['dets'] = dets.copy()

        if self.num_channels > 1:
            dets_stacked = np.stack([dets] * self.num_channels)
            res['dets'] = dets_stacked.copy()
        else:
            res['dets'] = dets.copy()

        return res


class ambgfun:

    def __init__(self, dtype: str, size: List[int]):
        cp.random.seed(1234)
        self.size = size
        self.dtype = dtype
        os.environ['CUPY_CACHE_DIR'] = "/tmp"

    def run(self):
        siglen = self.size[0]
        x = matx_common.randn_ndarray((siglen,), self.dtype)
        y = None
        fs = 1e3
        cutValue = 1.0

        _new_ynorm_kernel = cp.ElementwiseKernel(
            "int32 xlen, raw T xnorm, raw T ynorm",
            "T out",
            """
            int row = i / xlen;
            int col = i % xlen;

            int x_col = col - ( xlen - 1 ) + row;
            if ( ( x_col >= 0 ) && ( x_col < xlen ) ) {
                out = ynorm[col] * thrust::conj( xnorm[x_col] );
            } else {
                out = T(0,0);
            }
            """,
            "_new_ynorm_kernel",
            options=("-std=c++11",),
        )

        cut = 'delay'

        if 'float64' in x.dtype.name:
            x = cp.asarray(x, dtype=cp.complex128)
        elif 'float32' in x.dtype.name:
            x = cp.asarray(x, dtype=cp.complex64)
        else:
            x = cp.asarray(x)

        xnorm = x / cp.linalg.norm(x)

        if y is None:
            y = x
            ynorm = xnorm
        else:
            ynorm = y / cp.linalg.norm(y)

        len_seq = len(xnorm) + len(ynorm)
        nfreq = 2**math.ceil(math.log2(len_seq - 1))

        # Consider for deletion as we add different cut values
        """
        if len(xnorm) < len(ynorm):
            len_diff = len(ynorm) - len(xnorm)
            ynorm = cp.concatenate(ynorm, cp.zeros(len_diff))
        elif len(xnorm) > len(ynorm):
            len_diff = len(xnorm) - len(ynorm)
            xnorm = cp.concatenate(xnorm, cp.zeros(len_diff))
        """

        xlen = len(xnorm)

        # if cut == '2d':
        new_ynorm = cp.empty((len_seq - 1, xlen), dtype=xnorm.dtype)
        _new_ynorm_kernel(xlen, xnorm, ynorm, new_ynorm)
        amf_2d = nfreq * cp.abs(cp.fft.fftshift(
            cp.fft.ifft(new_ynorm, nfreq, axis=1), axes=1))

    # elif cut == 'delay':
        Fd = cp.arange(-fs / 2, fs / 2, fs / nfreq)
        fftx = cp.fft.fft(xnorm, nfreq) * \
            cp.exp(1j * 2 * cp.pi * Fd * cutValue)

        xshift = cp.fft.ifft(fftx)

        ynorm_pad = cp.zeros(nfreq) + cp.zeros(nfreq)*1j
        ynorm_pad[:ynorm.shape[0]] = ynorm

        amf_delay = nfreq * cp.abs(cp.fft.ifftshift(
            cp.fft.ifft(ynorm_pad * cp.conj(xshift), nfreq)))

    # elif cut == 'doppler':
        t = cp.arange(0, xlen) / fs
        ffty = cp.fft.fft(ynorm, len_seq - 1)
        fftx = cp.fft.fft(xnorm * cp.exp(1j * 2 * cp.pi * cutValue * t),
                          len_seq - 1)

        amf_doppler = cp.abs(cp.fft.fftshift(
            cp.fft.ifft(ffty * cp.conj(fftx))))

        return {
            'amf_2d': cp.asnumpy(amf_2d),
            'amf_delay': cp.asnumpy(amf_delay),
            'amf_doppler': cp.asnumpy(amf_doppler),
            'x': cp.asnumpy(x),
        }
