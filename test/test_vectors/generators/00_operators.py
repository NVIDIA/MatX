#!/usr/bin/env python3

import numpy as np
import scipy.signal as ss
import scipy.linalg as sl
from typing import Dict, List
import matx_common

class polyval_operator:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        pass

    def run(self) -> Dict[str, np.array]:
        c = np.random.rand(self.size[0])
        x = np.random.rand(self.size[1])

        return {
            'c': c,
            'x': x,
            'out': np.polyval(c, x),
        }


class kron_operator:
    def __init__(self, dtype: str, size: List[int]):
        pass

    def run(self) -> Dict[str, np.array]:
        b = np.array([[1, -1], [-1, 1]])
        self.square = np.kron(np.eye(4), b)

        a = np.array([[1, 2, 3], [4, 5, 6]])
        self.rect = np.kron(a, np.ones([2, 2]))

        return {
            'square': self.square,
            'rect': self.rect
        }


class meshgrid_operator:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size

    def run(self) -> Dict[str, np.array]:
        self.x = np.linspace(1, self.size[0], self.size[0])
        self.y = np.linspace(1, self.size[1], self.size[1])
        [X, Y] = np.meshgrid(self.x, self.y)

        return {
            'X': X,
            'Y': Y
        }


class window:
    def __init__(self, dtype: str, size: List[int]):
        self.win_size = size[0]

    def run(self) -> Dict[str, np.array]:
        self.hamming = np.hamming(self.win_size)
        self.hanning = np.hanning(self.win_size)
        self.blackman = np.blackman(self.win_size)
        self.bartlett = np.bartlett(self.win_size)
        self.flattop = ss.windows.flattop(self.win_size)

        return {
            'hamming': self.hamming,
            'hanning': self.hanning,
            'blackman': self.blackman,
            'bartlett': self.bartlett,
            'flattop': self.flattop
        }


class stats:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size

    def run(self) -> Dict[str, np.array]:
        x = np.random.rand(self.size[0])
        var_ub = np.var(x)
        var_ml = np.var(x, ddof = 0)
        std = np.std(x)

        return {
            'x': x,
            'var_ub': var_ub,
            'var_ml': var_ml,
            'std': std
        }

class contraction:
    def __init__(self, dtype: str, size: List[int]):
        pass

    def run(self) -> Dict[str, np.array]:
        a1 = np.arange(60.).reshape(3,4,5)
        b1 = np.arange(24.).reshape(4,3,2)
        c1 = np.einsum('ijk,jil->kl', a1, b1)

        return {
            'a_float3d': a1,
            'b_float3d': b1,
            'c_float3d': c1
        }

class toeplitz:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype        
        pass

    def run(self) -> Dict[str, np.array]:
        c = matx_common.randn_ndarray((self.size[0],), self.dtype)
        r = matx_common.randn_ndarray((self.size[1],), self.dtype)
        r2 = matx_common.randn_ndarray((self.size[2],), self.dtype)

        # The first element in each array must match
        r[0]  = c[0]
        r2[0] = c[0]

        return {
            'c': c,
            'r': r,
            'r2': r2,
            'out1': sl.toeplitz(c),
            'out2': sl.toeplitz(c, r),
            'out3': sl.toeplitz(c, r2)
        }        

class pwelch_operators:
    def __init__(self, dtype: str, cfg: Dict): #PWelchGeneratorCfg):
        self.dtype = dtype

        self.signal_size = cfg['signal_size']
        self.nperseg = cfg['nperseg']
        self.noverlap = cfg['noverlap']
        self.nfft = cfg['nfft']
        self.ftone = cfg['ftone']
        self.sigma = cfg['sigma']
        self.window_name = cfg['window_name']
        if (cfg['window_name'] == 'none'):
            self.window_name = 'boxcar'

        np.random.seed(1234)


    def pwelch_complex_exponential(self) -> Dict[str, np.ndarray]:
        s = np.exp(2j*np.pi*self.ftone*np.linspace(0,self.signal_size-1,self.signal_size)/self.nfft)
        n = np.random.normal(loc=0,scale=self.sigma,size=self.signal_size) + 1j*np.random.normal(loc=0,scale=self.sigma,size=self.signal_size)
        x = s + n
        w = ss.get_window(self.window_name,self.nperseg,fftbins=False)
        f, Pxx = ss.welch(x,
                          fs=1.,
                          window=w,
                          nperseg=self.nperseg,
                          noverlap=self.noverlap,
                          nfft=self.nfft,
                          return_onesided=False,
                          scaling = 'density',
                          detrend=False)
        w_scale = np.sum(w*w)
        Pxx = Pxx * w_scale
        return {
            'x_in': x,
            'Pxx_out': Pxx
        }
