#!/usr/bin/env python3

import numpy as np
import scipy.signal as ss
from typing import Dict, List


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
        self.flattop = ss.flattop(self.win_size)

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
        var = np.var(x)
        std = np.std(x)

        return {
            'x': x,
            'var': var,
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