#!/usr/bin/env python3

import numpy as np
from scipy import fft as sf
from scipy import signal as ss
from numpy import random
from typing import Dict, List


class dct:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype

    def run(self):
        N = self.size[0]

        # Create a positive-definite matrix
        x = np.random.randn(N,)
        Y = sf.dct(x)

        return {
            'x': x,
            'Y': Y
        }

class chirp:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype

    def run(self):
        N = self.size[0]
        end = self.size[1]
        f0 = self.size[2]
        f1 = self.size[3]

        t = np.linspace(0, end, N)
        Y = ss.chirp(t, f0, t[-1], f1, 'linear')

        return {
            'Y': Y
        }