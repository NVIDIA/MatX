#!/usr/bin/env python3

import numpy as np
from scipy import fft as sf
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
