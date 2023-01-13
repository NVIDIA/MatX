#!/usr/bin/env python3

import numpy as np
import sys
from scipy import special
from scipy.constants import c, pi
import matx_common
from typing import Dict, List


class softmax:
    def __init__(self, dtype: str, size: List[int]):
        np.random.seed(1234)
        self.t1 = matx_common.randn_ndarray((size[-1],), dtype)
        self.t3 = matx_common.randn_ndarray((size[0], size[1], size[2]), dtype)
        self.res = {
            't1': self.t1,
            't3': self.t3
        }

    def run(self):
        self.res['t1'] = self.t1
        self.res['t3'] = self.t3
        self.res['t1_sm'] = special.softmax(self.t1)   
        self.res['t3_sm_axis2'] = special.softmax(self.t3, axis=2)
        return self.res
