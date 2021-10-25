#!/usr/bin/env python3

import numpy as np
from scipy import signal
from scipy import io
from numpy import random
import math
import matx_common
from typing import Dict, List


class mvdr_beamformer:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        data_len = self.size[0]
        num_beams = self.size[1]
        num_el = self.size[2]

        v = np.random.randn(num_el, num_beams) + \
            np.random.randn(num_el, num_beams)*1j
        vh = v.conj().T

        in_vec = np.random.randn(num_el, data_len) + \
            np.random.randn(num_el, data_len)*1j
        out_cbf = np.matmul(vh, in_vec)

        snap_len = 2 * num_el
        load_coeff = 0.1

        inv_slice = in_vec[:, 0:snap_len]
        cov_mat = np.matmul(inv_slice, inv_slice.conj().T) / \
            snap_len + load_coeff * np.eye(num_el)
        cov_inv = np.linalg.inv(cov_mat)

        return {
            'cov_inv': cov_inv,
            'cov_mat': cov_mat,
            'in_vec': in_vec,
            'v': v,
            'out_cbf': out_cbf
        }
