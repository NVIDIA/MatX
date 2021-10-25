#!/usr/bin/env python3

import numpy as np
from scipy import linalg as slinalg
from numpy import random
import math
import matx_common
from typing import Dict, List


class cholesky:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        n = self.size[0]

        # Create a positive-definite matrix
        A = np.random.randn(n, n)
        B = np.matmul(A, A.conj().T)
        B = B + n*np.eye(n)

        L = np.linalg.cholesky(B)

        return {
            'B': B,
            'L': L
        }


class lu:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        m, n = self.size[0], self.size[1]

        A = np.random.randn(m, n)
        P, L, U = slinalg.lu(A)

        return {
            'A': A,
            'P': P,
            'L': L,
            'U': U,
        }


class qr:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        m, n = self.size[0], self.size[1]

        A = np.random.randn(m, n)
        Q, R = np.linalg.qr(A)

        return {
            'A': A,
            'Q': Q,
            'R': R,
        }


class svd:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        m, n = self.size[0], self.size[1]

        A = np.random.randn(m, n)
        U, S, V = np.linalg.svd(A)

        return {
            'A': A,
            'U': U,
            'S': S,
            'V': V
        }


class eig:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        n = self.size[0]
        # Create a positive-definite matrix
        A = np.random.randn(n, n)
        B = np.matmul(A, A.conj().T)
        B = B + n*np.eye(n)

        W, V = np.linalg.eig(B)
        return {
            'B': B,
            'W': W,
            'V': V
        }


class det:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        n = self.size[0]

        # Create a positive-definite matrix
        A = np.random.randn(n, n)
        det = np.linalg.det(A)

        return {
            'A': A,
            'det': det
        }
