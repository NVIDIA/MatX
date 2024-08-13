#!/usr/bin/env python3

import numpy as np
import cupy as cp
from cupyx.scipy import linalg as cplinalg
from numpy import random
import math
import matx_common
from typing import Dict, List


class inv:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        n = self.size[-1]
        if len(self.size) == 1:
            shape = (n, n)
        else:
            batch_size = self.size[0]
            shape = (batch_size, n, n)

        # Create an invertible matrix
        A = matx_common.randn_ndarray(shape, self.dtype)
        A_inv = np.linalg.inv(A)

        return {
            'A': A,
            'A_inv': A_inv
        }


class cholesky:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype
        np.random.seed(1234)

    def run(self):
        n = self.size[-1]
        if len(self.size) == 1:
            shape = (n, n)
        else:
            batch_size = self.size[0]
            shape = (batch_size, n, n)

        # Create a positive-definite matrix
        A = matx_common.randn_ndarray(shape, self.dtype)
        B = np.matmul(A, np.conj(A).swapaxes(-2, -1))
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
        m, n = self.size[-2:]
        if len(self.size) == 2:
            shape = (m, n)
        else:
            batch_size = self.size[0]
            shape = (batch_size, m, n)

        A = matx_common.randn_ndarray(shape, self.dtype)
        A_cp = cp.asarray(A)

        if len(self.size) == 2:
            P_cp, L_cp, U_cp = cplinalg.lu(A_cp)
        else:
            P_list = []
            L_list = []
            U_list = []

            for i in range(batch_size):
                P_i, L_i, U_i = cplinalg.lu(A_cp[i])
                P_list.append(P_i)
                L_list.append(L_i)
                U_list.append(U_i)
            
            P_cp = cp.stack(P_list)
            L_cp = cp.stack(L_list)
            U_cp = cp.stack(U_list)

        cp.cuda.Stream.null.synchronize()
        P, L, U = cp.asnumpy(P_cp), cp.asnumpy(L_cp), cp.asnumpy(U_cp)

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
        m, n = self.size[-2:]
        if len(self.size) == 2:
            shape = (m, n)
        else:
            batch_size = self.size[0]
            shape = (batch_size, m, n)

        A = matx_common.randn_ndarray(shape, self.dtype)
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
        m, n = self.size[-2:]
        if len(self.size) == 2:
            shape = (m, n)
        else:
            batch_size = self.size[0]
            shape = (batch_size, m, n)

        A = matx_common.randn_ndarray(shape, self.dtype)
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
        n = self.size[-1]
        if len(self.size) == 1:
            shape = (n, n)
        else:
            batch_size = self.size[0]
            shape = (batch_size, n, n)
        
        # Create a positive-definite matrix
        A = matx_common.randn_ndarray(shape, self.dtype)
        B = np.matmul(A, np.conj(A).swapaxes(-2, -1))
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
        n = self.size[-1]
        if len(self.size) == 1:
            shape = (n, n)
        else:
            batch_size = self.size[0]
            shape = (batch_size, n, n)

        # Create an invertible matrix
        A = matx_common.randn_ndarray(shape, self.dtype)
        det = np.linalg.det(A)

        return {
            'A': A,
            'det': det
        }
