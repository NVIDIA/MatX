#!/usr/bin/env python3

import numpy as np
import sys
from scipy import io
from scipy.constants import c, pi
import matx_common
from typing import Dict, List


class matx_python_tests:
    def __init__(self, dtype: str, size: List[int]):
        pass

    def run(self) -> Dict[str, np.ndarray]:
        seye = np.eye(1000, dtype=float)

        return {
            'eye_1000': seye,
        }
