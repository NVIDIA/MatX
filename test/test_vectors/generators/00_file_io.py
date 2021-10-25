#!/usr/bin/env python3

import numpy as np
from typing import Dict, List
import os


class csv:
    def __init__(self, dtype: str, size: List[int]):
        self.size = size
        self.dtype = dtype

    def run(self) -> Dict[str, np.array]:
        small_csv = np.genfromtxt(
            '../test/00_io/small_csv_comma_nh.csv', delimiter=',', skip_header=1)
        return {
            'small_csv': small_csv
        }
