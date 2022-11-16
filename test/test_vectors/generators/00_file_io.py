#!/usr/bin/env python3

import numpy as np
from typing import Dict, List
import os


class csv:
    def __init__(self, dtype: str, sizes: List[int]):
        self.dtype = dtype
        self.files = ("../test/00_io/small_csv_comma_nh.csv", "../test/00_io/small_csv_complex_comma_nh.csv")

    def run(self) -> Dict[str, np.array]:
        res = {}
        for file in self.files:
            res[file] = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=self.dtype)

        return res