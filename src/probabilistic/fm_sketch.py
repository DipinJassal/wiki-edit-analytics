"""Flajolet-Martin Sketch for distinct count estimation."""

import mmh3
import numpy as np


class FMSketch:
    def __init__(self, num_hashes=64):
        self.num_hashes = num_hashes
        self.max_trailing_zeros = np.zeros(num_hashes, dtype=np.int32)

    @staticmethod
    def _trailing_zeros(value):
        if value == 0:
            return 32
        count = 0
        while (value & 1) == 0:
            count += 1
            value >>= 1
        return count

    def add(self, item):
        s = str(item)
        for i in range(self.num_hashes):
            h = abs(mmh3.hash(s, seed=i))
            tz = self._trailing_zeros(h)
            if tz > self.max_trailing_zeros[i]:
                self.max_trailing_zeros[i] = tz

    def estimate(self):
        """Estimate distinct count using stochastic averaging (groups of 8)."""
        if self.num_hashes <= 8:
            avg = np.mean(self.max_trailing_zeros)
            return int((2 ** avg) / 0.77351)

        group_size = 8
        num_groups = self.num_hashes // group_size
        group_medians = [
            np.median(self.max_trailing_zeros[g * group_size:(g + 1) * group_size])
            for g in range(num_groups)
        ]
        avg = np.mean(group_medians)
        return int((2 ** avg) / 0.77351)

    def merge(self, other):
        assert self.num_hashes == other.num_hashes, "Sketch sizes must match"
        self.max_trailing_zeros = np.maximum(
            self.max_trailing_zeros, other.max_trailing_zeros
        )

    def reset(self):
        self.max_trailing_zeros = np.zeros(self.num_hashes, dtype=np.int32)
