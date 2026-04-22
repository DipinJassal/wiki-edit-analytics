"""Bloom Filter implementation from scratch using MurmurHash3."""

import math
import mmh3


class BloomFilter:
    def __init__(self, size=1_000_000, num_hashes=7):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = bytearray(math.ceil(size / 8))
        self.count = 0

    def _get_positions(self, item):
        s = str(item)
        return [abs(mmh3.hash(s, seed)) % self.size for seed in range(self.num_hashes)]

    def _get_bit(self, pos):
        return (self.bit_array[pos >> 3] >> (pos & 7)) & 1

    def _set_bit(self, pos):
        self.bit_array[pos >> 3] |= 1 << (pos & 7)

    def add(self, item):
        for pos in self._get_positions(item):
            self._set_bit(pos)
        self.count += 1

    def contains(self, item):
        return all(self._get_bit(pos) for pos in self._get_positions(item))

    def false_positive_rate(self):
        if self.count == 0:
            return 0.0
        return (1 - math.exp(-self.num_hashes * self.count / self.size)) ** self.num_hashes

    def memory_bytes(self):
        return len(self.bit_array)

    def reset(self):
        self.bit_array = bytearray(math.ceil(self.size / 8))
        self.count = 0
