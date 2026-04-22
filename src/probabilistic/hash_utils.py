"""Shared hash utilities for probabilistic data structures."""

import random
import mmh3


def murmur_hash(item, seed=0):
    """Consistent MurmurHash3 wrapper. Always returns non-negative int."""
    return abs(mmh3.hash(str(item), seed))


def generate_hash_functions(n, prime=4294967311):
    """Generate n hash functions h(x) = (a*x + b) mod prime.

    Returns list of (a, b) tuples. Seeded for reproducibility.
    """
    rng = random.Random(42)
    funcs = []
    for _ in range(n):
        a = rng.randint(1, prime - 1)
        b = rng.randint(0, prime - 1)
        funcs.append((a, b))
    return funcs
