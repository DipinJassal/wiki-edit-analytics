import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.similarity.minhash import MinHash


def test_identical_sets():
    """Identical sets should have estimated Jaccard = 1.0."""
    mh = MinHash(num_hashes=128)
    s = {1, 2, 3, 4, 5}
    sig1 = mh.compute_signature(s)
    sig2 = mh.compute_signature(s)
    assert mh.jaccard_from_signatures(sig1, sig2) == 1.0


def test_disjoint_sets():
    """Disjoint sets should have estimated Jaccard near 0."""
    mh = MinHash(num_hashes=128)
    sig1 = mh.compute_signature({1, 2, 3, 4, 5})
    sig2 = mh.compute_signature({100, 200, 300, 400, 500})
    assert mh.jaccard_from_signatures(sig1, sig2) < 0.1


def test_similar_sets():
    """50% overlapping sets: exact Jaccard = 1/3, estimate within 0.15."""
    mh = MinHash(num_hashes=128)
    set_a = set(range(100))
    set_b = set(range(50, 150))
    sig_a = mh.compute_signature(set_a)
    sig_b = mh.compute_signature(set_b)
    estimated = mh.jaccard_from_signatures(sig_a, sig_b)
    assert abs(estimated - 0.333) < 0.15


def test_signature_length():
    """Signature should have correct length."""
    mh = MinHash(num_hashes=64)
    sig = mh.compute_signature({1, 2, 3})
    assert len(sig) == 64
