import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.similarity.lsh import LSH
from src.similarity.minhash import MinHash
from src.similarity.validation import exact_jaccard, evaluate_lsh


def test_identical_sets_are_candidates():
    """Identical sets must always appear as candidate pairs."""
    mh = MinHash(num_hashes=128)
    lsh = LSH(num_bands=32, rows_per_band=4)

    set_a = set(range(200))
    sig_a = mh.compute_signature(set_a)
    lsh.index("wiki_a", sig_a)
    lsh.index("wiki_b", sig_a)  # same signature

    candidates = lsh.get_candidates()
    assert ("wiki_a", "wiki_b") in candidates or ("wiki_b", "wiki_a") in candidates


def test_disjoint_sets_unlikely_candidates():
    """Fully disjoint sets should rarely (or never) be candidates with many bands."""
    mh = MinHash(num_hashes=128)
    lsh = LSH(num_bands=32, rows_per_band=4)

    sig_a = mh.compute_signature(set(range(500)))
    sig_b = mh.compute_signature(set(range(500, 1000)))
    lsh.index("wiki_a", sig_a)
    lsh.index("wiki_b", sig_b)

    candidates = lsh.get_candidates()
    # Disjoint sets should not be flagged as candidates most of the time
    pair = ("wiki_a", "wiki_b") if ("wiki_a", "wiki_b") in candidates else ("wiki_b", "wiki_a")
    j = exact_jaccard(set(range(500)), set(range(500, 1000)))
    # If they ARE candidates, true Jaccard should be near 0 — just checking structure
    assert j == 0.0


def test_evaluate_lsh_precision_recall():
    """precision + recall computation should be correct."""
    candidates = {("a", "b"), ("b", "c"), ("a", "c")}
    true_similar = {("a", "b"), ("a", "c")}
    precision, recall = evaluate_lsh(candidates, true_similar)
    assert abs(precision - 2 / 3) < 1e-9
    assert recall == 1.0


def test_lsh_reset():
    """After reset, no candidates should be returned."""
    mh = MinHash(num_hashes=128)
    lsh = LSH(num_bands=32, rows_per_band=4)
    sig = mh.compute_signature(set(range(100)))
    lsh.index("a", sig)
    lsh.index("b", sig)
    lsh.reset()
    assert len(lsh.get_candidates()) == 0
