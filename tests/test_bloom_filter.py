import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.probabilistic.bloom_filter import BloomFilter


def test_no_false_negatives():
    """Items added to filter must always be found."""
    bf = BloomFilter(size=10000, num_hashes=5)
    for i in range(100):
        bf.add(f"editor_{i}")
    for i in range(100):
        assert bf.contains(f"editor_{i}") is True


def test_false_positive_rate():
    """Empirical FP rate should be well under 5% for these settings."""
    bf = BloomFilter(size=100_000, num_hashes=7)
    for i in range(1000):
        bf.add(f"editor_{i}")
    fps = sum(1 for i in range(1000, 11000) if bf.contains(f"editor_{i}"))
    fp_rate = fps / 10000
    assert fp_rate < 0.05


def test_theoretical_fp_rate():
    """Theoretical FP rate should match expected formula."""
    bf = BloomFilter(size=1_000_000, num_hashes=7)
    for i in range(10_000):
        bf.add(f"editor_{i}")
    # For m=1M, k=7, n=10K: rate should be very small
    assert bf.false_positive_rate() < 0.01


def test_reset_clears_filter():
    """After reset, previously added items should not be found."""
    bf = BloomFilter(size=10000, num_hashes=5)
    bf.add("editor_1")
    assert bf.contains("editor_1") is True
    bf.reset()
    assert bf.contains("editor_1") is False
    assert bf.count == 0


def test_memory_bytes():
    """Memory should be approximately size/8 bytes."""
    bf = BloomFilter(size=800_000, num_hashes=7)
    assert bf.memory_bytes() == 100_000
