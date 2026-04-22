import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.probabilistic.fm_sketch import FMSketch


def test_estimate_accuracy():
    """FM estimate should be within 50% of true count for k=64."""
    fm = FMSketch(num_hashes=64)
    true_count = 10_000
    for i in range(true_count):
        fm.add(f"editor_{i}")
    estimate = fm.estimate()
    error = abs(estimate - true_count) / true_count
    assert error < 0.5


def test_merge():
    """Merged sketch should estimate combined distinct count within 50%."""
    fm1 = FMSketch(num_hashes=64)
    fm2 = FMSketch(num_hashes=64)
    for i in range(5000):
        fm1.add(f"editor_{i}")
    for i in range(3000, 8000):
        fm2.add(f"editor_{i}")
    fm1.merge(fm2)
    estimate = fm1.estimate()
    error = abs(estimate - 8000) / 8000
    assert error < 0.5


def test_small_set():
    """Small sets (n=10) should still get a reasonable estimate."""
    fm = FMSketch(num_hashes=64)
    for i in range(10):
        fm.add(f"editor_{i}")
    estimate = fm.estimate()
    assert estimate > 0


def test_reset():
    """After reset, estimate should be near zero."""
    fm = FMSketch(num_hashes=64)
    for i in range(1000):
        fm.add(f"editor_{i}")
    fm.reset()
    assert fm.estimate() == 0 or fm.estimate() == 1
