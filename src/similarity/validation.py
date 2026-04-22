"""Exact Jaccard computation and LSH precision/recall evaluation."""

import random


def exact_jaccard(set_a, set_b):
    """Compute exact Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def create_synthetic_test_set(original_sets, perturbation_levels=(0.1, 0.2, 0.3, 0.5), seed=42):
    """Create synthetic near-duplicate sets for LSH evaluation.

    For each original set and perturbation level:
    - Remove `level` fraction of elements
    - Add `level` fraction of new random elements
    - Record expected Jaccard similarity

    Args:
        original_sets: dict mapping id -> set of editor IDs
        perturbation_levels: fractions to perturb

    Returns:
        list of (original_id, synthetic_id, modified_set, expected_jaccard)
    """
    rng = random.Random(seed)
    test_pairs = []

    all_ids = [eid for s in original_sets.values() for eid in s]
    max_id = max(all_ids) if all_ids else 0

    for wiki_id, editor_set in original_sets.items():
        editors = list(editor_set)
        for level in perturbation_levels:
            num_remove = int(len(editors) * level)
            remaining = set(rng.sample(editors, len(editors) - num_remove))

            num_add = int(len(editors) * level)
            new_editors = set(
                rng.randint(max_id + 1, max_id + 10_000_000) for _ in range(num_add)
            )
            modified = remaining | new_editors

            expected_j = exact_jaccard(editor_set, modified)
            synthetic_id = f"{wiki_id}_perturbed_{int(level * 100)}pct"
            test_pairs.append((wiki_id, synthetic_id, modified, expected_j))

    return test_pairs


def evaluate_lsh(candidates, all_true_similar):
    """Compute precision and recall for LSH candidate pairs.

    Args:
        candidates: set of (a, b) pairs found by LSH
        all_true_similar: set of (a, b) pairs with true Jaccard above threshold

    Returns:
        (precision, recall) floats
    """
    if not candidates:
        return 0.0, 0.0
    true_positives = candidates & all_true_similar
    precision = len(true_positives) / len(candidates)
    recall = len(true_positives) / len(all_true_similar) if all_true_similar else 0.0
    return precision, recall
