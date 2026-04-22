"""MinHash + LSH similarity analysis on Wikipedia editor sets.

Reads wiki_editor_sets parquet, computes MinHash signatures,
runs LSH banding, evaluates precision/recall, saves results.
"""

import os
import sys
import time
import yaml
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.similarity.minhash import MinHash
from src.similarity.lsh import LSH
from src.similarity.validation import exact_jaccard, create_synthetic_test_set, evaluate_lsh


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_wiki_editor_sets(cfg):
    path = os.path.join(cfg['paths']['results'], 'editor_sets', 'wiki_editor_sets')
    df = pq.read_table(path).to_pandas()
    return {row['wiki']: set(row['editor_ids']) for _, row in df.iterrows()}


def compute_signatures(editor_sets, num_hashes=128):
    mh = MinHash(num_hashes=num_hashes)
    print(f"Computing MinHash signatures (num_hashes={num_hashes}) ...")
    sigs = {}
    for wiki, editors in editor_sets.items():
        t0 = time.time()
        sigs[wiki] = mh.compute_signature(editors)
        print(f"  {wiki}: {len(editors):,} editors  →  {time.time()-t0:.1f}s")
    return mh, sigs


def run_lsh(sigs, num_bands, rows_per_band):
    lsh = LSH(num_bands=num_bands, rows_per_band=rows_per_band)
    for wiki, sig in sigs.items():
        lsh.index(wiki, sig)
    return lsh.get_candidates()


def build_jaccard_matrix(editor_sets, mh, sigs):
    wikis = sorted(editor_sets.keys())
    rows = []
    for i in range(len(wikis)):
        for j in range(i + 1, len(wikis)):
            a, b = wikis[i], wikis[j]
            exact  = exact_jaccard(editor_sets[a], editor_sets[b])
            approx = mh.jaccard_from_signatures(sigs[a], sigs[b])
            rows.append({'wiki_a': a, 'wiki_b': b, 'jaccard': exact, 'minhash_approx': approx})
            print(f"  {a} ↔ {b}:  exact={exact:.4f}  minhash={approx:.4f}")
    return pd.DataFrame(rows)


def run_band_sweep(editor_sets, sigs, band_configs, jaccard_threshold=0.05):
    """Evaluate LSH precision/recall across different (bands, rows) settings."""
    # Build ground truth: all pairs with exact Jaccard >= threshold
    wikis = sorted(editor_sets.keys())
    all_pairs = {
        (a, b)
        for i, a in enumerate(wikis)
        for b in wikis[i+1:]
    }
    true_similar = {
        (a, b) for a, b in all_pairs
        if exact_jaccard(editor_sets[a], editor_sets[b]) >= jaccard_threshold
    }
    print(f"\nGround truth pairs with Jaccard >= {jaccard_threshold}: {len(true_similar)}")

    results = []
    for bands, rows in band_configs:
        candidates = run_lsh(sigs, bands, rows)
        precision, recall = evaluate_lsh(candidates, true_similar)
        results.append({
            'bands': bands, 'rows': rows,
            'num_candidates': len(candidates),
            'precision': precision, 'recall': recall,
        })
        print(f"  b={bands:3d} r={rows}: candidates={len(candidates)}  P={precision:.3f}  R={recall:.3f}")

    return pd.DataFrame(results)


def run_synthetic_eval(editor_sets, num_hashes=128):
    """Evaluate on synthetic perturbed sets to measure MinHash accuracy."""
    print("\nGenerating synthetic perturbed sets ...")
    test_pairs = create_synthetic_test_set(editor_sets, perturbation_levels=[0.1, 0.2, 0.3, 0.5])

    mh = MinHash(num_hashes=num_hashes)
    all_sets = dict(editor_sets)
    for orig_id, syn_id, syn_set, _ in test_pairs:
        all_sets[syn_id] = syn_set

    print(f"Computing signatures for {len(all_sets)} sets ...")
    sigs = {wiki: mh.compute_signature(s) for wiki, s in all_sets.items()}

    rows = []
    for orig_id, syn_id, syn_set, expected_j in test_pairs:
        estimated_j = mh.jaccard_from_signatures(sigs[orig_id], sigs[syn_id])
        rows.append({
            'original': orig_id, 'synthetic': syn_id,
            'expected_jaccard': expected_j,
            'minhash_jaccard': estimated_j,
            'error': abs(estimated_j - expected_j),
        })

    df = pd.DataFrame(rows)
    mae = df['error'].mean()
    print(f"Mean Absolute Error across synthetic pairs: {mae:.4f}")
    return df


def main():
    cfg = load_config()
    out_dir = cfg['paths']['results']
    os.makedirs(out_dir, exist_ok=True)

    print("=== Loading editor sets ===")
    editor_sets = load_wiki_editor_sets(cfg)
    for wiki, s in editor_sets.items():
        print(f"  {wiki}: {len(s):,} distinct editors")

    print("\n=== MinHash Signatures ===")
    mh, sigs = compute_signatures(editor_sets, num_hashes=cfg['minhash']['num_hash_functions'])

    print("\n=== Jaccard Matrix (exact vs MinHash) ===")
    jaccard_df = build_jaccard_matrix(editor_sets, mh, sigs)
    jaccard_df.to_parquet(os.path.join(out_dir, 'lsh_jaccard_matrix.parquet'), index=False)
    print(jaccard_df.to_string(index=False))

    print("\n=== LSH Band Sweep ===")
    band_configs = [(16, 8), (32, 4), (64, 2), (128, 1)]
    pr_df = run_band_sweep(editor_sets, sigs, band_configs)
    pr_df.to_parquet(os.path.join(out_dir, 'lsh_precision_recall.parquet'), index=False)

    print("\n=== Synthetic Set Evaluation ===")
    syn_df = run_synthetic_eval(editor_sets, num_hashes=cfg['minhash']['num_hash_functions'])
    syn_df.to_parquet(os.path.join(out_dir, 'lsh_synthetic_eval.parquet'), index=False)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()
