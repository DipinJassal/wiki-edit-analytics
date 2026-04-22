"""FM Sketch + Bloom Filter evaluation against exact counts from Parquet."""

import os
import sys
import yaml
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.probabilistic.fm_sketch import FMSketch
from src.probabilistic.bloom_filter import BloomFilter


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def load_page_editor_sets(cfg, max_pages=2000):
    path = os.path.join(cfg['paths']['results'], 'editor_sets', 'page_editor_sets')
    df = pq.read_table(path).to_pandas()
    # Keep pages with enough editors for meaningful evaluation
    df = df[df['exact_distinct_editors'] >= 10].head(max_pages)
    print(f"Loaded {len(df):,} pages for evaluation")
    return df


# ── FM Sketch Evaluation ─────────────────────────────────────────────────────

def evaluate_fm(page_df, num_hashes_list=(16, 32, 64, 128, 256)):
    print("\n=== FM Sketch Evaluation ===")
    summary_rows = []
    detail_rows = []

    for num_hashes in num_hashes_list:
        errors = []
        for _, row in page_df.iterrows():
            fm = FMSketch(num_hashes=num_hashes)
            for eid in row['editor_ids']:
                fm.add(eid)
            estimate = fm.estimate()
            exact    = row['exact_distinct_editors']
            err_pct  = abs(estimate - exact) / exact * 100
            errors.append(err_pct)

            if num_hashes == 64:
                detail_rows.append({
                    'wiki':       row['wiki'],
                    'page_title': row['page_title'],
                    'exact':      exact,
                    'estimate':   estimate,
                    'error_pct':  err_pct,
                    'num_hashes': num_hashes,
                })

        mape = sum(errors) / len(errors)
        mem  = FMSketch(num_hashes=num_hashes).__class__.__module__  # just for ref
        summary_rows.append({
            'num_hashes':   num_hashes,
            'MAPE':         mape,
            'max_error':    max(errors),
            'memory_bytes': num_hashes * 4,  # int32 array
        })
        print(f"  num_hashes={num_hashes:3d}  MAPE={mape:6.2f}%  max_err={max(errors):7.2f}%  mem={num_hashes*4}B")

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ── Bloom Filter Evaluation ───────────────────────────────────────────────────

def evaluate_bloom(page_df, sizes=(100_000, 500_000, 1_000_000, 5_000_000),
                   hash_counts=(3, 5, 7, 10)):
    print("\n=== Bloom Filter Evaluation ===")
    rows = []

    # Use all editors from page_df as the "known" set
    all_editors = set()
    for editor_ids in page_df['editor_ids']:
        all_editors.update(editor_ids)
    known = list(all_editors)[:10_000]   # add first 10K
    unknown_start = max(all_editors) + 1  # guaranteed not in filter

    for m in sizes:
        for k in hash_counts:
            bf = BloomFilter(size=m, num_hashes=k)
            for eid in known:
                bf.add(eid)

            # Empirical FP rate: test 5000 items NOT in the filter
            fps = sum(
                1 for i in range(5000)
                if bf.contains(unknown_start + i)
            )
            empirical_fp = fps / 5000
            theoretical_fp = bf.false_positive_rate()

            rows.append({
                'm':             m,
                'k':             k,
                'num_elements':  len(known),
                'theoretical_fp': theoretical_fp,
                'empirical_fp':  empirical_fp,
                'memory_bytes':  bf.memory_bytes(),
            })
            print(f"  m={m:>9,}  k={k:2d}  theoretical={theoretical_fp:.4f}  empirical={empirical_fp:.4f}  mem={bf.memory_bytes():,}B")

    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    out_dir = cfg['paths']['results']
    os.makedirs(out_dir, exist_ok=True)

    page_df = load_page_editor_sets(cfg)

    # FM Sketch
    fm_summary, fm_detail = evaluate_fm(page_df)
    fm_summary.to_parquet(os.path.join(out_dir, 'fm_summary.parquet'), index=False)
    fm_detail.to_parquet(os.path.join(out_dir, 'fm_evaluation.parquet'), index=False)
    print("\nFM Summary:")
    print(fm_summary.to_string(index=False))

    # Bloom Filter
    bloom_df = evaluate_bloom(page_df)
    bloom_df.to_parquet(os.path.join(out_dir, 'bloom_evaluation.parquet'), index=False)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == '__main__':
    main()
