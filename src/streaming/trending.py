"""Top-K trending articles per wiki per window from streaming results."""

import os
import yaml
import pandas as pd


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_top_k(article_counts_df, k=20):
    """Return top-K articles per wiki per window.

    Args:
        article_counts_df: pandas DataFrame with columns window, wiki, title, edit_count
        k: number of top articles to return

    Returns:
        pandas DataFrame ranked by edit_count within each (window, wiki)
    """
    ranked = (article_counts_df
              .sort_values('edit_count', ascending=False)
              .groupby(['window', 'wiki'])
              .head(k)
              .reset_index(drop=True))
    ranked['rank'] = (ranked
                      .groupby(['window', 'wiki'])['edit_count']
                      .rank(method='first', ascending=False)
                      .astype(int))
    return ranked


def save_trending(cfg=None):
    if cfg is None:
        cfg = load_config()

    article_path = os.path.join(cfg['paths']['results'], "streaming", "article_counts")
    out_path = os.path.join(cfg['paths']['results'], "trending")
    os.makedirs(out_path, exist_ok=True)

    if not os.path.exists(article_path):
        print(f"No article counts found at {article_path}. Run kafka_consumer.py first.")
        return

    df = pd.read_parquet(article_path)
    df['window'] = df['window'].astype(str)

    top_k = compute_top_k(df, k=cfg['streaming']['top_k'])
    out_file = os.path.join(out_path, "top_k_trending.parquet")
    top_k.to_parquet(out_file, index=False)
    print(f"Saved Top-{cfg['streaming']['top_k']} trending results to {out_file}")
    return top_k


if __name__ == '__main__':
    save_trending()
