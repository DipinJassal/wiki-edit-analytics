"""MinHash signature computation — from-scratch and MLlib versions."""

import numpy as np
from src.probabilistic.hash_utils import generate_hash_functions


class MinHash:
    def __init__(self, num_hashes=128, prime=4294967311):
        self.num_hashes = num_hashes
        self.prime = prime
        self.hash_funcs = generate_hash_functions(num_hashes, prime)

    def compute_signature(self, item_set):
        """Compute MinHash signature for a set of integer items.

        Args:
            item_set: iterable of integers (e.g., editor IDs)

        Returns:
            numpy array of shape (num_hashes,)
        """
        signature = np.full(self.num_hashes, np.iinfo(np.int64).max, dtype=np.int64)
        for item in item_set:
            x = int(item)
            for i, (a, b) in enumerate(self.hash_funcs):
                h = (a * x + b) % self.prime
                if h < signature[i]:
                    signature[i] = h
        return signature

    def jaccard_from_signatures(self, sig_a, sig_b):
        """Estimate Jaccard similarity from two MinHash signatures."""
        return float(np.mean(sig_a == sig_b))


def run_minhash_lsh_mllib(spark, editor_sets_df, num_hash_tables=5):
    """Run MinHash LSH using Spark MLlib.

    Args:
        spark: SparkSession
        editor_sets_df: DataFrame with columns (wiki, features) where
                        features is a SparseVector of binary editor presence
        num_hash_tables: number of hash tables

    Returns:
        DataFrame of similar wiki pairs with jaccard_distance column
    """
    from pyspark.ml.feature import MinHashLSH
    from pyspark.sql.functions import col

    mh = (MinHashLSH()
          .setNumHashTables(num_hash_tables)
          .setInputCol("features")
          .setOutputCol("hashes"))

    model = mh.fit(editor_sets_df)

    similar_pairs = (model
                     .approxSimilarityJoin(
                         editor_sets_df, editor_sets_df,
                         threshold=0.8,
                         distCol="jaccard_distance")
                     .filter(col("datasetA.wiki") < col("datasetB.wiki")))

    return similar_pairs


def editor_sets_to_sparse_vectors(spark, editor_sets_df):
    """Convert editor_ids arrays to MLlib SparseVectors.

    Maps global editor_ids to consecutive indices [0, N) required by MLlib.
    """
    from pyspark.sql.functions import explode, col, collect_list, monotonically_increasing_id
    from pyspark.ml.linalg import Vectors, SparseVector
    import pandas as pd

    pandas_df = editor_sets_df.toPandas()

    all_ids = sorted({eid for row in pandas_df['editor_ids'] for eid in row})
    id_to_idx = {eid: i for i, eid in enumerate(all_ids)}
    n = len(all_ids)

    rows = []
    for _, row in pandas_df.iterrows():
        indices = sorted(id_to_idx[eid] for eid in row['editor_ids'])
        vec = Vectors.sparse(n, indices, [1.0] * len(indices))
        rows.append((row['wiki'], vec))

    return spark.createDataFrame(rows, ['wiki', 'features'])
