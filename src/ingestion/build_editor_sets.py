"""Build per-wiki and per-page editor sets from Parquet data for MinHash / FM evaluation."""

import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, countDistinct


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_spark(cfg):
    return (SparkSession.builder
            .appName("BuildEditorSets")
            .config("spark.driver.memory", cfg['spark']['driver_memory'])
            .config("spark.executor.memory", cfg['spark']['executor_memory'])
            .getOrCreate())


def build_wiki_editor_sets(spark, cfg):
    parquet_dir = cfg['paths']['parquet']
    out_dir = os.path.join(cfg['paths']['results'], 'editor_sets')
    os.makedirs(out_dir, exist_ok=True)

    df = spark.read.parquet(parquet_dir)

    # Per-wiki editor sets
    wiki_sets = (df
                 .groupBy("wiki")
                 .agg(collect_set("contributor_id").alias("editor_ids")))

    wiki_sets.coalesce(1).write.mode("overwrite").parquet(
        os.path.join(out_dir, "wiki_editor_sets")
    )
    print("Saved wiki editor sets.")

    # Per-page editor sets (for FM sketch evaluation), keep pages with >= 10 edits
    page_sets = (df
                 .groupBy("wiki", "page_id", "page_title")
                 .agg(
                     collect_set("contributor_id").alias("editor_ids"),
                     countDistinct("contributor_id").alias("exact_distinct_editors"),
                 )
                 .filter(col("exact_distinct_editors") >= 10))

    page_sets.coalesce(16).write.mode("overwrite").parquet(
        os.path.join(out_dir, "page_editor_sets")
    )
    print("Saved page editor sets.")


def main():
    cfg = load_config()
    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")
    build_wiki_editor_sets(spark, cfg)
    spark.stop()


if __name__ == '__main__':
    main()
