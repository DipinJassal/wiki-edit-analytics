"""Spark batch analysis on Parquet data — demonstrates Spark at scale.

Runs aggregations, window functions, and approx_count_distinct
on the full Wikipedia revision history across all 5 wikis.
"""

import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, countDistinct, approx_count_distinct,
    sum as spark_sum, avg, when, year, month,
    rank, desc, row_number
)
from pyspark.sql.window import Window


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def get_spark(cfg):
    return (SparkSession.builder
            .appName("WikiBatchAnalysis")
            .config("spark.driver.memory", "8g")
            .config("spark.driver.maxResultSize", "4g")
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.sql.files.maxPartitionBytes", "134217728")
            .getOrCreate())


def main():
    cfg = load_config()
    spark = get_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(cfg['paths']['parquet'])

    print(f"\nTotal revisions: {df.count():,}")
    print(f"Schema:")
    df.printSchema()

    # ── 1. Edit and editor counts per wiki ───────────────────────────────────
    print("\n=== Edit + Editor Counts per Wiki ===")
    wiki_stats = (df.groupBy("wiki")
                  .agg(
                      count("*").alias("total_revisions"),
                      countDistinct("contributor_id").alias("distinct_editors"),
                      approx_count_distinct("contributor_id").alias("approx_editors_hll"),
                      spark_sum(when(col("text_bytes") > 0, col("text_bytes"))).alias("total_bytes"),
                      avg("text_bytes").alias("avg_bytes_per_edit"),
                  )
                  .orderBy(desc("total_revisions")))
    wiki_stats.show(truncate=False)
    wiki_stats.write.mode("overwrite").parquet(
        os.path.join(cfg['paths']['results'], "batch_wiki_stats")
    )

    # ── 2. Top 20 most-edited articles per wiki ───────────────────────────────
    print("\n=== Top 20 Most-Edited Articles per Wiki ===")
    w = Window.partitionBy("wiki").orderBy(desc("edit_count"))
    top_articles = (df.groupBy("wiki", "page_title")
                    .agg(
                        count("*").alias("edit_count"),
                        countDistinct("contributor_id").alias("distinct_editors"),
                    )
                    .withColumn("rank", row_number().over(w))
                    .filter(col("rank") <= 20)
                    .orderBy("wiki", "rank"))
    top_articles.show(40, truncate=40)
    top_articles.write.mode("overwrite").parquet(
        os.path.join(cfg['paths']['results'], "batch_top_articles")
    )

    # ── 3. Yearly edit volume per wiki ────────────────────────────────────────
    print("\n=== Yearly Edit Volume ===")
    yearly = (df.withColumn("year", year(col("timestamp")))
              .groupBy("wiki", "year")
              .agg(count("*").alias("revisions"))
              .filter(col("year") >= 2010)
              .orderBy("wiki", "year"))
    yearly.show(60, truncate=False)
    yearly.write.mode("overwrite").parquet(
        os.path.join(cfg['paths']['results'], "batch_yearly_edits")
    )

    # ── 4. Editor productivity distribution (Pareto analysis) ────────────────
    print("\n=== Editor Productivity (edits per editor, per wiki) ===")
    editor_productivity = (df.groupBy("wiki", "contributor_id")
                           .agg(count("*").alias("edits"))
                           .groupBy("wiki")
                           .agg(
                               count("*").alias("total_editors"),
                               avg("edits").alias("avg_edits_per_editor"),
                           ))
    editor_productivity.show(truncate=False)
    editor_productivity.write.mode("overwrite").parquet(
        os.path.join(cfg['paths']['results'], "batch_editor_productivity")
    )

    # ── 5. HyperLogLog vs exact distinct count accuracy ───────────────────────
    print("\n=== HyperLogLog vs Exact Distinct Editors (sample of articles) ===")
    hll_vs_exact = (df.groupBy("wiki", "page_title")
                    .agg(
                        countDistinct("contributor_id").alias("exact"),
                        approx_count_distinct("contributor_id", 0.05).alias("hll_rsd005"),
                        approx_count_distinct("contributor_id", 0.1).alias("hll_rsd010"),
                    )
                    .filter(col("exact") >= 50)
                    .withColumn("err_5pct", (col("hll_rsd005") - col("exact")) / col("exact") * 100)
                    .withColumn("err_10pct", (col("hll_rsd010") - col("exact")) / col("exact") * 100)
                    .orderBy(desc("exact"))
                    .limit(100))
    hll_vs_exact.show(20, truncate=30)
    hll_vs_exact.write.mode("overwrite").parquet(
        os.path.join(cfg['paths']['results'], "batch_hll_vs_exact")
    )

    print("\nAll batch analysis results saved to data/results/")
    spark.stop()


if __name__ == '__main__':
    main()
