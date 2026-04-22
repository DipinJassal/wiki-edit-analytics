"""Spark Structured Streaming consumer from Kafka wiki-edits topic."""

import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, from_unixtime, window,
    count, approx_count_distinct, sum as spark_sum, when
)
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
    BooleanType, IntegerType
)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


EVENT_SCHEMA = StructType([
    StructField("wiki", StringType()),
    StructField("title", StringType()),
    StructField("user", StringType()),
    StructField("bot", BooleanType()),
    StructField("timestamp", LongType()),
    StructField("namespace", IntegerType()),
    StructField("length", StructType([
        StructField("old", IntegerType()),
        StructField("new", IntegerType()),
    ])),
    StructField("comment", StringType()),
])


def main():
    cfg = load_config()

    spark = (SparkSession.builder
             .appName("WikiStreamConsumer")
             .config("spark.driver.memory", cfg['spark']['driver_memory'])
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", cfg['kafka']['bootstrap_servers'])
           .option("subscribe", cfg['kafka']['topic'])
           .option("startingOffsets", "latest")
           .load())

    parsed = (raw
              .select(from_json(col("value").cast("string"), EVENT_SCHEMA).alias("d"))
              .select("d.*")
              .withColumn("event_time", from_unixtime(col("timestamp")).cast("timestamp")))

    windowed = (parsed
                .withWatermark("event_time", cfg['streaming']['watermark'])
                .groupBy(
                    window(col("event_time"), cfg['streaming']['window_duration']),
                    col("wiki")
                ).agg(
                    count("*").alias("edit_count"),
                    approx_count_distinct("user").alias("distinct_editors"),
                    spark_sum(when(col("bot") == True, 1).otherwise(0)).alias("bot_edits"),
                ))

    # Per-article trending counts (for Top-K)
    article_counts = (parsed
                      .withWatermark("event_time", cfg['streaming']['watermark'])
                      .groupBy(
                          window(col("event_time"), cfg['streaming']['window_duration']),
                          col("wiki"),
                          col("title")
                      ).agg(count("*").alias("edit_count")))

    results_path = os.path.join(cfg['paths']['results'], "streaming")
    checkpoint_base = "checkpoint"

    # Write windowed wiki stats
    (windowed.writeStream
     .outputMode("append")
     .format("parquet")
     .option("path", os.path.join(results_path, "wiki_stats"))
     .option("checkpointLocation", os.path.join(checkpoint_base, "wiki_stats"))
     .start())

    # Write article counts for trending
    query = (article_counts.writeStream
             .outputMode("append")
             .format("parquet")
             .option("path", os.path.join(results_path, "article_counts"))
             .option("checkpointLocation", os.path.join(checkpoint_base, "article_counts"))
             .start())

    # Also print wiki stats to console for monitoring
    (windowed.writeStream
     .outputMode("complete")
     .format("console")
     .option("truncate", False)
     .option("checkpointLocation", os.path.join(checkpoint_base, "console"))
     .start())

    query.awaitTermination()


if __name__ == '__main__':
    main()
