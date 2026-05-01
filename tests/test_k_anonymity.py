"""Tests for k-anonymity implementation."""

import pytest
from src.privacy.k_anonymity import assign_edit_bucket


def test_edit_bucket_single():
    """Edit count of 1 maps to single bucket."""
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import lit
    spark = SparkSession.builder.master("local").appName("test").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.createDataFrame([(1,),(3,),(10,),(50,),(200,)], ["edit_count"])
    result = df.withColumn("bucket", assign_edit_bucket(df["edit_count"]))
    buckets = [r["bucket"] for r in result.collect()]
    assert buckets[0] == "1_single"
    assert buckets[1] == "2_low"
    assert buckets[2] == "3_medium"
    assert buckets[3] == "4_high"
    assert buckets[4] == "5_power"
    spark.stop()


def test_k_anonymity_suppression():
    """Groups with fewer than k distinct editors must be suppressed."""
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
    from datetime import datetime
    from src.privacy.k_anonymity import apply_k_anonymity

    spark = SparkSession.builder.master("local").appName("test_kanon").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    ts = datetime(2020, 6, 1)
    # 3 editors with 1 edit each — should be suppressed at k=5, kept at k=2
    rows = [(i, "enwiki", 1, f"editor_{i}", f"editor_{i}", False, "", 100, ts)
            for i in range(3)]
    schema = StructType([
        StructField("page_id",          LongType()),
        StructField("wiki",             StringType()),
        StructField("page_title",       StringType()),
        StructField("contributor_id",   StringType()),
        StructField("contributor_name", StringType()),
        StructField("is_anonymous",     StringType()),
        StructField("comment",          StringType()),
        StructField("text_bytes",       LongType()),
        StructField("timestamp",        TimestampType()),
    ])
    df = spark.createDataFrame(rows, schema)

    anon_k5, supp_k5, stats_k5 = apply_k_anonymity(spark, df, k=5)
    assert anon_k5.count() == 0, "All groups should be suppressed at k=5"
    assert stats_k5['suppressed_groups'] > 0

    anon_k2, supp_k2, stats_k2 = apply_k_anonymity(spark, df, k=2)
    assert anon_k2.count() > 0, "Groups should survive at k=2"

    spark.stop()


def test_k_anonymity_no_individual_ids():
    """Output must not contain contributor_id column."""
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
    from datetime import datetime
    from src.privacy.k_anonymity import apply_k_anonymity

    spark = SparkSession.builder.master("local").appName("test_noid").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    ts = datetime(2021, 3, 15)
    rows = [(i, "dewiki", 1, f"editor_{i}", f"editor_{i}", False, "", 200, ts)
            for i in range(10)]
    schema = StructType([
        StructField("page_id",          LongType()),
        StructField("wiki",             StringType()),
        StructField("page_title",       StringType()),
        StructField("contributor_id",   StringType()),
        StructField("contributor_name", StringType()),
        StructField("is_anonymous",     StringType()),
        StructField("comment",          StringType()),
        StructField("text_bytes",       LongType()),
        StructField("timestamp",        TimestampType()),
    ])
    df = spark.createDataFrame(rows, schema)

    anon, _, _ = apply_k_anonymity(spark, df, k=2)
    assert "contributor_id" not in anon.columns, \
        "Anonymised output must not expose contributor_id"
    assert "contributor_name" not in anon.columns, \
        "Anonymised output must not expose contributor_name"

    spark.stop()
