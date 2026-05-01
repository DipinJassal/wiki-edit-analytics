"""K-Anonymity implementation for Wikipedia editor data.

Applies k-anonymity to the editor-article activity dataset by suppressing
quasi-identifier combinations that appear in fewer than k records.

Quasi-identifiers used:
    - wiki          (which language edition)
    - year          (year of edit activity)
    - edit_bucket   (editor activity level: low/medium/high/power)

Sensitive attribute:
    - contributor_id (the editor being protected)

A group is k-anonymous if at least k distinct contributor_ids share the
same (wiki, year, edit_bucket) combination. Groups with fewer than k
records are suppressed (dropped) to prevent re-identification.
"""

import os
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, countDistinct, year, when, lit
)


def load_config(path='config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def assign_edit_bucket(edit_count_col):
    """Generalise edit count into 4 ordered buckets (generalisation step)."""
    return (
        when(edit_count_col == 1,          lit('1_single'))
        .when(edit_count_col <= 5,         lit('2_low'))
        .when(edit_count_col <= 20,        lit('3_medium'))
        .when(edit_count_col <= 100,       lit('4_high'))
        .otherwise(                         lit('5_power'))
    )


def apply_k_anonymity(spark, df, k=5):
    """
    Apply k-anonymity to the revision DataFrame.

    Steps:
        1. Compute per-editor per-wiki per-year edit counts.
        2. Generalise edit_count into edit_bucket (generalisation).
        3. Count distinct contributor_ids per (wiki, year, edit_bucket) group.
        4. Suppress groups where distinct editors < k (suppression).
        5. Return the anonymised aggregate (no individual IDs exposed).

    Args:
        spark : SparkSession
        df    : raw revisions DataFrame
        k     : minimum group size (default 5)

    Returns:
        anonymised_df  — k-anonymous aggregate
        suppressed_df  — suppressed groups (for reporting)
        stats          — dict with summary numbers
    """

    # Step 1: per-editor activity per wiki per year
    editor_activity = (
        df.withColumn('year', year(col('timestamp')))
          .groupBy('wiki', 'year', 'contributor_id')
          .agg(count('*').alias('edit_count'))
    )

    # Step 2: generalise edit_count → edit_bucket
    editor_activity = editor_activity.withColumn(
        'edit_bucket', assign_edit_bucket(col('edit_count'))
    )

    # Step 3: group by quasi-identifiers, count distinct editors per group
    grouped = (
        editor_activity
        .groupBy('wiki', 'year', 'edit_bucket')
        .agg(
            countDistinct('contributor_id').alias('distinct_editors'),
            count('*').alias('record_count'),
        )
        .filter(col('year') >= 2010)
        .orderBy('wiki', 'year', 'edit_bucket')
    )

    # Step 4: partition into satisfying vs suppressed
    anonymised = grouped.filter(col('distinct_editors') >= k)
    suppressed = grouped.filter(col('distinct_editors') < k)

    total_groups     = grouped.count()
    anon_groups      = anonymised.count()
    suppressed_groups = suppressed.count()

    suppressed_editors = (
        suppressed.agg({'distinct_editors': 'sum'})
                  .collect()[0][0] or 0
    )
    total_editors = (
        grouped.agg({'distinct_editors': 'sum'})
               .collect()[0][0] or 0
    )

    stats = {
        'k':                   k,
        'total_groups':        total_groups,
        'anonymised_groups':   anon_groups,
        'suppressed_groups':   suppressed_groups,
        'suppression_rate':    round(suppressed_groups / total_groups * 100, 2),
        'editors_suppressed':  int(suppressed_editors),
        'editors_total':       int(total_editors),
        'editor_suppression_pct': round(suppressed_editors / total_editors * 100, 2)
                                  if total_editors else 0,
    }

    return anonymised, suppressed, stats


def main():
    cfg   = load_config()
    spark = (SparkSession.builder
             .appName('WikiKAnonymity')
             .config('spark.driver.memory', '8g')
             .config('spark.sql.shuffle.partitions', '16')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')

    df = spark.read.parquet(cfg['paths']['parquet'])

    print('\n' + '='*60)
    print('K-ANONYMITY ANALYSIS')
    print('='*60)
    print('Quasi-identifiers : wiki, year, edit_bucket')
    print('Sensitive attribute: contributor_id')
    print('Generalisation    : edit_count → 5 ordered buckets')
    print('Suppression       : groups with distinct editors < k\n')

    results_path = cfg['paths']['results']
    os.makedirs(results_path, exist_ok=True)

    for k in [2, 5, 10]:
        print(f'\n--- k = {k} ---')
        anonymised, suppressed, stats = apply_k_anonymity(spark, df, k=k)

        print(f"Total quasi-identifier groups : {stats['total_groups']:,}")
        print(f"Groups satisfying k={k}         : {stats['anonymised_groups']:,}")
        print(f"Groups suppressed              : {stats['suppressed_groups']:,}  "
              f"({stats['suppression_rate']}%)")
        print(f"Editors in suppressed groups   : {stats['editors_suppressed']:,}  "
              f"({stats['editor_suppression_pct']}% of all editors)")

        anonymised.show(20, truncate=False)

        # Save
        out_path = os.path.join(results_path, f'k_anonymity_k{k}')
        anonymised.write.mode('overwrite').parquet(out_path)
        print(f"Saved anonymised result → {out_path}")

    spark.stop()
    print('\nK-anonymity analysis complete.')


if __name__ == '__main__':
    main()
