# Wikipedia Real-Time Edit Stream Analytics

Analyzes editing activity across 5 Wikipedia language editions (English, German, French, Japanese, Spanish) using batch processing, real-time streaming, and probabilistic data structures implemented from scratch.

**Three questions this project answers:**
1. What articles are trending right now? — Spark Structured Streaming + Top-K
2. How many distinct editors does each article have? — Flajolet-Martin Sketch + Bloom Filter
3. Which Wikipedia language editions share the same editors? — MinHash + LSH

---

## Project Structure

```
wiki-edit-analytics/
├── config.yaml                        # All settings (paths, Kafka, algorithm params)
├── docker-compose.yml                 # Kafka + Zookeeper
├── requirements.txt
├── scripts/
│   ├── download_dumps.py              # Download Wikimedia XML dump files
│   └── sse_to_kafka.py               # Live Wikimedia EventStream → Kafka producer
├── src/
│   ├── ingestion/
│   │   ├── xml_to_parquet.py         # Parse XML dumps → Parquet (iterparse + PyArrow)
│   │   ├── build_editor_sets.py      # Build per-wiki/per-article editor ID sets
│   │   └── batch_analysis.py         # Spark batch aggregations on full revision history
│   ├── streaming/
│   │   ├── kafka_consumer.py         # Spark Structured Streaming from Kafka
│   │   └── trending.py               # Top-K trending articles per window
│   ├── probabilistic/
│   │   ├── bloom_filter.py           # Bloom Filter from scratch (MurmurHash3)
│   │   ├── fm_sketch.py              # Flajolet-Martin Sketch from scratch
│   │   ├── hash_utils.py             # MurmurHash3 wrappers + hash function generator
│   │   └── run_evaluation.py         # Generate evaluation data for notebooks
│   ├── similarity/
│   │   ├── minhash.py                # MinHash signatures from scratch + MLlib version
│   │   ├── lsh.py                    # LSH banding from scratch
│   │   ├── validation.py             # Exact Jaccard, synthetic test sets, precision/recall
│   │   └── run_analysis.py           # Full MinHash + LSH pipeline across 5 wikis
│   └── dashboard/
│       └── app.py                    # Streamlit dashboard (4 tabs)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Parquet schema, edit volume, bot vs human breakdown
│   ├── 02_bloom_evaluation.ipynb     # FP rate vs m and k, memory comparison
│   ├── 03_fm_evaluation.ipynb        # MAPE vs k, error distribution, HLL comparison
│   ├── 04_lsh_evaluation.ipynb       # Jaccard heatmap, LSH detection curves, P/R by config
│   └── 05_scaling_experiments.ipynb  # Storage, runtime scaling, throughput benchmarks
└── tests/
    ├── test_bloom_filter.py
    ├── test_fm_sketch.py
    ├── test_minhash.py
    └── test_lsh.py
```

Data files (XML dumps, Parquet, results) are excluded from git — see the Google Drive section below.

---

## Data Sources

### Batch: Wikimedia Database Dumps

Monthly XML stub-meta-history dumps containing every revision ever made, without article text. One part file per wiki (~800 MB–1 GB compressed).

- URL pattern: `https://dumps.wikimedia.org/{wiki}/{date}/{wiki}-{date}-stub-meta-history1.xml.gz`
- Fields extracted: page title, page ID, revision ID, timestamp, contributor ID/name, is_anonymous, text bytes, edit comment

XML parsing uses Python `iterparse` + PyArrow batch writer (50K rows/batch) instead of spark-xml — gzip-compressed XML is unsplittable so Spark would load the entire file into one executor and OOM.

### Streaming: Wikimedia EventStreams

Live SSE stream of every edit happening across all Wikimedia wikis in real time.

- URL: `https://stream.wikimedia.org/v2/stream/recentchange`
- No authentication required, ~6,000–10,000 events/minute globally
- Custom SSE parser using `requests.get(stream=True)` + `iter_lines()` (sseclient-py 1.9.0 has a broken API)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Batch processing | PySpark 3.5, PyArrow |
| Stream processing | Spark Structured Streaming, Kafka/Redpanda |
| Probabilistic structures | From-scratch Python (MurmurHash3, numpy) |
| Similarity | From-scratch MinHash + LSH, Spark MLlib MinHashLSH |
| Dashboard | Streamlit |
| Storage | Parquet (Snappy compression) |

---

## Setup

```bash
pip install -r requirements.txt
```

Start Kafka (or use an existing Redpanda container on port 9092):

```bash
docker-compose up -d
```

---

## Execution Order

```bash
# 1. Download one XML dump per wiki (~800 MB each)
python scripts/download_dumps.py --wikis enwiki dewiki frwiki jawiki eswiki --date 20260201

# 2. Parse XML → Parquet (run once per wiki)
python src/ingestion/xml_to_parquet.py

# 3. Run tests
pytest tests/

# 4. Start live event stream → Kafka (runs in background)
python scripts/sse_to_kafka.py &

# 5. Start Spark Structured Streaming consumer
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    src/streaming/kafka_consumer.py

# 6. Build editor sets for similarity analysis
spark-submit src/ingestion/build_editor_sets.py

# 7. Run Spark batch analysis
spark-submit --driver-memory 8g src/ingestion/batch_analysis.py

# 8. Run probabilistic evaluations (generates data for notebooks)
python src/probabilistic/run_evaluation.py

# 9. Run MinHash + LSH similarity analysis
python src/similarity/run_analysis.py

# 10. Launch dashboard
streamlit run src/dashboard/app.py

# 11. Open notebooks 01–05 for evaluation and plots
```

---

## Probabilistic Data Structures

### Bloom Filter

Implemented from scratch using MurmurHash3 and a bytearray bit array.

- Used in streaming to classify each incoming editor as new vs returning within a window
- Theoretical FP rate formula: `(1 - e^(-kn/m))^k`
- Empirical FP rate matches theory closely (validated in notebook 02)
- At m=1M bits, k=7, n=10K elements: FP rate < 0.001, memory ~125 KB vs ~500 KB for an exact Python set

### Flajolet-Martin Sketch

Implemented from scratch using trailing-zero counting + stochastic averaging in groups of 8.

- Estimates distinct editor counts per article without storing all editor IDs
- MAPE ≈ 42% at k=64, decreases as ~1/√k with more hash functions
- Memory usage: 256 bytes at k=64 vs megabytes for an exact set
- Spark's built-in `approx_count_distinct` (HyperLogLog) achieves ~2% error at higher implementation complexity

### MinHash + LSH

MinHash signatures computed from scratch using universal hashing (`h(x) = (ax + b) mod p`).

- 128 hash functions per wiki editor set
- Cross-wiki Jaccard similarity (exact): 1.7–6.1% — Wikipedia language editions have largely distinct editor communities
- German ↔ French share the most editors (~6%), English ↔ Japanese the least (~1.7%)
- MinHash MAE vs exact Jaccard: < 0.02
- LSH banding: with Jaccard values this low (< 0.06), fine-grained banding (b=128, r=1) is needed for reliable detection

---

## Batch Analysis Results

Run on all 5 wikis combined via Spark:

| Metric | Finding |
|---|---|
| Total revisions | ~16M across 5 wikis |
| Most-edited articles | Dominant articles account for thousands of revisions |
| HyperLogLog vs exact | < 2% error at default RSD |
| Yearly growth | Edit volume peaks 2007–2014, stable since |

---

## Dashboard

Four-tab Streamlit app:

| Tab | Content |
|---|---|
| Trending Articles | Top-20 articles per wiki per streaming window, bar chart with wiki/window selectors |
| FM Sketch Accuracy | Estimate vs exact scatter, MAPE vs k bar chart |
| Bloom Filter | FP rate vs m and k, memory comparison vs exact set |
| LSH Similarity | Jaccard heatmap, detection probability curves, precision/recall by band config |

```bash
streamlit run src/dashboard/app.py
```

---

## Tests

```bash
pytest tests/ -v
```

17 tests covering: Bloom Filter (no false negatives, FP rate), FM Sketch (accuracy, merge), MinHash (identical/disjoint/overlapping sets), LSH (candidate detection, precision/recall).

---

## Data Files (Google Drive)

Large data files are not in this repo. Upload to Google Drive with this structure:

```
wiki-edit-analytics-data/
├── dumps/
│   ├── enwiki/enwiki-20260201-stub-meta-history1.xml.gz
│   ├── dewiki/dewiki-20260201-stub-meta-history1.xml.gz
│   ├── frwiki/frwiki-20260201-stub-meta-history1.xml.gz
│   ├── jawiki/jawiki-20260201-stub-meta-history1.xml.gz
│   └── eswiki/eswiki-20260201-stub-meta-history1.xml.gz
├── parquet/
│   ├── enwiki.parquet
│   ├── dewiki.parquet
│   ├── frwiki.parquet
│   ├── jawiki.parquet
│   └── eswiki.parquet
└── results/
    ├── bloom_evaluation.parquet
    ├── fm_evaluation.parquet
    ├── fm_summary.parquet
    ├── lsh_jaccard_matrix.parquet
    ├── lsh_precision_recall.parquet
    ├── lsh_synthetic_eval.parquet
    ├── batch_wiki_stats/
    ├── batch_top_articles/
    ├── batch_yearly_edits/
    ├── batch_editor_productivity/
    ├── batch_hll_vs_exact/
    ├── editor_sets/
    │   ├── wiki_editor_sets/
    │   └── page_editor_sets/
    └── (all .png plots from notebooks 01–05)
```

Download to `wiki-edit-analytics/data/` to match the paths in `config.yaml`.
