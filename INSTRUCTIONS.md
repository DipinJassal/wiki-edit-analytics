# How to Run the Project

All commands are run from inside the project folder:

```bash
cd /Users/dipinjassal/sem_2/bigdata_project/wiki-edit-analytics
```

Use this Python for everything:

```
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
```

---

## Step 0 — Install dependencies

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/pip3 install -r requirements.txt
```

---

## Step 1 — Start Kafka

Open Docker Desktop first, wait for it to fully start, then:

```bash
docker rm -f zookeeper kafka 2>/dev/null
docker-compose up -d
```

Verify both containers are running:

```bash
docker ps
```

You should see `zookeeper` and `kafka` listed as `Up`.

---

## Step 2 — Download Wikipedia dumps

Skip this step if `data/parquet/` already contains the 5 `.parquet` files.

```bash
mkdir -p data/dumps data/parquet data/results

/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/download_dumps.py \
    --wiki enwiki dewiki frwiki jawiki eswiki \
    --date 20260201 \
    --output data/dumps \
    --parts 1
```

This downloads one part file per wiki (~800 MB each, ~4 GB total). Takes 15--30 minutes depending on connection speed. Files are saved to `data/dumps/{wiki}/`.

---

## Step 3 — Parse XML dumps to Parquet

Skip this step if `data/parquet/` already contains the 5 `.parquet` files.

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 src/ingestion/xml_to_parquet.py
```

Processes each wiki one at a time. Takes about 5--10 minutes per wiki. Writes `data/parquet/enwiki.parquet`, `data/parquet/dewiki.parquet`, etc.

---

## Step 4 — Build editor sets

Skip this step if `data/results/editor_sets/` already exists.

```bash
spark-submit src/ingestion/build_editor_sets.py
```

Reads the Parquet files and groups editor IDs by wiki and by article. Output goes to `data/results/editor_sets/`. Takes about 5 minutes.

---

## Step 5 — Run batch analysis

Skip this step if `data/results/batch_wiki_stats/` already exists.

```bash
spark-submit --driver-memory 8g src/ingestion/batch_analysis.py
```

Runs Spark aggregations on all 101M revisions: wiki stats, top articles, yearly edits, HyperLogLog vs exact count, editor productivity. Takes about 10--15 minutes. Results go to `data/results/`.

---

## Step 6 — Run probabilistic evaluations

Skip this step if `data/results/fm_evaluation.parquet` and `data/results/bloom_evaluation.parquet` already exist.

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 src/probabilistic/run_evaluation.py
```

Evaluates FM Sketch and Bloom Filter against exact counts. Takes about 2--3 minutes. Results go to `data/results/`.

---

## Step 7 — Run MinHash and LSH similarity analysis

Skip this step if `data/results/lsh_jaccard_matrix.parquet` already exists.

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 src/similarity/run_analysis.py
```

Computes MinHash signatures for all 5 wikis, runs LSH banding, evaluates precision/recall, saves Jaccard matrix. Takes about 3--5 minutes. Results go to `data/results/`.

---

## Step 8 — Run tests

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest tests/ -v
```

17 tests covering Bloom Filter, FM Sketch, MinHash, and LSH. All should pass.

---

## Step 9 — Launch the dashboard

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m streamlit run src/dashboard/app.py
```

Open your browser at **http://localhost:8501**

Four tabs:
- **Trending Articles** -- top articles per wiki per streaming window
- **FM Sketch Accuracy** -- estimate vs exact, MAPE at different k values
- **Bloom Filter** -- false positive rate vs m and k, memory comparison
- **LSH Similarity** -- Jaccard heatmap, detection curves, precision/recall

---

## Step 10 — Live streaming (optional)

This runs the real-time pipeline. Requires Kafka from Step 1 to be running.

Open **Terminal 1** -- start the Wikimedia event stream producer:

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/sse_to_kafka.py
```

You will see events flowing like:
```
[enwiki] Barack Obama | user123 | 2026-04-23T16:30:00Z
```

Open **Terminal 2** -- start the Spark streaming consumer (after Terminal 1 shows events):

```bash
spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    src/streaming/kafka_consumer.py
```

The consumer runs 1-hour tumbling windows. After one full hour, the Trending Articles tab in the dashboard will update with live data.

To stop streaming: press `Ctrl+C` in both terminals.

---

## Step 11 — Run Jupyter notebooks

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m jupyter lab notebooks/
```

Open in browser at **http://localhost:8888**

Five notebooks:
| Notebook | What it covers |
|---|---|
| 01_data_exploration | Edit volume, yearly trends, storage comparison |
| 02_bloom_evaluation | FP rate vs m and k, memory comparison |
| 03_fm_evaluation | MAPE vs k, error distribution, per-wiki breakdown |
| 04_lsh_evaluation | Jaccard heatmap, LSH detection curves, precision/recall |
| 05_scaling_experiments | Runtime scaling for MinHash, FM Sketch, Bloom Filter |

The notebooks read pre-computed results from `data/results/` so Steps 4--7 must be completed first.

---

## Current state (what is already done)

If you are running this on the existing machine, everything from Steps 2--7 is already complete. You can go straight to:

- **Step 8** -- run tests
- **Step 9** -- launch dashboard
- **Step 10** -- start live streaming
- **Step 11** -- open notebooks

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: container name already in use` | Run `docker rm -f zookeeper kafka` then `docker-compose up -d` |
| `spark-submit: command not found` | Use the full path: `/Library/Frameworks/Python.framework/Versions/3.13/bin/spark-submit` |
| Dashboard shows no trending data | Streaming must run for at least 1 hour to produce a full window |
| Out of memory during batch analysis | Make sure you use `--driver-memory 8g` with spark-submit |
| Wikimedia stream returns 403 | The SSE script handles this automatically via the User-Agent header |
