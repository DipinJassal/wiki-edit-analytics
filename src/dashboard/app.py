"""Streamlit dashboard for Wikipedia Edit Stream Analytics."""

import os
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def safe_read_parquet(path, default_cols):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=default_cols)


st.set_page_config(page_title="Wiki Edit Analytics", layout="wide")
st.title("Wikipedia Real-Time Edit Stream Analytics")

cfg = load_config()
results = cfg['paths']['results']

tab1, tab2, tab3, tab4 = st.tabs([
    "Trending Articles",
    "FM Sketch Accuracy",
    "Bloom Filter",
    "LSH Similarity",
])

# ── Tab 1: Trending Articles ──────────────────────────────────────────────────
with tab1:
    st.header("Top Trending Articles")
    trending_path = os.path.join(results, "trending", "top_k_trending.parquet")
    df_trend = safe_read_parquet(trending_path, ["window", "wiki", "title", "edit_count", "rank"])

    if df_trend.empty:
        st.info("No trending data yet. Run kafka_consumer.py and then trending.py.")
    else:
        wikis = sorted(df_trend['wiki'].unique())
        windows = sorted(df_trend['window'].unique(), reverse=True)
        def _fmt_window(w):
            # Already a formatted string from regenerated parquet
            if isinstance(w, str) and '→' in w:
                return w
            try:
                import ast
                d = ast.literal_eval(w) if isinstance(w, str) else w
                start = pd.Timestamp(d['start'], unit='ns')
                end   = pd.Timestamp(d['end'],   unit='ns')
                return f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%H:%M')}"
            except Exception:
                return str(w)
        window_labels = {w: _fmt_window(w) for w in windows}

        col1, col2 = st.columns(2)
        selected_wiki = col1.selectbox("Wiki", wikis)
        selected_window = col2.selectbox("Time Window", windows, format_func=lambda w: window_labels[w])

        selected_label = window_labels[selected_window]
        df_trend['_window_label'] = df_trend['window'].map(lambda w: _fmt_window(w))
        filtered = df_trend[
            (df_trend['wiki'] == selected_wiki) &
            (df_trend['_window_label'] == selected_label)
        ].nsmallest(cfg['streaming']['top_k'], 'rank')

        fig = px.bar(
            filtered, x='edit_count', y='title',
            orientation='h', title=f"Top Articles — {selected_wiki}",
            labels={'edit_count': 'Edits', 'title': 'Article'},
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: FM Sketch Accuracy ─────────────────────────────────────────────────
with tab2:
    st.header("FM Sketch Accuracy vs Exact Count")
    fm_path = os.path.join(results, "fm_evaluation.parquet")
    df_fm = safe_read_parquet(fm_path, ["page_title", "exact", "estimate", "num_hashes", "error_pct"])

    if df_fm.empty:
        st.info("Run notebooks/03_fm_evaluation.ipynb to generate this data.")
    else:
        fig_scatter = px.scatter(
            df_fm, x='exact', y='estimate',
            title="FM Estimate vs Exact Count",
            labels={'exact': 'Exact Distinct Editors', 'estimate': 'FM Estimate'},
            opacity=0.5,
        )
        max_val = max(df_fm['exact'].max(), df_fm['estimate'].max())
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                         mode='lines', name='y=x', line=dict(dash='dash')))
        st.plotly_chart(fig_scatter, use_container_width=True)

        fm_summary_path = os.path.join(results, "fm_summary.parquet")
        df_fm_summary = safe_read_parquet(fm_summary_path, ["num_hashes", "MAPE"])
        if not df_fm_summary.empty:
            fig_mape = px.bar(df_fm_summary, x='num_hashes', y='MAPE',
                              title="MAPE vs Number of Hash Functions",
                              labels={'num_hashes': 'Number of Hash Functions', 'MAPE': 'MAPE (%)'},
                              category_orders={'num_hashes': sorted(df_fm_summary['num_hashes'].unique())})
            st.plotly_chart(fig_mape, use_container_width=True)

        st.dataframe(df_fm[['page_title', 'exact', 'estimate', 'error_pct']].head(50))

# ── Tab 3: Bloom Filter Evaluation ───────────────────────────────────────────
with tab3:
    st.header("Bloom Filter Evaluation")
    bloom_path = os.path.join(results, "bloom_evaluation.parquet")
    df_bloom = safe_read_parquet(bloom_path, ["m", "k", "num_elements", "theoretical_fp", "empirical_fp", "memory_bytes"])

    if df_bloom.empty:
        st.info("Run notebooks/02_bloom_evaluation.ipynb to generate this data.")
    else:
        fig_fp = px.line(
            df_bloom, x='m', y=['theoretical_fp', 'empirical_fp'],
            title="False Positive Rate vs Bit Array Size",
            labels={'value': 'FP Rate', 'm': 'Bit Array Size (m)'},
            log_x=True,
        )
        st.plotly_chart(fig_fp, use_container_width=True)

        fig_mem = px.bar(
            df_bloom.drop_duplicates('m'), x='m', y='memory_bytes',
            title="Bloom Filter Memory vs Bit Array Size",
            labels={'memory_bytes': 'Memory (bytes)', 'm': 'Bit Array Size (m)'},
        )
        st.plotly_chart(fig_mem, use_container_width=True)
        st.dataframe(df_bloom)

# ── Tab 4: LSH Similarity Results ────────────────────────────────────────────
with tab4:
    st.header("Wiki Editor Community Overlap (LSH)")
    lsh_path = os.path.join(results, "lsh_jaccard_matrix.parquet")
    df_lsh = safe_read_parquet(lsh_path, ["wiki_a", "wiki_b", "jaccard"])

    if df_lsh.empty:
        st.info("Run notebooks/04_lsh_evaluation.ipynb to generate this data.")
    else:
        wikis_all = sorted(set(df_lsh['wiki_a']) | set(df_lsh['wiki_b']))
        matrix = pd.DataFrame(1.0, index=wikis_all, columns=wikis_all)
        for _, row in df_lsh.iterrows():
            matrix.loc[row['wiki_a'], row['wiki_b']] = row['jaccard']
            matrix.loc[row['wiki_b'], row['wiki_a']] = row['jaccard']

        fig_heat = px.imshow(
            matrix, text_auto=".3f",
            title="Jaccard Similarity Between Wikipedia Editions",
            color_continuous_scale="Blues",
            zmin=0, zmax=1,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Candidate Pairs")
        st.dataframe(df_lsh.sort_values('jaccard', ascending=False))

        pr_path = os.path.join(results, "lsh_precision_recall.parquet")
        df_pr = safe_read_parquet(pr_path, ["bands", "rows", "precision", "recall"])
        if not df_pr.empty:
            df_pr = df_pr.sort_values('bands')
            df_pr['config'] = df_pr.apply(lambda r: f"b={int(r.bands)},r={int(r.rows)}", axis=1)

            # Melt to long form for grouped bar chart
            df_pr_long = df_pr[['config', 'precision', 'recall']].melt(
                id_vars='config', var_name='Metric', value_name='Score'
            )
            fig_pr = px.bar(
                df_pr_long, x='config', y='Score', color='Metric',
                barmode='group',
                title="LSH Precision & Recall at Different Band/Row Settings",
                labels={'config': 'Band/Row Config', 'Score': 'Score (0–1)'},
                color_discrete_map={'precision': '#4C9BE8', 'recall': '#E8834C'},
                text_auto='.2f',
            )
            fig_pr.update_layout(yaxis_range=[0, 1.1])
            st.plotly_chart(fig_pr, use_container_width=True)

            st.subheader("Band/Row Settings Summary")
            st.dataframe(df_pr[['config', 'num_candidates', 'precision', 'recall']].reset_index(drop=True))
