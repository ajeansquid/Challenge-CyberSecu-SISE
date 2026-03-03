# -*- coding: utf-8 -*-
"""Predictions Page — Supervised, Anomaly Detection, and Clustering inference."""

import traceback

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from app.state import get_state


def render():
    """Render predictions page."""
    state = get_state()

    st.title("Predictions")
    st.caption("Apply a trained model to new data or the current feature set.")

    has_supervised = state.has_trained_model()
    has_anomaly    = state.model_service.has_fitted_anomaly_detector
    has_clustering = state.model_service.has_fitted_clusterer

    if not any([has_supervised, has_anomaly, has_clustering]):
        st.warning(
            "**No trained model available.** "
            "Go to **Model Training** first and train (or load) at least one model.\n\n"
            "| Model type | Where to train |\n"
            "|---|---|\n"
            "| Supervised classifier | Model Training → Supervised tab |\n"
            "| Anomaly detector | Model Training → Unsupervised tab → Anomaly Detection |\n"
            "| Clustering | Model Training → Unsupervised tab → Clustering |"
        )
        return

    # Build tab list dynamically based on what is fitted
    tab_labels = []
    if has_supervised:
        tab_labels.append("🎯 Supervised")
    if has_anomaly:
        tab_labels.append("🚨 Anomaly Detection")
    if has_clustering:
        tab_labels.append("🗂 Clustering")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    if has_supervised:
        with tabs[tab_idx]:
            _render_supervised(state)
        tab_idx += 1

    if has_anomaly:
        with tabs[tab_idx]:
            _render_anomaly(state)
        tab_idx += 1

    if has_clustering:
        with tabs[tab_idx]:
            _render_clustering(state)


# ---------------------------------------------------------------------------
# Data source selector (shared helper)
# ---------------------------------------------------------------------------

def _pick_dataframe(state, key_prefix: str):
    """Return (df, feature_cols) chosen by the user."""
    options = {}
    if state.has_features():
        options["Generated features"] = state.features_data
    if state.has_labeled_data():
        options["Labeled data"] = state.labeled_data

    if not options:
        st.warning("No feature data available. Run Feature Engineering first.")
        return None, []

    source_label = st.selectbox("Data source", list(options.keys()), key=f"{key_prefix}_source")
    df = options[source_label]

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    upload = st.file_uploader(
        "…or upload a CSV / Excel file",
        type=['csv', 'xlsx'],
        key=f"{key_prefix}_upload",
    )
    if upload is not None:
        try:
            if upload.name.endswith('.csv'):
                df = pd.read_csv(upload)
            else:
                df = pd.read_excel(upload)
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            st.success(f"Loaded {len(df):,} rows from uploaded file.")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            return None, []

    feature_cols = st.multiselect(
        "Features to use",
        numeric_cols,
        default=numeric_cols,
        key=f"{key_prefix}_features",
    )
    return df, feature_cols


# ---------------------------------------------------------------------------
# Supervised tab
# ---------------------------------------------------------------------------

def _render_supervised(state):
    st.header("Supervised Predictions")
    pipeline = state.model_service.active_model
    if pipeline is None:
        st.error("Supervised pipeline is not available.")
        return

    st.caption(
        f"Model: **{pipeline.model_key}** | "
        f"Features: {', '.join(pipeline.feature_names) or '—'}"
    )

    inner_tab_single, inner_tab_batch = st.tabs(["Single observation", "Batch"])

    with inner_tab_single:
        _render_single_prediction(state, pipeline)

    with inner_tab_batch:
        _render_supervised_batch(state)


def _render_single_prediction(state, pipeline):
    st.subheader("Enter feature values")
    feature_names = pipeline.feature_names or []
    if not feature_names:
        st.info("No feature names on the loaded model — cannot build the form.")
        return

    ref_df = state.features_data if state.has_features() else (
        state.labeled_data if state.has_labeled_data() else None
    )

    values = {}
    cols_per_row = 3
    rows = [feature_names[i:i + cols_per_row] for i in range(0, len(feature_names), cols_per_row)]
    for row_features in rows:
        cols = st.columns(len(row_features))
        for col, feat in zip(cols, row_features):
            default_val = (
                float(ref_df[feat].median())
                if ref_df is not None and feat in ref_df.columns
                else 0.0
            )
            values[feat] = col.number_input(feat, value=default_val, key=f"single_{feat}")

    if st.button("Predict", type="primary", key="btn_single_predict"):
        try:
            result = state.model_service.predict(pd.DataFrame([values]))
            pred = result.predictions[0]
            proba = float(result.probabilities[0]) if result.probabilities is not None else None
            positive_label = st.session_state.get('sup_positive_label', 'positive')
            if pred == positive_label:
                st.error(f"Prediction: **{pred}** 🚨 THREAT")
            else:
                st.success(f"Prediction: **{pred}** ✅")
            if proba is not None:
                st.metric("Threat probability", f"{proba:.1%}")
                st.progress(proba)
        except Exception as exc:
            st.error(f"Error: {exc}")


def _render_supervised_batch(state):
    st.subheader("Batch predictions")
    df, feature_cols = _pick_dataframe(state, "sup_batch")
    if df is None or not feature_cols:
        return

    if st.button("Run Predictions", type="primary", key="sup_batch_btn"):
        try:
            results = state.model_service.predict_dataframe(df)
            state.predictions = results
            st.success(f"{len(results):,} predictions generated.")
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.code(traceback.format_exc())

    if state.has_predictions():
        _show_supervised_results(
            state.predictions,
            st.session_state.get('sup_positive_label', 'positive'),
        )


def _show_supervised_results(results: pd.DataFrame, positive_label: str):
    pos_count = (results.get('prediction', pd.Series(dtype=str)) == positive_label).sum()
    cols = st.columns(3)
    cols[0].metric("Positive predictions", f"{pos_count:,}")
    cols[1].metric("Total", f"{len(results):,}")
    cols[2].metric("Positive rate", f"{pos_count / len(results) * 100:.1f}%")

    st.subheader("Top 20 results")
    st.dataframe(results.head(20), width='stretch')

    if 'probability' in results.columns:
        st.subheader("Probability distribution")
        fig = px.histogram(results, x='probability', nbins=50)
        st.plotly_chart(fig, width='stretch')

    csv = results.to_csv(index=False)
    st.download_button("Download results (CSV)", csv, "predictions_supervised.csv", "text/csv")


# ---------------------------------------------------------------------------
# Anomaly detection tab
# ---------------------------------------------------------------------------

def _render_anomaly(state):
    st.header("Anomaly Detection Predictions")
    det = state.model_service._anomaly_detector
    st.caption(f"Model: **{det.name}**")

    # Cache key tied to the model so stale results from a different model are dropped
    _cache_key = f"_anom_results_{det.name}"

    df, feature_cols = _pick_dataframe(state, "anom")
    if df is None or not feature_cols:
        return

    if st.button("Score for Anomalies", type="primary", key="anom_run_btn"):
        # Clear stale SHAP values whenever we rescore
        st.session_state.pop('_shap_values', None)
        try:
            result_df = state.model_service.apply_anomaly_detector(df, feature_cols)
            st.session_state[_cache_key] = result_df
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.code(traceback.format_exc())

    result_df = st.session_state.get(_cache_key)
    if result_df is not None:
        _show_anomaly_results(result_df, feature_cols, state)


def _show_anomaly_results(result_df: pd.DataFrame, feature_cols: list, state=None):
    n_total = len(result_df)
    n_anom = int(result_df['is_anomaly'].sum())

    cols = st.columns(4)
    cols[0].metric("Total IPs", f"{n_total:,}")
    cols[1].metric("Anomalies", f"{n_anom:,}")
    cols[2].metric("Normal", f"{n_total - n_anom:,}")
    cols[3].metric("Anomaly rate", f"{n_anom / n_total * 100:.1f}%")

    st.subheader("Anomaly score distribution")
    fig = px.histogram(
        result_df, x='anomaly_score', color='is_anomaly',
        nbins=50,
        color_discrete_map={True: '#e74c3c', False: '#2ecc71'},
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Top anomalous rows")
    anomalies = result_df[result_df['is_anomaly']].head(20)
    display_cols = [c for c in feature_cols + ['anomaly_score', 'is_anomaly'] if c in anomalies.columns]
    st.dataframe(anomalies[display_cols], width='stretch')

    # ── SHAP feature importance ──────────────────────────────────────────────
    if state is not None:
        st.subheader("Feature Importance (SHAP)")
        st.caption(
            "SHAP values show which features push the anomaly score lower (more anomalous). "
            "Positive SHAP = feature increased anomaly score. "
            "Negative SHAP = feature drove the IP toward anomalous."
        )
        if st.button("Compute SHAP values", key="shap_compute_btn"):
            with st.spinner("Computing SHAP values… (may take a moment for KernelExplainer)"):
                shap_result = state.model_service.compute_shap_values(result_df, feature_cols)
            # shap_result is either (shap_arr, df, cols, explainer) or (None, error_msg)
            if shap_result[0] is None:
                st.error(f"SHAP computation failed: {shap_result[1]}")
            else:
                shap_values, shap_df, shap_cols, explainer_used = shap_result
                st.session_state['_shap_values'] = (shap_values, shap_df, shap_cols)
                st.caption(f"Explainer: {explainer_used}")

        if '_shap_values' in st.session_state:
            shap_values, shap_df, shap_cols = st.session_state['_shap_values']
            # shap_values already normalized to 2-D ndarray by model_service
            shap_arr = np.asarray(shap_values)
            if shap_arr.ndim == 3:  # safety: (outputs, rows, features) — take first output
                shap_arr = shap_arr[0]
            # Mean absolute SHAP per feature
            mean_abs = np.abs(shap_arr).mean(axis=0)
            shap_importance = pd.DataFrame({
                'feature': shap_cols,
                'mean_|shap|': mean_abs,
            }).sort_values('mean_|shap|', ascending=False)

            fig = px.bar(
                shap_importance,
                x='mean_|shap|', y='feature',
                orientation='h',
                title='Mean |SHAP| per Feature (higher = more influential)',
                color='mean_|shap|',
                color_continuous_scale='Reds',
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, width='stretch')

            # SHAP beeswarm-style scatter (top anomalies)
            st.markdown("**SHAP values for top anomalies** (anomaly score ← which feature drove it)")
            n_show = min(50, len(shap_df))
            shap_long = []
            for i, row in shap_df.head(n_show).iterrows():
                for j, feat in enumerate(shap_cols):
                    shap_long.append({
                        'IP': str(i),
                        'feature': feat,
                        'shap_value': float(shap_arr[shap_df.index.get_loc(i), j]),
                        'feature_value': float(row[feat]),
                    })
            shap_long_df = pd.DataFrame(shap_long)
            fig2 = px.scatter(
                shap_long_df, x='shap_value', y='feature',
                color='feature_value',
                color_continuous_scale='RdBu',
                hover_data=['IP', 'feature_value'],
                title=f'SHAP values — top {n_show} anomalies',
            )
            fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig2, width='stretch')

    csv = result_df.to_csv(index=False)
    st.download_button("Download results (CSV)", csv, "predictions_anomaly.csv", "text/csv")


# ---------------------------------------------------------------------------
# Clustering tab
# ---------------------------------------------------------------------------

def _render_clustering(state):
    st.header("Clustering Predictions")
    cl = state.model_service._clusterer
    st.caption(f"Model: **{cl.name}** | Clusters: **{cl.n_clusters}**")

    df, feature_cols = _pick_dataframe(state, "clust")
    if df is None or not feature_cols:
        return

    if st.button("Assign Clusters", type="primary", key="clust_run_btn"):
        try:
            result_df = state.model_service.apply_clusterer(df, feature_cols)
            st.session_state['_clust_results'] = result_df
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.code(traceback.format_exc())

    result_df = st.session_state.get('_clust_results')
    if result_df is not None:
        _show_clustering_results(result_df, feature_cols)


def _show_clustering_results(result_df: pd.DataFrame, feature_cols: list):
    cluster_counts = result_df['cluster'].value_counts().sort_index()

    cols = st.columns(min(5, len(cluster_counts)))
    for i, (cluster_id, count) in enumerate(cluster_counts.items()):
        if i < 5:
            cols[i].metric(f"Cluster {cluster_id}", f"{count:,}")

    if len(feature_cols) >= 2:
        st.subheader("Cluster scatter")
        fig = px.scatter(
            result_df, x=feature_cols[0], y=feature_cols[1],
            color=result_df['cluster'].astype(str),
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("Cluster statistics (mean per cluster)")
    valid_cols = [c for c in feature_cols if c in result_df.columns]
    stats = result_df.groupby('cluster')[valid_cols].mean()
    st.dataframe(stats, width='stretch')

    csv = result_df.to_csv(index=False)
    st.download_button("Download results (CSV)", csv, "predictions_clusters.csv", "text/csv")


