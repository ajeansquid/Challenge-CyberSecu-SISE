# -*- coding: utf-8 -*-
"""Analysis Dashboard – read-only view of computed state (features, models, predictions)."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from app.state import get_state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render():
    """Render the analysis dashboard."""
    state = get_state()

    st.title("Analysis Dashboard")
    st.markdown(
        "Read-only summary of everything computed so far. "
        "Run **Feature Engineering** and **Model Training** first to populate this page."
    )

    # Resolve best available feature DataFrame (attach predictions if present)
    df = _resolve_df(state)

    tabs = st.tabs(["Overview", "Feature Distributions", "Unsupervised Results", "Supervised Results"])

    with tabs[0]:
        render_overview(state, df)

    with tabs[1]:
        if df is not None:
            render_distributions(df)
        else:
            st.info("No feature data available yet.")

    with tabs[2]:
        render_unsupervised_results(state)

    with tabs[3]:
        render_supervised_results(state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_df(state):
    """Return best available feature DataFrame with predictions merged in."""
    df = None
    if state.has_labeled_data():
        df = state.labeled_data.copy()
    elif state.has_features():
        df = state.features_data.copy()

    if df is not None and state.has_predictions():
        preds = state.predictions
        if 'prediction' in preds.columns:
            df['prediction'] = preds['prediction'].reindex(df.index)

    return df


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def render_overview(state, df):
    """Pipeline status + high-level metrics."""
    st.header("Pipeline Status")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw rows loaded", f"{len(state.raw_data):,}" if state.has_raw_data() else "—")
    col2.metric("IPs / feature rows", f"{len(df):,}" if df is not None else "—")

    unsup = state.unsupervised_results
    if unsup:
        result_df = unsup.get('result_df')
        if unsup['type'] == 'anomaly' and result_df is not None:
            n = result_df['is_anomaly'].sum()
            col3.metric("Anomalies detected", f"{n:,}")
        elif unsup['type'] == 'clustering' and result_df is not None:
            col3.metric("Clusters", result_df['cluster'].nunique())
    else:
        col3.metric("Unsupervised model", "—")

    tr = state.training_results
    if tr and tr.get('type') == 'supervised':
        cv = tr.get('cv', {})
        scorer = cv.get('scoring', 'score')
        col4.metric(f"CV {scorer}", f"{cv.get('mean', 0):.3f} ± {cv.get('std', 0):.3f}")
    else:
        col4.metric("Supervised model", "—")

    if df is not None:
        st.markdown("---")
        st.header("Feature Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Features", len(numeric_cols))
        if 'total_flows' in df.columns:
            c2.metric("Total flows", f"{df['total_flows'].sum():,}")
        if 'deny' in df.columns:
            c3.metric("Total denies", f"{df['deny'].sum():,}")
        if 'prediction' in df.columns:
            pos = (df['prediction'] == 'positive').sum()
            c4.metric("Predicted threats", f"{pos:,}")

        with st.expander("Feature statistics"):
            st.dataframe(df.describe(), width='stretch')


# ---------------------------------------------------------------------------
# Tab: Feature Distributions
# ---------------------------------------------------------------------------

@st.fragment
def render_distributions(df):
    """Histograms and box plots, optionally split by prediction/risk label."""
    st.header("Feature Distributions")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    feature = st.selectbox("Feature", numeric_cols, key='dash_dist_feat')

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=feature, nbins=50, title=f'{feature} — histogram')
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig = px.box(df, y=feature, title=f'{feature} — box plot')
        st.plotly_chart(fig, width='stretch')

    # Split by label if available
    class_col = next((c for c in ('prediction', 'risk') if c in df.columns), None)
    if class_col:
        st.subheader(f"By {class_col}")
        fig = px.histogram(
            df, x=feature, color=class_col, barmode='overlay',
            title=f'{feature} by {class_col}',
            color_discrete_map={'positive': '#e74c3c', 'negative': '#2ecc71',
                                 'anomaly': '#e74c3c', 'normal': '#2ecc71'}
        )
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Tab: Unsupervised Results
# ---------------------------------------------------------------------------

def render_unsupervised_results(state):
    """Read-only view of state.unsupervised_results."""
    st.header("Unsupervised Results")

    unsup = state.unsupervised_results
    if not unsup:
        st.info("No unsupervised model trained yet. Go to **Model Training → Unsupervised** tab.")
        return

    model_key  = unsup.get('model_key', '—')
    features   = unsup.get('features', [])
    result_df  = unsup.get('result_df')
    model_type = unsup.get('type')

    st.caption(f"Model: `{model_key}` · Features: {', '.join(features)}")

    if result_df is None:
        st.warning("Result data not found in state.")
        return

    if model_type == 'anomaly':
        _render_anomaly_results(result_df, features)
    elif model_type == 'clustering':
        _render_clustering_results(result_df, features, unsup.get('n_clusters', '?'))
    else:
        st.warning(f"Unknown result type: {model_type}")


@st.fragment
def _render_anomaly_results(result_df, features):
    n_anomalies = result_df['is_anomaly'].sum()
    n_total = len(result_df)

    cols = st.columns(4)
    cols[0].metric("Total IPs", f"{n_total:,}")
    cols[1].metric("Anomalies", f"{n_anomalies:,}")
    cols[2].metric("Normal", f"{n_total - n_anomalies:,}")
    cols[3].metric("Anomaly rate", f"{n_anomalies / n_total * 100:.1f}%")

    fig = px.histogram(
        result_df, x='anomaly_score', color='is_anomaly', nbins=50,
        title='Anomaly Score Distribution',
        color_discrete_map={True: '#e74c3c', False: '#2ecc71'}
    )
    st.plotly_chart(fig, width='stretch')

    if len(features) >= 2:
        x, y = features[0], features[1]
        fig = px.scatter(
            result_df.reset_index(), x=x, y=y,
            color=result_df['is_anomaly'].map({True: 'Anomaly', False: 'Normal'}),
            title=f'Anomalies — {x} vs {y}',
            color_discrete_map={'Anomaly': '#e74c3c', 'Normal': '#2ecc71'},
            opacity=0.6
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("Top anomalous IPs")
    anomalies = result_df[result_df['is_anomaly']].sort_values('anomaly_score').head(50)
    st.dataframe(anomalies[features + ['anomaly_score']], hide_index=True, width='stretch')


@st.fragment
def _render_clustering_results(result_df, features, n_clusters):
    cluster_counts = result_df['cluster'].value_counts().sort_index()
    cols = st.columns(min(len(cluster_counts), 6))
    for i, (cluster, count) in enumerate(cluster_counts.items()):
        if i < 6:
            cols[i].metric(f"Cluster {cluster}", f"{count:,}")

    if len(features) >= 2:
        x, y = features[0], features[1]
        fig = px.scatter(
            result_df.reset_index(), x=x, y=y,
            color=result_df['cluster'].astype(str),
            title=f'Clusters — {x} vs {y}',
            color_discrete_sequence=px.colors.qualitative.Set1,
            opacity=0.7
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("Cluster statistics (mean features)")
    cluster_stats = result_df.groupby('cluster')[features].mean()
    st.dataframe(cluster_stats, width='stretch')

    if len(features) >= 3:
        st.subheader("Cluster profiles")
        normalized = cluster_stats.copy()
        for col in features:
            max_val = normalized[col].max()
            if max_val > 0:
                normalized[col] /= max_val / 100

        fig = go.Figure()
        for cid in normalized.index:
            fig.add_trace(go.Scatterpolar(
                r=[normalized.loc[cid, f] for f in features],
                theta=features, fill='toself', name=f'Cluster {cid}'
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Normalized Cluster Profiles"
        )
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Tab: Supervised Results
# ---------------------------------------------------------------------------

def render_supervised_results(state):
    """Read-only view of state.training_results."""
    st.header("Supervised Results")

    tr = state.training_results
    if not tr or tr.get('type') != 'supervised':
        st.info("No supervised model trained yet. Go to **Model Training → Supervised** tab.")
        return

    cv = tr.get('cv', {})

    scorer = cv.get('scoring', 'score')
    cols = st.columns(4)
    cols[0].metric(f"CV {scorer} (mean)", f"{cv.get('mean', 0):.4f}")
    cols[1].metric("CV std", f"{cv.get('std', 0):.4f}")
    cols[2].metric("CV method", cv.get('cv_method', '—'))
    cols[3].metric("Features used", len(cv.get('features_used', [])))

    with st.expander("Per-fold scores"):
        for i, s in enumerate(cv.get('scores', []), 1):
            st.write(f"Fold {i}: {s:.4f}")

    # Feature importance from active pipeline (if model still in memory)
    try:
        importance = state.model_service.get_feature_importance()
        if importance is not None and not importance.empty:
            st.subheader("Feature Importance")
            fig = px.bar(
                importance.head(15),
                x='importance', y='feature', orientation='h',
                title='Top 15 Features by Importance'
            )
            st.plotly_chart(fig, width='stretch')
    except Exception:
        pass

    # Predictions summary if available
    if state.has_predictions():
        st.subheader("Prediction Summary")
        preds = state.predictions
        if 'prediction' in preds.columns:
            counts = preds['prediction'].value_counts()
            fig = px.pie(
                values=counts.values, names=counts.index,
                title='Prediction Distribution',
                color_discrete_map={'positive': '#e74c3c', 'negative': '#2ecc71'}
            )
            st.plotly_chart(fig, width='stretch')
