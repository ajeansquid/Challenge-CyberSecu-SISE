# -*- coding: utf-8 -*-
"""Analysis Dashboard Page"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from app.state import get_state


def render():
    """Render analysis dashboard."""
    state = get_state()

    st.title("Analysis Dashboard")

    # Priority: labeled_data > features_data > predictions
    # labeled_data is the ground-truth CSV with all features intact.
    # features_data may have zero-filled columns when action parsing failed.
    # Merge in the prediction label when available so charts can colour by class.
    df = None
    if state.has_labeled_data():
        df = state.labeled_data.copy()
        st.info("Analyzing labeled data")
        if state.has_predictions() and 'prediction' in state.predictions.columns:
            pred_series = state.predictions['prediction'].reindex(df.index)
            df['prediction'] = pred_series
    elif state.has_features():
        df = state.features_data.copy()
        st.info("Analyzing feature data")
        if state.has_predictions() and 'prediction' in state.predictions.columns:
            pred_series = state.predictions['prediction'].reindex(df.index)
            df['prediction'] = pred_series
    elif state.has_predictions():
        df = state.predictions
        st.info("Analyzing prediction results")
    else:
        st.warning("No data available for analysis.")
        return

    render_overview(df)

    tabs = st.tabs(["Distribution", "Correlation", "Clustering", "Anomaly Detection"])

    with tabs[0]:
        render_distribution(df)

    with tabs[1]:
        render_correlation(df)

    with tabs[2]:
        render_clustering(state, df)

    with tabs[3]:
        render_anomaly(state, df)


def render_overview(df):
    """Render overview metrics."""
    st.header("Overview")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    cols = st.columns(4)
    cols[0].metric("Total IPs", len(df))

    if 'nombre' in df.columns:
        cols[1].metric("Total Accesses", f"{df['nombre'].sum():,}")
    if 'deny' in df.columns:
        cols[2].metric("Total Denies", f"{df['deny'].sum():,}")
    if 'prediction' in df.columns:
        pos = (df['prediction'] == 'positif').sum()
        cols[3].metric("Threats", pos)


def render_distribution(df):
    """Render distribution analysis."""
    st.header("Distribution Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature = st.selectbox("Feature", numeric_cols, key='dist_feat')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x=feature, nbins=50, title=f'{feature} Distribution')
        st.plotly_chart(fig)

    with col2:
        fig = px.box(df, y=feature, title=f'{feature} Box Plot')
        st.plotly_chart(fig)

    # By class if available
    class_col = None
    if 'prediction' in df.columns:
        class_col = 'prediction'
    elif 'risque' in df.columns:
        class_col = 'risque'

    if class_col:
        st.subheader(f"By {class_col}")
        fig = px.histogram(
            df, x=feature, color=class_col,
            barmode='overlay', title=f'{feature} by {class_col}'
        )
        st.plotly_chart(fig)


def render_correlation(df):
    """Render correlation analysis."""
    st.header("Correlation Analysis")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    selected = st.multiselect("Features", numeric_cols, default=numeric_cols[:8])

    if len(selected) >= 2:
        corr = df[selected].corr()
        fig = px.imshow(
            corr, title="Correlation Matrix",
            color_continuous_scale='RdBu_r', text_auto='.2f'
        )
        st.plotly_chart(fig)

        # Scatter
        st.subheader("Scatter Plot")
        col1, col2 = st.columns(2)
        x_feat = col1.selectbox("X", selected, key='scatter_x')
        y_feat = col2.selectbox("Y", selected, index=1, key='scatter_y')

        color_by = None
        if 'prediction' in df.columns:
            color_by = 'prediction'
        elif 'risque' in df.columns:
            color_by = 'risque'

        df_plot = df.reset_index(drop=True)
        fig = px.scatter(
            df_plot, x=x_feat, y=y_feat, color=color_by,
            title=f'{x_feat} vs {y_feat}',
            hover_data=[df_plot.columns[0]]
        )
        st.plotly_chart(fig)


def render_clustering(state, df):
    """Render clustering analysis."""
    st.header("Clustering")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    default_feats = ['nombre', 'cnbripdst', 'cnportdst']
    default = [f for f in default_feats if f in numeric_cols] or numeric_cols[:3]

    features = st.multiselect("Features", numeric_cols, default=default, key='cluster_feats')
    n_clusters = st.slider("Clusters", 2, 10, 3)

    if st.button("Run Clustering") and len(features) >= 2:
        try:
            # Find optimal K
            inertias = state.model_service.find_optimal_clusters(df, features)

            st.subheader("Elbow Plot")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(inertias.keys()),
                y=list(inertias.values()),
                mode='lines+markers'
            ))
            fig.update_layout(xaxis_title="K", yaxis_title="Inertia")
            st.plotly_chart(fig)

            # Cluster
            result = state.model_service.cluster(df, features, n_clusters)

            st.subheader("Clusters")
            fig = px.scatter(
                result, x=features[0], y=features[1],
                color='cluster', title='Cluster Visualization'
            )
            st.plotly_chart(fig)

            st.subheader("Cluster Stats")
            stats = result.groupby('cluster')[features].mean()
            st.dataframe(stats, width='stretch')

        except Exception as e:
            st.error(f"Error: {e}")


def render_anomaly(state, df):
    """Render anomaly detection."""
    st.header("Anomaly Detection")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    features = st.multiselect(
        "Features", numeric_cols,
        default=numeric_cols[:5], key='anomaly_feats'
    )
    contamination = st.slider("Anomaly Rate", 0.01, 0.5, 0.1)

    if st.button("Detect Anomalies") and features:
        try:
            result = state.model_service.detect_anomalies(
                df, features, contamination
            )

            n_anomalies = result['is_anomaly'].sum()
            st.metric("Detected Anomalies", n_anomalies)

            fig = px.histogram(
                result, x='anomaly_score', color='is_anomaly',
                title='Anomaly Score Distribution'
            )
            st.plotly_chart(fig)

            st.subheader("Anomalous IPs")
            anomalies = result[result['is_anomaly']].sort_values('anomaly_score')
            st.dataframe(anomalies, width='stretch')

        except Exception as e:
            st.error(f"Error: {e}")
