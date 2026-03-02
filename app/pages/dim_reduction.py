# -*- coding: utf-8 -*-
"""Dimensionality Reduction Page – PCA, t-SNE, UMAP with 2D / 3D visualization."""

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from app.state import get_state


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render():
    """Render the dimensionality-reduction page."""
    state = get_state()

    st.title("Dimensionality Reduction")
    st.markdown(
        "Explore high-dimensional feature spaces with **PCA**, **t-SNE** and **UMAP**."
    )

    # ---- pick the best available dataframe ----
    df = _resolve_dataframe(state)
    if df is None:
        st.warning("No numeric data available. Upload data or generate features first.")
        return

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) < 2:
        st.warning("At least 2 numeric columns are required for dimensionality reduction.")
        return

    # ---- sidebar-style controls in an expander ----
    with st.expander("⚙️ Settings", expanded=True):
        selected_features, algo, n_dims, color_col, params = _render_controls(
            df, numeric_cols
        )

    if st.button("Run Projection", type="primary"):
        if len(selected_features) < 2:
            st.error("Please select at least 2 features.")
            return

        with st.spinner(f"Computing {algo} projection…"):
            projected, explained = _compute_projection(
                df, selected_features, algo, n_dims, params
            )

        _render_results(projected, n_dims, color_col, algo, explained)


# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------

def _resolve_dataframe(state):
    """Return the most relevant dataframe available in state.

    Priority: labeled_data (has risque) > features_data > predictions > raw_data.
    When labeled_data is used, prediction labels are merged in if available.
    """
    if state.has_labeled_data():
        df = state.labeled_data.copy()
        if state.has_predictions() and 'prediction' in state.predictions.columns:
            df['prediction'] = state.predictions['prediction'].reindex(df.index)
        st.info("Using labeled data")
        return df
    if state.has_features():
        df = state.features_data.copy()
        if state.has_predictions() and 'prediction' in state.predictions.columns:
            df['prediction'] = state.predictions['prediction'].reindex(df.index)
        st.info("Using generated features")
        return df
    if state.has_predictions():
        st.info("Using prediction results")
        return state.predictions
    if state.has_raw_data():
        st.info("Using raw data")
        return state.raw_data
    return None


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def _render_controls(df, numeric_cols):
    """Render user controls and return (features, algo, dims, color_col, params)."""

    col1, col2 = st.columns(2)

    with col1:
        selected_features = st.multiselect(
            "Features to project",
            numeric_cols,
            default=numeric_cols[:min(8, len(numeric_cols))],
            key="dr_features",
        )

        algo = st.selectbox(
            "Algorithm",
            ["PCA", "t-SNE", "UMAP"],
            key="dr_algo",
        )

        n_dims = st.radio("Dimensions", [2, 3], horizontal=True, key="dr_dims")

    with col2:
        # Optional colour column – default to risque / prediction if present
        color_options = ["None"] + list(df.columns)
        preferred = next(
            (c for c in ("risque", "prediction") if c in df.columns), "None"
        )
        default_idx = color_options.index(preferred)
        color_col = st.selectbox(
            "Color by", color_options, index=default_idx, key="dr_color"
        )
        if color_col == "None":
            color_col = None

        params = _render_algo_params(algo)

    return selected_features, algo, n_dims, color_col, params


def _render_algo_params(algo: str) -> dict:
    """Render algorithm-specific hyper-parameters."""
    params: dict = {}
    if algo == "PCA":
        # No extra params needed
        pass
    elif algo == "t-SNE":
        params["perplexity"] = st.slider(
            "Perplexity", 5, 100, 30, key="tsne_perplexity"
        )
        params["learning_rate"] = st.slider(
            "Learning rate", 10.0, 1000.0, 200.0, step=10.0, key="tsne_lr"
        )
        params["n_iter"] = st.slider(
            "Iterations", 250, 5000, 1000, step=250, key="tsne_iter"
        )
    elif algo == "UMAP":
        params["n_neighbors"] = st.slider(
            "n_neighbors", 2, 200, 15, key="umap_neighbors"
        )
        params["min_dist"] = st.slider(
            "min_dist", 0.0, 1.0, 0.1, step=0.05, key="umap_mindist"
        )
        params["metric"] = st.selectbox(
            "Metric",
            ["euclidean", "manhattan", "cosine", "chebyshev"],
            key="umap_metric",
        )
    return params


# ---------------------------------------------------------------------------
# Projection computation
# ---------------------------------------------------------------------------

def _compute_projection(df, features, algo, n_dims, params):
    """
    Return (projected_df, explained_variance_or_None).
    projected_df has columns Dim1, Dim2 [, Dim3] plus all original columns.
    """
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    explained = None

    if algo == "PCA":
        model = PCA(n_components=n_dims, random_state=42)
        coords = model.fit_transform(X_scaled)
        explained = model.explained_variance_ratio_

    elif algo == "t-SNE":
        model = TSNE(
            n_components=n_dims,
            perplexity=min(params.get("perplexity", 30), len(X_scaled) - 1),
            learning_rate=params.get("learning_rate", 200.0),
            n_iter=params.get("n_iter", 1000),
            random_state=42,
        )
        coords = model.fit_transform(X_scaled)

    elif algo == "UMAP":
        try:
            import umap  # noqa: F811
        except ImportError:
            st.error(
                "**umap-learn** is not installed. "
                "Run `pip install umap-learn` and restart the app."
            )
            return pd.DataFrame(), None

        reducer = umap.UMAP(
            n_components=n_dims,
            n_neighbors=params.get("n_neighbors", 15),
            min_dist=params.get("min_dist", 0.1),
            metric=params.get("metric", "euclidean"),
            random_state=42,
        )
        coords = reducer.fit_transform(X_scaled)
    else:
        st.error(f"Unknown algorithm: {algo}")
        return pd.DataFrame(), None

    dim_labels = [f"Dim{i+1}" for i in range(n_dims)]
    proj_df = pd.DataFrame(coords, columns=dim_labels, index=X.index)

    # Attach original columns for hover / colouring.
    # Use drop=True to avoid inserting the index as a column, which would
    # clash when a column already has the same name as the index (e.g. 'nombre').
    orig = df.loc[X.index].reset_index(drop=True)
    proj_df = proj_df.reset_index(drop=True)
    proj_df = pd.concat([proj_df, orig], axis=1)

    return proj_df, explained


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _render_results(proj_df, n_dims, color_col, algo, explained):
    """Render the projection results with interactive Plotly charts."""
    if proj_df.empty:
        return

    st.markdown("---")
    st.header(f"{algo} – {n_dims}D projection")

    # --- Explained variance (PCA only) ---
    if explained is not None:
        cols = st.columns(n_dims + 1)
        total = sum(explained)
        for i, ev in enumerate(explained):
            cols[i].metric(f"Dim {i+1}", f"{ev*100:.1f}%")
        cols[n_dims].metric("Total", f"{total*100:.1f}%")

    # --- 2D plot ---
    if n_dims == 2:
        fig = px.scatter(
            proj_df,
            x="Dim1",
            y="Dim2",
            color=color_col,
            title=f"{algo} – 2D",
            hover_data=proj_df.columns[:10],
            template="plotly_white",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')

    # --- 3D plot ---
    if n_dims == 3:
        fig = px.scatter_3d(
            proj_df,
            x="Dim1",
            y="Dim2",
            z="Dim3",
            color=color_col,
            title=f"{algo} – 3D",
            hover_data=proj_df.columns[:10],
            template="plotly_white",
            color_continuous_scale="Viridis",
        )
        fig.update_traces(marker=dict(size=3, opacity=0.7))
        fig.update_layout(height=700)
        st.plotly_chart(fig, width='stretch')

    # --- Data table ---
    with st.expander("View projected data"):
        st.dataframe(proj_df, width='stretch')

    # --- Download ---
    csv = proj_df.to_csv(index=False)
    st.download_button(
        "Download Projected Data",
        csv,
        f"{algo.lower()}_{n_dims}d.csv",
        "text/csv",
    )
