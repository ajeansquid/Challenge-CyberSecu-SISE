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
        selected_features, algo, n_dims, color_col, params, max_samples = _render_controls(
            df, numeric_cols
        )

    if st.button("Run Projection", type="primary"):
        if len(selected_features) < 2:
            st.error("Please select at least 2 features.")
            return

        # Apply sampling if needed
        df_sample, was_sampled = _maybe_sample(df, max_samples)
        if was_sampled:
            st.info(f"Sampled {len(df_sample):,} rows from {len(df):,} for faster computation.")

        with st.spinner(f"Computing {algo} projection…"):
            projected, explained = _compute_projection(
                df_sample, selected_features, algo, n_dims, params
            )

        _render_results(projected, n_dims, color_col, algo, explained, was_sampled, len(df))


# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------

def _resolve_dataframe(state):
    """Return the most relevant dataframe available in state.

    Priority: labeled_data > features_data > predictions > raw_data.
    Merges in prediction labels and unsupervised results (cluster / is_anomaly)
    if available, so they can be used as colour columns in the projection.
    """
    if state.has_labeled_data():
        df = state.labeled_data.copy()
        st.info("Using labeled data")
    elif state.has_features():
        df = state.features_data.copy()
        st.info("Using generated features")
    elif state.has_predictions():
        st.info("Using prediction results")
        return state.predictions
    elif state.has_raw_data():
        st.info("Using raw data")
        return state.raw_data
    else:
        return None

    # Merge supervised predictions (prediction / probability columns)
    if state.has_predictions():
        for col in ('prediction', 'probability'):
            if col in state.predictions.columns:
                df[col] = state.predictions[col].reindex(df.index)

    # Merge unsupervised results (cluster or is_anomaly / anomaly_score)
    unsup = getattr(state, 'unsupervised_results', None)
    if unsup is not None:
        result_df = unsup.get('result_df')
        if result_df is not None:
            for col in ('cluster', 'is_anomaly', 'anomaly_score'):
                if col in result_df.columns:
                    df[col] = result_df[col].reindex(df.index)

    return df


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _maybe_sample(df: pd.DataFrame, max_samples: int) -> tuple:
    """Sample DataFrame if it exceeds max_samples. Returns (df, was_sampled)."""
    if max_samples <= 0 or len(df) <= max_samples:
        return df, False
    return df.sample(n=max_samples, random_state=42), True


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def _render_controls(df, numeric_cols):
    """Render user controls and return (features, algo, dims, color_col, params, max_samples)."""

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
        # Optional colour column – default to risk / prediction if present
        color_options = ["None"] + list(df.columns)
        preferred = next(
            (c for c in ("risk", "prediction", "cluster", "is_anomaly") if c in df.columns), "None"
        )
        default_idx = color_options.index(preferred)
        color_col = st.selectbox(
            "Color by", color_options, index=default_idx, key="dr_color"
        )
        if color_col == "None":
            color_col = None

        params = _render_algo_params(algo, len(df))

        # Sampling control
        st.markdown("---")
        if algo in ("t-SNE", "UMAP"):
            max_samples = st.slider(
                "Max samples (0 = all)",
                min_value=0,
                max_value=min(50000, len(df)),
                value=min(5000, len(df)),
                step=1000,
                help="t-SNE and UMAP are slow on large datasets. Sampling speeds up computation."
            )
        else:
            # PCA is fast to compute but Plotly 3D rendering lags on large point clouds.
            # Cap at 20 000 by default; set to 0 to disable.
            max_samples = st.slider(
                "Max samples for rendering (0 = all)",
                min_value=0,
                max_value=min(50000, len(df)),
                value=min(20000, len(df)),
                step=1000,
                help=(
                    "PCA computation is fast, but the 3D Plotly chart lags with many points. "
                    "Set to 0 to plot all points (may be slow for > 20 000 rows)."
                ),
            )

    return selected_features, algo, n_dims, color_col, params, max_samples


def _render_algo_params(algo: str, n_samples: int) -> dict:
    """Render algorithm-specific hyper-parameters."""
    params: dict = {}
    if algo == "PCA":
        # No extra params needed
        pass
    elif algo == "t-SNE":
        # Adjust perplexity range based on sample size
        max_perp = min(100, max(5, n_samples // 3))
        default_perp = min(30, max_perp)
        params["perplexity"] = st.slider(
            "Perplexity", 5, max_perp, default_perp, key="tsne_perplexity"
        )
        params["learning_rate"] = st.slider(
            "Learning rate", 10.0, 1000.0, 200.0, step=10.0, key="tsne_lr"
        )
        params["max_iter"] = st.slider(
            "Iterations", 250, 2000, 500, step=250, key="tsne_iter"
        )
    elif algo == "UMAP":
        params["n_neighbors"] = st.slider(
            "n_neighbors", 2, 100, 15, key="umap_neighbors"
        )
        params["min_dist"] = st.slider(
            "min_dist", 0.0, 1.0, 0.1, step=0.05, key="umap_mindist"
        )
        params["metric"] = st.selectbox(
            "Metric",
            ["euclidean", "manhattan", "cosine"],
            key="umap_metric",
        )
        params["reproducible"] = st.checkbox(
            "Reproducible (fixed seed)",
            value=False,
            help="Fixed seed gives the same layout every run but forces single-core. "
                 "Uncheck for faster parallel computation (layout varies slightly between runs).",
            key="umap_reproducible",
        )
    return params


# ---------------------------------------------------------------------------
# Projection computation (with caching)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _compute_projection_cached(
    X_scaled: np.ndarray,
    algo: str,
    n_dims: int,
    perplexity: int = 30,
    learning_rate: float = 200.0,
    max_iter: int = 500,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    reproducible: bool = False,
):
    """Cached projection computation."""
    explained = None

    if algo == "PCA":
        model = PCA(n_components=n_dims, random_state=42)
        coords = model.fit_transform(X_scaled)
        explained = model.explained_variance_ratio_

    elif algo == "t-SNE":
        model = TSNE(
            n_components=n_dims,
            perplexity=min(perplexity, len(X_scaled) - 1),
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42,
            init='pca',
        )
        coords = model.fit_transform(X_scaled)

    elif algo == "UMAP":
        try:
            import umap
        except ImportError:
            return None, None, "umap_not_installed"

        reducer = umap.UMAP(
            n_components=n_dims,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42 if reproducible else None,
        )
        coords = reducer.fit_transform(X_scaled)
    else:
        return None, None, f"unknown_algo_{algo}"

    return coords, explained, None


def _compute_projection(df, features, algo, n_dims, params):
    """
    Return (projected_df, explained_variance_or_None).
    projected_df has columns Dim1, Dim2 [, Dim3] plus all original columns.
    """
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Call cached computation
    coords, explained, error = _compute_projection_cached(
        X_scaled,
        algo,
        n_dims,
        perplexity=params.get("perplexity", 30),
        learning_rate=params.get("learning_rate", 200.0),
        max_iter=params.get("max_iter", 500),
        n_neighbors=params.get("n_neighbors", 15),
        min_dist=params.get("min_dist", 0.1),
        metric=params.get("metric", "euclidean"),
        reproducible=params.get("reproducible", False),
    )

    if error == "umap_not_installed":
        st.error(
            "**umap-learn** is not installed. "
            "Run `pip install umap-learn` and restart the app."
        )
        return pd.DataFrame(), None
    elif error:
        st.error(f"Error: {error}")
        return pd.DataFrame(), None

    dim_labels = [f"Dim{i+1}" for i in range(n_dims)]
    proj_df = pd.DataFrame(coords, columns=dim_labels, index=X.index)

    # Attach original columns for hover / colouring.
    orig = df.loc[X.index].reset_index(drop=True)
    proj_df = proj_df.reset_index(drop=True)
    proj_df = pd.concat([proj_df, orig], axis=1)

    return proj_df, explained


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@st.fragment
def _render_results(proj_df, n_dims, color_col, algo, explained, was_sampled, total_rows):
    """Render the projection results with interactive Plotly charts."""
    if proj_df.empty:
        return

    st.markdown("---")
    st.header(f"{algo} – {n_dims}D projection")

    if was_sampled:
        st.caption(f"Showing {len(proj_df):,} sampled points from {total_rows:,} total rows")

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
    dim_cols = [c for c in proj_df.columns if c.startswith("Dim")]
    _meta = {'cluster', 'is_anomaly', 'anomaly_score', 'prediction', 'probability', 'risk'}
    label_cols = [
        c for c in proj_df.columns
        if c not in dim_cols
        and (proj_df[c].dtype == object or c in _meta)
    ]
    with st.expander("View projected data"):
        st.caption(
            "Dim columns are computed locally for visualization only — "
            "they are **not** written back to the feature data in state."
        )
        tab_coords, tab_full = st.tabs(["Coordinates + labels", "Full joined table"])
        with tab_coords:
            show_cols = dim_cols + [c for c in label_cols if c in proj_df.columns]
            st.dataframe(proj_df[show_cols], width='stretch')
        with tab_full:
            st.dataframe(proj_df, width='stretch')

    # --- Download ---
    csv = proj_df.to_csv(index=False)
    st.download_button(
        "Download Projected Data",
        csv,
        f"{algo.lower()}_{n_dims}d.csv",
        "text/csv",
    )
