# -*- coding: utf-8 -*-
"""Feature Engineering Page"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from app.state import get_state
from utils.helpers import normalize_log_columns
from core.interfaces import FeatureSet


def render():
    """Render feature engineering page."""
    state = get_state()

    st.title("Feature Engineering")

    has_raw    = state.has_raw_data()
    has_labeled = state.has_labeled_data()

    if not has_raw and not has_labeled:
        st.warning("Please upload raw log data or a labeled dataset first.")
        return

    # If only labeled data is loaded, the features already exist — warn and offer early exit.
    if has_labeled and not has_raw:
        st.info(
            "Your file already contains the aggregated features (e.g. `nombre`, `permit`, `deny` …). "
            "**Feature Engineering is not needed** — head straight to **Model Training**."
        )
        with st.expander("Run anyway (advanced)"):
            render_config_section(state, state.labeled_data)
        return

    # Let user choose source when both are available
    if has_raw and has_labeled:
        source = st.radio(
            "Input data source",
            ["Raw logs", "Labeled data"],
            horizontal=True,
        )
        input_df = normalize_log_columns(state.raw_data.copy()) if source == "Raw logs" else state.labeled_data
    elif has_raw:
        input_df = normalize_log_columns(state.raw_data.copy())
        st.info("Using raw logs as input.")
    else:
        input_df = state.labeled_data
        st.info("Using labeled data as input (feature aggregation will use all rows).")

    render_config_section(state, input_df)

    if state.has_features():
        render_features_section(state)


@st.cache_data(show_spinner=False)
def _cached_extract_features(
    df: pd.DataFrame,
    ip_col: str,
    dst_col: str,
    port_col: str,
    action_col: str,
    date_col,
    include_time: bool,
    include_ratios: bool,
    include_stats: bool,
    remove_correlated: bool,
    corr_threshold: float,
) -> FeatureSet:
    """Cached wrapper so repeated UI interactions don't re-run extraction."""
    from services import FeatureService
    svc = FeatureService()
    return svc.extract_full_features(
        df,
        ip_col=ip_col,
        dst_col=dst_col,
        port_col=port_col,
        action_col=action_col,
        date_col=date_col,
        include_time=include_time,
        include_ratios=include_ratios,
        include_stats=include_stats,
        remove_correlated=remove_correlated,
        corr_threshold=corr_threshold,
        save_as=None,
    )


def render_config_section(state, df):
    """Render configuration controls."""
    st.header("Generate Features")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Options")
        include_time = st.checkbox("Include time features", value=True)
        include_ratios = st.checkbox("Include ratio features", value=True)
        include_stats = st.checkbox("Include statistical features", value=True)
        st.text_input("Admin ports", value="21,22,3389,3306")

        st.markdown("---")
        remove_correlated = st.checkbox(
            "Remove highly correlated features",
            value=False,
            help="Automatically remove features with |r| > threshold to reduce multicollinearity. "
                 "⚠️ May remove intentionally engineered features."
        )
        corr_threshold = 0.95  # default
        if remove_correlated:
            corr_threshold = st.slider("Correlation threshold", 0.8, 0.99, 0.95, 0.01)

    with col2:
        st.subheader("Column Mapping")
        cols_list = list(df.columns)

        # Find canonical column indices with fallbacks
        def find_col_idx(preferred, fallback_idx=0):
            if preferred in cols_list:
                return cols_list.index(preferred)
            return min(fallback_idx, len(cols_list) - 1)

        ip_col = st.selectbox("Source IP", cols_list, index=find_col_idx('ipsrc', 0))
        dst_col = st.selectbox("Destination IP", cols_list, index=find_col_idx('ipdst', 1))
        port_col = st.selectbox("Port", cols_list, index=find_col_idx('portdst', 2))
        action_col = st.selectbox("Action", cols_list, index=find_col_idx('action', 4))
        date_col = st.selectbox("Date", ['None'] + cols_list,
                                 index=cols_list.index('date') + 1 if 'date' in cols_list else 0)

    if st.button("Generate Features", type="primary"):
        with st.spinner("Extracting features — this may take a moment for large datasets…"):
            try:
                feature_set = _cached_extract_features(
                    df,
                    ip_col=ip_col,
                    dst_col=dst_col,
                    port_col=port_col,
                    action_col=action_col,
                    date_col=date_col if date_col != 'None' else None,
                    include_time=include_time,
                    include_ratios=include_ratios,
                    include_stats=include_stats,
                    remove_correlated=remove_correlated,
                    corr_threshold=corr_threshold if remove_correlated else 0.95,
                )
                state.features_data = feature_set.data

                # Check for removed features
                removed_const = feature_set.metadata.get('removed_constant_features', [])
                removed_corr = feature_set.metadata.get('removed_correlated_features', [])

                if removed_const:
                    st.info(
                        f"ℹ️ Removed {len(removed_const)} constant feature(s) (zero variance): "
                        f"{', '.join(removed_const)}"
                    )
                if removed_corr:
                    threshold_used = corr_threshold if remove_correlated else 0.95
                    st.info(
                        f"ℹ️ Removed {len(removed_corr)} highly correlated feature(s) (|r| > {threshold_used}): "
                        f"{', '.join(removed_corr)}"
                    )

                st.success(
                    f"✅ Generated {len(feature_set.feature_names)} features "
                    f"for {len(feature_set.data)} IPs!"
                )
            except Exception as e:
                st.error(f"Error: {e}")


def render_features_section(state):
    """Render generated features."""
    st.header("Generated Features")
    df = state.features_data

    st.subheader("Statistics")
    st.dataframe(df.describe(), width='stretch')

    st.subheader("Preview")
    st.dataframe(df.head(20), width='stretch')

    # Download
    csv = df.to_csv()
    st.download_button("Download CSV", csv, "features.csv", "text/csv")

    # Correlation
    st.subheader("Correlations")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    st.plotly_chart(fig)
