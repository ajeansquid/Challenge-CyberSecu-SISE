# -*- coding: utf-8 -*-
"""Feature Engineering Page"""

import streamlit as st
import plotly.express as px
import numpy as np

from app.state import get_state


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
        input_df = state.raw_data if source == "Raw logs" else state.labeled_data
    elif has_raw:
        input_df = state.raw_data
        st.info("Using raw logs as input.")
    else:
        input_df = state.labeled_data
        st.info("Using labeled data as input (feature aggregation will use all rows).")

    render_config_section(state, input_df)

    if state.has_features():
        render_features_section(state)


def render_config_section(state, df):
    """Render configuration controls."""
    st.header("Generate Features")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Options")
        include_time = st.checkbox("Include time features", value=True)
        include_ratios = st.checkbox("Include ratio features", value=True)
        include_stats = st.checkbox("Include statistical features", value=True)
        admin_ports = st.text_input("Admin ports", value="21,22,3389,3306")

    with col2:
        st.subheader("Column Mapping")
        ip_col = st.selectbox("Source IP", df.columns, index=0)
        dst_col = st.selectbox("Destination IP", df.columns, index=min(1, len(df.columns)-1))
        port_col = st.selectbox("Port", df.columns, index=min(2, len(df.columns)-1))
        action_col = st.selectbox("Action", df.columns, index=min(4, len(df.columns)-1))
        date_col = st.selectbox("Date", ['None'] + list(df.columns))

    if st.button("Generate Features", type="primary"):
        try:
            feature_set = state.feature_service.extract_full_features(
                df,
                ip_col=ip_col,
                dst_col=dst_col,
                port_col=port_col,
                action_col=action_col,
                date_col=date_col if date_col != 'None' else None,
                include_time=include_time,
                include_ratios=include_ratios,
                include_stats=include_stats,
                save_as='generated'
            )
            state.features_data = feature_set.data
            st.success(
                f"Generated {len(feature_set.feature_names)} features "
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
