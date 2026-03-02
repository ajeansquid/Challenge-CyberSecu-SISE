# -*- coding: utf-8 -*-
"""Data Upload Page"""

import streamlit as st
import pandas as pd
import tempfile
import plotly.express as px

from app.state import get_state


def render():
    """Render data upload page."""
    state = get_state()

    st.title("Data Upload & Exploration")

    tab1, tab2, tab3 = st.tabs(["Upload", "Preview", "Statistics"])

    with tab1:
        render_upload_section(state)

    with tab2:
        render_preview_section(state)

    with tab3:
        render_stats_section(state)


def render_upload_section(state):
    """Render upload controls."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Logs (CSV)")
        raw_file = st.file_uploader(
            "Upload raw log file",
            type=['csv', 'txt'],
            key='raw_upload'
        )

        if raw_file:
            with st.expander("Parser Configuration"):
                separator = st.text_input("Separator", value=",")

            if st.button("Parse Logs", key='parse_btn'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
                    f.write(raw_file.getvalue())
                    temp_path = f.name

                try:
                    df = state.data_service.load_raw_logs(
                        temp_path,
                        parser_type='firewall',
                        separator=separator
                    )
                    state.raw_data = df
                    st.success(f"Loaded {len(df)} rows!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("Excel / CSV File")
        labeled_file = st.file_uploader(
            "Upload Excel or CSV file",
            type=['xlsx', 'csv'],
            key='labeled_upload'
        )

        if labeled_file:
            load_as = st.radio(
                "Load as",
                ["Labeled data (has target column)", "Raw logs (unlabeled, for feature engineering)"],
                key='load_as_radio'
            )

            if st.button("Load File", key='load_labeled_btn'):
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=labeled_file.name[-5:]
                    ) as f:
                        f.write(labeled_file.getvalue())
                        temp_path = f.name

                    if load_as.startswith("Labeled"):
                        df = state.data_service.load_labeled_data(temp_path)
                        state.labeled_data = df
                        st.success(f"Loaded {len(df)} labeled samples!")
                    else:
                        # Load raw — use generic CSV/Excel parser
                        import pandas as pd
                        from pathlib import Path
                        p = Path(temp_path)
                        df = pd.read_excel(temp_path) if p.suffix == '.xlsx' else pd.read_csv(temp_path)
                        state.raw_data = df
                        st.success(f"Loaded {len(df)} rows as raw data!")
                except Exception as e:
                    st.error(f"Error: {e}")


def render_preview_section(state):
    """Render data preview."""
    if state.has_raw_data():
        st.subheader("Raw Logs")
        st.dataframe(state.raw_data.head(100), width='stretch')

    if state.has_labeled_data():
        st.subheader("Labeled Data")
        st.dataframe(state.labeled_data.head(50), width='stretch')

    if not state.has_raw_data() and not state.has_labeled_data():
        st.info("Upload data to see preview")


def render_stats_section(state):
    """Render statistics."""
    if state.has_raw_data():
        df = state.raw_data
        st.subheader("Raw Data Statistics")

        cols = st.columns(4)
        cols[0].metric("Total Rows", f"{len(df):,}")
        cols[1].metric(
            "Unique IPs",
            df['ipsrc'].nunique() if 'ipsrc' in df.columns else "N/A"
        )
        cols[2].metric("Columns", len(df.columns))

        if 'action' in df.columns:
            deny_pct = (df['action'] == 'Deny').mean() * 100
            cols[3].metric("Deny Rate", f"{deny_pct:.1f}%")

            fig = px.pie(df, names='action', title='Action Distribution')
            st.plotly_chart(fig)

    if state.has_labeled_data():
        df = state.labeled_data
        st.subheader("Labeled Data Statistics")

        if 'risque' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    df, names='risque', title='Risk Distribution',
                    color_discrete_map={'positif': 'red', 'negatif': 'green'}
                )
                st.plotly_chart(fig)
            with col2:
                st.write("Class Distribution:")
                st.write(df['risque'].value_counts())
