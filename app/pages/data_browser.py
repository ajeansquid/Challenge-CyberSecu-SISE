# -*- coding: utf-8 -*-
"""Data Browser Page - Interactive paginated data table with filtering and export"""

import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

from app.state import get_state
from utils.helpers import normalize_log_columns


def get_data_from_state():
    """Get data from AppState (uploaded via Data Upload page)."""
    state = get_state()
    if not state.has_raw_data():
        return None
    df = normalize_log_columns(state.raw_data.copy())

    # Pre-convert heavy columns
    date_col = 'date' if 'date' in df.columns else ('timestamp' if 'timestamp' in df.columns else None)
    if date_col:
        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    if 'portdst' in df.columns:
        df['portdst_num'] = pd.to_numeric(df['portdst'], errors='coerce')

    return df


@st.cache_data(show_spinner=False)
def _filter_options(df: pd.DataFrame) -> dict:
    """Pre-compute unique values for filter widgets (cached)."""
    opts: dict = {}
    for col in ('proto', 'action', 'regle', 'interface_in'):
        if col in df.columns:
            opts[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    return opts


def render():
    """Render data browser page."""
    st.title("Data Browser")
    st.markdown("Browse, filter, search, and export firewall log data.")

    # Get data from AppState (uploaded via Data Upload page)
    df = get_data_from_state()

    if df is None:
        st.warning("No data loaded. Please upload data in the **Data Upload** page first.")
        return

    st.info(f"Total records: {len(df):,}")

    opts = _filter_options(df)

    # Filters in sidebar
    st.sidebar.header("Filters")

    # --- build boolean mask incrementally (no intermediate copies) ---
    mask = pd.Series(True, index=df.index)

    # Date range filter
    if 'datetime' in df.columns:
        min_date = df['datetime'].min()
        max_date = df['datetime'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask &= (df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)

    # Protocol filter
    protocols = ['All'] + opts.get('proto', [])
    selected_proto = st.sidebar.selectbox("Protocol", protocols)
    if selected_proto != 'All':
        mask &= df['proto'] == selected_proto

    # Action filter
    actions = ['All'] + opts.get('action', [])
    selected_action = st.sidebar.selectbox("Action", actions)
    if selected_action != 'All':
        mask &= df['action'] == selected_action

    # Port range filter
    st.sidebar.subheader("Port Filters")
    port_min = st.sidebar.number_input("Min Dest Port", min_value=0, max_value=65535, value=0)
    port_max = st.sidebar.number_input("Max Dest Port", min_value=0, max_value=65535, value=65535)
    if 'portdst_num' in df.columns:
        mask &= (df['portdst_num'] >= port_min) & (df['portdst_num'] <= port_max)

    # Rule filter
    rules = ['All'] + opts.get('regle', [])
    selected_rule = st.sidebar.selectbox("Rule", rules)
    if selected_rule != 'All':
        mask &= df['regle'] == selected_rule

    # Interface filter
    if 'interface_in' in df.columns:
        interfaces = ['All'] + opts.get('interface_in', [])
        selected_interface = st.sidebar.selectbox("Interface", interfaces)
        if selected_interface != 'All':
            mask &= df['interface_in'] == selected_interface

    df = df[mask]

    # Search functionality
    st.subheader("Search")
    col1, col2 = st.columns([3, 1])

    with col1:
        search_term = st.text_input("Search (IP, port, rule...)", placeholder="Enter search term")

    with col2:
        search_column = st.selectbox(
            "Search in",
            ["All columns", "ipsrc", "ipdst", "portdst", "regle"]
        )

    if search_term:
        if search_column == "All columns":
            # Limit to key string columns to avoid O(n*cols) scan on 4.5M rows
            str_cols = [c for c in ['ipsrc', 'ipdst', 'portdst', 'regle', 'src_ip', 'dst_ip', 'dst_port'] if c in df.columns]
            smask = pd.Series(False, index=df.index)
            for col in str_cols:
                smask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
        else:
            smask = df[search_column].astype(str).str.contains(search_term, case=False, na=False)
        df = df[smask]

    # Display filtered count
    st.success(f"Showing {len(df):,} records after filters")

    # Column selection
    st.subheader("Display Options")
    available_columns = [c for c in df.columns if c not in ['datetime', 'portdst_num', 'date']]
    selected_columns = st.multiselect(
        "Columns to display",
        available_columns,
        default=available_columns
    )

    # Sorting
    col1, col2 = st.columns(2)
    with col1:
        sort_column = st.selectbox("Sort by", selected_columns)
    with col2:
        sort_order = st.selectbox("Order", ["Ascending", "Descending"])

    ascending = sort_order == "Ascending"
    df_display = df[selected_columns].sort_values(by=sort_column, ascending=ascending)

    # Pagination
    st.subheader("Data Table")

    rows_per_page = st.select_slider(
        "Rows per page",
        options=[10, 25, 50, 100, 250, 500],
        value=50
    )

    total_rows = len(df_display)
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)

    # Clamp saved page so buttons don't leave us out-of-bounds after filter changes
    if 'data_browser_page' in st.session_state:
        st.session_state.data_browser_page = max(1, min(st.session_state.data_browser_page, total_pages))

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key='data_browser_page',
        )

    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)

    st.write(f"Showing rows {start_idx + 1:,} to {end_idx:,} of {total_rows:,}")

    # Display the data table
    st.dataframe(
        df_display.iloc[start_idx:end_idx],
        width='stretch',
        hide_index=True
    )

    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("First Page", disabled=page == 1):
            st.session_state.data_browser_page = 1
            st.rerun()
    with col2:
        if st.button("Previous", disabled=page == 1):
            st.session_state.data_browser_page = page - 1
            st.rerun()
    with col3:
        if st.button("Next", disabled=page == total_pages):
            st.session_state.data_browser_page = page + 1
            st.rerun()
    with col4:
        if st.button("Last Page", disabled=page == total_pages):
            st.session_state.data_browser_page = total_pages
            st.rerun()

    # Export functionality
    st.markdown("---")
    st.subheader("Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export current page as CSV
        csv_page = df_display.iloc[start_idx:end_idx].to_csv(index=False)
        st.download_button(
            label="Download Current Page (CSV)",
            data=csv_page,
            file_name="firewall_logs_page.csv",
            mime="text/csv"
        )

    with col2:
        # Export filtered data as CSV (limit to 100k rows for performance)
        if len(df_display) <= 100000:
            csv_filtered = df_display.to_csv(index=False)
            st.download_button(
                label=f"Download Filtered Data ({len(df_display):,} rows)",
                data=csv_filtered,
                file_name="firewall_logs_filtered.csv",
                mime="text/csv"
            )
        else:
            st.warning(f"Filtered data too large ({len(df_display):,} rows). Export limited to 100k rows.")
            csv_filtered = df_display.head(100000).to_csv(index=False)
            st.download_button(
                label="Download First 100k Rows (CSV)",
                data=csv_filtered,
                file_name="firewall_logs_filtered_100k.csv",
                mime="text/csv"
            )

    with col3:
        # Export summary statistics
        summary = df_display.describe(include='all').to_csv()
        st.download_button(
            label="Download Summary Stats",
            data=summary,
            file_name="firewall_logs_summary.csv",
            mime="text/csv"
        )

    # Quick statistics for filtered data
    st.markdown("---")
    st.subheader("Quick Statistics (Filtered Data)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Records", f"{len(df_display):,}")
    with col2:
        if 'ipsrc' in df_display.columns:
            st.metric("Unique Source IPs", f"{df_display['ipsrc'].nunique():,}")
    with col3:
        if 'portdst' in df_display.columns:
            st.metric("Unique Dest Ports", f"{df_display['portdst'].nunique():,}")
    with col4:
        if 'action' in df_display.columns:
            permits = (df_display['action'] == 'PERMIT').sum()
            st.metric("PERMIT/DENY", f"{permits:,} / {len(df_display) - permits:,}")
