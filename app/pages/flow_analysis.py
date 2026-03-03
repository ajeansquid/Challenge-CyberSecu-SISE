# -*- coding: utf-8 -*-
"""Flow Analysis Page - PERMIT/DENY by protocol and port ranges"""

import streamlit as st
import pandas as pd
import plotly.express as px

from app.state import get_state
from utils.helpers import normalize_log_columns


# RFC 6056 port range categories
PORT_RANGES = {
    "Well-known (0-1023)": (0, 1023),
    "Registered (1024-49151)": (1024, 49151),
    "Dynamic/Private (49152-65535)": (49152, 65535),
}
PORT_LABELS = list(PORT_RANGES.keys())
PORT_BINS   = [0, 1023, 49151, 65535]


def get_data_from_state():
    """Get data from AppState (uploaded via Data Upload page)."""
    state = get_state()
    if not state.has_raw_data():
        return None
    df = normalize_log_columns(state.raw_data.copy())

    # Ensure date column is datetime
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Vectorised port conversion + categorisation
    if 'portdst' in df.columns:
        df['portdst'] = pd.to_numeric(df['portdst'], errors='coerce')
    else:
        df['portdst'] = float('nan')
    if 'portsrc' in df.columns:
        df['portsrc'] = pd.to_numeric(df['portsrc'], errors='coerce')
    df['port_category'] = pd.cut(
        df['portdst'],
        bins=[-1, 1023, 49151, 65535],
        labels=PORT_LABELS,
        right=True,
    ).astype(str)
    df.loc[df['port_category'] == 'nan', 'port_category'] = 'Unknown'
    return df


def render():
    """Render flow analysis page."""
    st.title("Flow Analysis")
    st.markdown("Analyze network traffic patterns by protocol, action, and port ranges.")

    # Get data from AppState (uploaded via Data Upload page)
    df = get_data_from_state()

    if df is None:
        st.warning("No data loaded. Please upload data in the **Data Upload** page first.")
        return

    # Display data info
    st.info(f"Loaded {len(df):,} log entries")

    # Filters in sidebar
    st.sidebar.header("Filters")

    # Protocol filter
    protocols = df['proto'].unique().tolist()
    selected_protocols = st.sidebar.multiselect(
        "Protocols",
        protocols,
        default=protocols
    )

    # Action filter
    actions = df['action'].unique().tolist()
    selected_actions = st.sidebar.multiselect(
        "Actions",
        actions,
        default=actions
    )

    # Port range filter
    selected_port_ranges = st.sidebar.multiselect(
        "Port Ranges (RFC 6056)",
        list(PORT_RANGES.keys()),
        default=list(PORT_RANGES.keys())
    )

    # Apply filters (boolean indexing on cached df — no copy until needed)
    mask = (
        df['proto'].isin(selected_protocols) &
        df['action'].isin(selected_actions) &
        df['port_category'].isin(selected_port_ranges)
    )
    filtered_df = df.loc[mask]

    st.success(f"Showing {len(filtered_df):,} entries after filters")

    # Overview metrics
    render_overview_metrics(filtered_df)

    # Tabs for different visualizations
    tabs = st.tabs(["By Protocol", "By Port Range", "Time Series", "Detailed Breakdown"])

    with tabs[0]:
        render_protocol_analysis(filtered_df)

    with tabs[1]:
        render_port_range_analysis(filtered_df)

    with tabs[2]:
        render_time_series(filtered_df)

    with tabs[3]:
        render_detailed_breakdown(filtered_df)


def render_overview_metrics(df: pd.DataFrame):
    """Render overview metrics cards."""
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    permits = (df['action'] == 'PERMIT').sum()
    denies = (df['action'] == 'DENY').sum()
    tcp_count = (df['proto'] == 'TCP').sum()
    udp_count = (df['proto'] == 'UDP').sum()

    col1.metric("Total Flows", f"{total:,}")
    col2.metric("PERMIT", f"{permits:,}", f"{permits/total*100:.1f}%" if total > 0 else "0%")
    col3.metric("DENY", f"{denies:,}", f"{denies/total*100:.1f}%" if total > 0 else "0%")
    col4.metric("TCP / UDP", f"{tcp_count:,} / {udp_count:,}")


def render_protocol_analysis(df: pd.DataFrame):
    """Render protocol-based analysis."""
    st.subheader("PERMIT/DENY by Protocol")

    # Group by protocol and action
    proto_action = df.groupby(['proto', 'action'], observed=False).size().reset_index(name='count')

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart
        fig = px.bar(
            proto_action,
            x='proto',
            y='count',
            color='action',
            barmode='group',
            title="Flow Count by Protocol and Action",
            color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
        )
        fig.update_layout(xaxis_title="Protocol", yaxis_title="Count")
        st.plotly_chart(fig, width='stretch')

    with col2:
        # Pie charts for each protocol
        fig = px.sunburst(
            proto_action,
            path=['proto', 'action'],
            values='count',
            title="Protocol and Action Distribution",
            color='action',
            color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
        )
        st.plotly_chart(fig, width='stretch')

    # Data table
    st.subheader("Summary Table")
    pivot = df.pivot_table(
        index='proto',
        columns='action',
        aggfunc='size',
        fill_value=0,
        observed=False
    )
    pivot['Total'] = pivot.sum(axis=1)
    pivot['PERMIT %'] = (pivot.get('PERMIT', 0) / pivot['Total'] * 100).round(1)
    st.dataframe(pivot, width='stretch')


def render_port_range_analysis(df: pd.DataFrame):
    """Render port range analysis."""
    st.subheader("Analysis by Port Range (RFC 6056)")

    # Group by port category and action
    port_action = df.groupby(['port_category', 'action'], observed=False).size().reset_index(name='count')

    # Order categories properly
    category_order = list(PORT_RANGES.keys())
    port_action['port_category'] = pd.Categorical(
        port_action['port_category'],
        categories=category_order,
        ordered=True
    )
    port_action = port_action.sort_values('port_category')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            port_action,
            x='port_category',
            y='count',
            color='action',
            barmode='group',
            title="Flows by Port Range and Action",
            color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
        )
        fig.update_layout(xaxis_title="Port Range", yaxis_title="Count")
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = px.pie(
            port_action,
            values='count',
            names='port_category',
            title="Distribution by Port Range",
            hole=0.4
        )
        st.plotly_chart(fig, width='stretch')

    # Detailed breakdown
    st.subheader("Port Range Details")
    pivot = df.pivot_table(
        index='port_category',
        columns='action',
        aggfunc='size',
        fill_value=0,
        observed=False
    )
    pivot['Total'] = pivot.sum(axis=1)
    if 'PERMIT' in pivot.columns:
        pivot['PERMIT %'] = (pivot['PERMIT'] / pivot['Total'] * 100).round(1)
    if 'DENY' in pivot.columns:
        pivot['DENY %'] = (pivot['DENY'] / pivot['Total'] * 100).round(1)
    st.dataframe(pivot, width='stretch')


def render_time_series(df: pd.DataFrame):
    """Render time series analysis."""
    st.subheader("Traffic Over Time")

    # 'date' is already a datetime column (converted at load time)
    df_ts = df.dropna(subset=['date']).rename(columns={'date': 'datetime'})

    if len(df_ts) == 0:
        st.warning("No valid timestamps found in data.")
        return

    # Time granularity selector
    granularity = st.selectbox(
        "Time Granularity",
        ["Hour", "Day", "Week"],
        index=1
    )

    freq_map = {"Hour": "h", "Day": "D", "Week": "W"}
    freq = freq_map[granularity]

    # Group by time and action
    df_ts['period'] = df_ts['datetime'].dt.to_period(freq).astype(str)
    time_action = df_ts.groupby(['period', 'action']).size().reset_index(name='count')

    fig = px.line(
        time_action,
        x='period',
        y='count',
        color='action',
        title=f"Traffic by {granularity}",
        color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Flow Count")
    st.plotly_chart(fig, width='stretch')

    # Heatmap by hour of day and day of week
    st.subheader("Activity Heatmap")
    df_ts['hour'] = df_ts['datetime'].dt.hour
    df_ts['dayofweek'] = df_ts['datetime'].dt.day_name()

    heatmap_data = df_ts.groupby(['dayofweek', 'hour']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='dayofweek', columns='hour', values='count').fillna(0)

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    fig = px.imshow(
        heatmap_pivot,
        title="Activity by Day of Week and Hour",
        labels={'x': 'Hour', 'y': 'Day', 'color': 'Count'},
        color_continuous_scale='YlOrRd'
    )
    st.plotly_chart(fig, width='stretch')


def render_detailed_breakdown(df: pd.DataFrame):
    """Render detailed breakdown combining all dimensions."""
    st.subheader("Detailed Breakdown")

    # Multi-dimensional grouping
    breakdown = df.groupby(['proto', 'action', 'port_category'], observed=False).size().reset_index(name='count')

    fig = px.treemap(
        breakdown,
        path=['proto', 'action', 'port_category'],
        values='count',
        title="Hierarchical View: Protocol > Action > Port Range",
        color='count',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, width='stretch')

    # Top destination ports
    st.subheader("Top 20 Destination Ports")
    top_ports = df.groupby(['portdst', 'action'], observed=False).size().reset_index(name='count')
    top_ports_total = top_ports.groupby('portdst')['count'].sum().nlargest(20).index
    top_ports = top_ports[top_ports['portdst'].isin(top_ports_total)]

    fig = px.bar(
        top_ports,
        x='portdst',
        y='count',
        color='action',
        title="Top 20 Destination Ports",
        color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
    )
    fig.update_layout(xaxis_title="Destination Port", yaxis_title="Count")
    st.plotly_chart(fig, width='stretch')
