# -*- coding: utf-8 -*-
"""IP Visualization Page - Interactive scatter plots and IP details"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from app.state import get_state
from utils.helpers import normalize_log_columns


def get_data_from_state():
    """Get data from AppState (uploaded via Data Upload page)."""
    state = get_state()
    if not state.has_raw_data():
        return None
    df = normalize_log_columns(state.raw_data.copy())

    # Ensure date column is datetime
    date_col = 'date' if 'date' in df.columns else None
    if date_col and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    if 'portdst' in df.columns:
        df['portdst'] = pd.to_numeric(df['portdst'], errors='coerce')
    return df


@st.cache_data(show_spinner=False)
def compute_ip_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-source-IP statistics (cached)."""
    # Use pivot_table + value_counts instead of lambda aggs — much faster
    grp = df.groupby('ipsrc')

    totals = grp.size().rename('total_flows')
    permits = (df[df['action'] == 'PERMIT']
               .groupby('ipsrc').size().rename('permits'))
    denies  = (df[df['action'] == 'DENY']
               .groupby('ipsrc').size().rename('denies'))
    uniq_dst_ips   = grp['ipdst'].nunique().rename('unique_dst_ips') if 'ipdst' in df.columns else pd.Series(0, index=grp.groups, name='unique_dst_ips')
    uniq_dst_ports = grp['portdst'].nunique().rename('unique_dst_ports') if 'portdst' in df.columns else pd.Series(0, index=grp.groups, name='unique_dst_ports')
    uniq_rules     = grp['regle'].nunique().rename('unique_rules') if 'regle' in df.columns else pd.Series(0, index=grp.groups, name='unique_rules')

    stats = pd.concat(
        [totals, permits, denies, uniq_dst_ips, uniq_dst_ports, uniq_rules],
        axis=1
    ).fillna(0).reset_index()
    stats['permit_ratio'] = (stats['permits'] / stats['total_flows'] * 100).round(1)
    stats['deny_ratio']   = (stats['denies']  / stats['total_flows'] * 100).round(1)
    return stats


def render():
    """Render IP visualization page."""
    st.title("IP Visualization")
    st.markdown("Explore source IP behavior with interactive visualizations.")

    # Get data from AppState (uploaded via Data Upload page)
    df = get_data_from_state()

    if df is None:
        st.warning("No data loaded. Please upload data in the **Data Upload** page first.")
        return

    # Compute IP statistics (cached)
    with st.spinner("Computing IP statistics..."):
        ip_stats = compute_ip_stats(df)

    st.info(f"Analyzing {len(ip_stats):,} unique source IPs from {len(df):,} log entries")

    # Main visualization tabs
    tabs = st.tabs(["Scatter Plot", "IP Browser", "IP Details", "Comparison"])

    with tabs[0]:
        render_scatter_plot(ip_stats)

    with tabs[1]:
        render_ip_browser(ip_stats, df)

    with tabs[2]:
        render_ip_details(ip_stats, df)

    with tabs[3]:
        render_ip_comparison(ip_stats, df)


def render_scatter_plot(ip_stats: pd.DataFrame):
    """Render interactive scatter plot of IPs."""
    st.subheader("IP Behavior Scatter Plot")

    # Axis selection
    col1, col2, col3 = st.columns(3)

    numeric_cols = ['total_flows', 'permits', 'denies', 'unique_dst_ips', 'unique_dst_ports', 'unique_rules', 'permit_ratio', 'deny_ratio']
    col_labels = {
        'total_flows': 'Total Flows',
        'permits': 'PERMIT Count',
        'denies': 'DENY Count',
        'unique_dst_ips': 'Unique Dest IPs',
        'unique_dst_ports': 'Unique Dest Ports',
        'unique_rules': 'Unique Rules',
        'permit_ratio': 'PERMIT %',
        'deny_ratio': 'DENY %'
    }

    with col1:
        x_axis = st.selectbox("X Axis", numeric_cols, index=0, format_func=lambda x: col_labels[x])
    with col2:
        y_axis = st.selectbox("Y Axis", numeric_cols, index=4, format_func=lambda x: col_labels[x])
    with col3:
        color_by = st.selectbox("Color by", numeric_cols, index=2, format_func=lambda x: col_labels[x])

    # Size option
    size_by = st.selectbox("Size by", [None] + numeric_cols, format_func=lambda x: "None" if x is None else col_labels[x])

    # Filter options
    st.subheader("Filters")
    col1, col2 = st.columns(2)

    with col1:
        min_flows = st.slider("Minimum total flows", 1, int(ip_stats['total_flows'].max()), 1)
    with col2:
        top_n = st.slider("Show top N IPs", 10, min(1000, len(ip_stats)), 100)

    # Apply filters
    filtered = ip_stats[ip_stats['total_flows'] >= min_flows].nlargest(top_n, 'total_flows')

    st.write(f"Showing {len(filtered):,} IPs")

    # Create scatter plot
    fig = px.scatter(
        filtered,
        x=x_axis,
        y=y_axis,
        color=color_by,
        size=size_by if size_by else None,
        hover_name='ipsrc',
        hover_data=['total_flows', 'permits', 'denies', 'unique_dst_ports'],
        title=f"IP Behavior: {col_labels[x_axis]} vs {col_labels[y_axis]}",
        color_continuous_scale='RdYlGn_r'
    )

    fig.update_layout(
        xaxis_title=col_labels[x_axis],
        yaxis_title=col_labels[y_axis],
        height=600
    )

    st.plotly_chart(fig, width='stretch')

    # Potential scanners highlight
    st.subheader("Potential Port Scanners")
    st.markdown("IPs with high unique destination ports relative to their flow count may be scanning.")

    # Calculate scan ratio
    filtered['scan_score'] = filtered['unique_dst_ports'] / filtered['total_flows']
    potential_scanners = filtered.nlargest(10, 'scan_score')

    st.dataframe(
        potential_scanners[['ipsrc', 'total_flows', 'unique_dst_ports', 'denies', 'scan_score']].rename(columns={
            'ipsrc': 'Source IP',
            'total_flows': 'Total Flows',
            'unique_dst_ports': 'Unique Ports',
            'denies': 'DENY Count',
            'scan_score': 'Scan Score'
        }),
        width='stretch',
        hide_index=True
    )


def render_ip_browser(ip_stats: pd.DataFrame, df: pd.DataFrame):
    """Render IP browser with slider."""
    st.subheader("IP Browser")

    # Sort options
    sort_by = st.selectbox(
        "Sort IPs by",
        ['total_flows', 'denies', 'unique_dst_ports', 'unique_dst_ips'],
        format_func=lambda x: {
            'total_flows': 'Total Flows',
            'denies': 'DENY Count',
            'unique_dst_ports': 'Unique Ports',
            'unique_dst_ips': 'Unique Destinations'
        }[x]
    )

    sorted_stats = ip_stats.nlargest(100, sort_by)
    ip_list = sorted_stats['ipsrc'].tolist()

    if not ip_list:
        st.warning("No IPs to display.")
        return

    # Slider to browse IPs
    st.write("Use the slider to browse through IPs:")
    ip_index = st.slider("IP Index", 0, len(ip_list) - 1, 0)

    selected_ip = ip_list[ip_index]

    # Display IP info card
    ip_info = sorted_stats[sorted_stats['ipsrc'] == selected_ip].iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### {selected_ip}")
        st.metric("Total Flows", f"{ip_info['total_flows']:,}")
        st.metric("PERMIT", f"{ip_info['permits']:,}", f"{ip_info['permit_ratio']}%")
        st.metric("DENY", f"{ip_info['denies']:,}", f"{ip_info['deny_ratio']}%")
        st.metric("Unique Dest Ports", f"{ip_info['unique_dst_ports']:,}")
        st.metric("Unique Dest IPs", f"{ip_info['unique_dst_ips']:,}")

    with col2:
        # Get flows for this IP (indexed lookup)
        ip_flows = df.loc[df['ipsrc'] == selected_ip]

        # Port distribution for this IP
        port_counts = ip_flows['portdst'].value_counts().head(10)

        fig = px.bar(
            x=port_counts.index.astype(str),
            y=port_counts.values,
            title=f"Top 10 Destination Ports for {selected_ip}",
            labels={'x': 'Destination Port', 'y': 'Count'}
        )
        st.plotly_chart(fig, width='stretch')

    # Action timeline for this IP ('date' already datetime from load)
    ip_flows = ip_flows.dropna(subset=['date']).rename(columns={'date': 'datetime'})

    if len(ip_flows) > 0:
        ip_flows['hour'] = ip_flows['datetime'].dt.floor('h')
        timeline = ip_flows.groupby(['hour', 'action']).size().reset_index(name='count')

        fig = px.line(
            timeline,
            x='hour',
            y='count',
            color='action',
            title=f"Activity Timeline for {selected_ip}",
            color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
        )
        st.plotly_chart(fig, width='stretch')


def render_ip_details(ip_stats: pd.DataFrame, df: pd.DataFrame):
    """Render detailed view for a specific IP."""
    st.subheader("IP Details Lookup")

    # IP search
    selected_ip = st.text_input("Enter IP address to analyze", placeholder="e.g., 192.168.1.1")

    if selected_ip:
        if selected_ip not in ip_stats['ipsrc'].values:
            st.error(f"IP {selected_ip} not found in the logs.")
            return

        ip_flows = df.loc[df['ipsrc'] == selected_ip]

        st.success(f"Found {len(ip_flows):,} flows from {selected_ip}")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flows", f"{len(ip_flows):,}")
        col2.metric("PERMIT", f"{(ip_flows['action'] == 'PERMIT').sum():,}")
        col3.metric("DENY", f"{(ip_flows['action'] == 'DENY').sum():,}")
        col4.metric("Unique Ports", f"{ip_flows['portdst'].nunique():,}")

        # Tabs for different views
        detail_tabs = st.tabs(["Destinations", "Ports", "Rules", "Timeline", "Raw Data"])

        with detail_tabs[0]:
            # Destination IPs
            dst_stats = ip_flows.groupby('ipdst').agg(
                count=('ipdst', 'count'),
                permits=('action', lambda x: (x == 'PERMIT').sum()),
                denies=('action', lambda x: (x == 'DENY').sum())
            ).nlargest(20, 'count').reset_index()

            fig = px.bar(
                dst_stats,
                x='ipdst',
                y='count',
                color='permits',
                title="Top 20 Destination IPs",
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, width='stretch')
            st.dataframe(dst_stats, width='stretch', hide_index=True)

        with detail_tabs[1]:
            # Ports analysis
            port_stats = ip_flows.groupby('portdst').agg(
                count=('portdst', 'count'),
                permits=('action', lambda x: (x == 'PERMIT').sum()),
                denies=('action', lambda x: (x == 'DENY').sum())
            ).nlargest(20, 'count').reset_index()

            fig = px.bar(
                port_stats,
                x='portdst',
                y=['permits', 'denies'],
                title="Top 20 Destination Ports",
                barmode='stack',
                color_discrete_map={'permits': '#2ecc71', 'denies': '#e74c3c'}
            )
            st.plotly_chart(fig, width='stretch')

        with detail_tabs[2]:
            # Rules triggered
            rule_stats = ip_flows.groupby('regle').agg(
                count=('regle', 'count'),
                action=('action', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A')
            ).nlargest(10, 'count').reset_index()

            fig = px.pie(
                rule_stats,
                values='count',
                names='regle',
                title="Rules Triggered",
                hole=0.4
            )
            st.plotly_chart(fig, width='stretch')
            st.dataframe(rule_stats, width='stretch', hide_index=True)

        with detail_tabs[3]:
            # Timeline ('date' already datetime from load)
            ip_flows_ts = ip_flows.dropna(subset=['date']).rename(columns={'date': 'datetime'})

            if len(ip_flows_ts) > 0:
                ip_flows_ts['period'] = ip_flows_ts['datetime'].dt.floor('h')
                timeline = ip_flows_ts.groupby(['period', 'action']).size().reset_index(name='count')

                fig = px.area(
                    timeline,
                    x='period',
                    y='count',
                    color='action',
                    title="Activity Over Time",
                    color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
                )
                st.plotly_chart(fig, width='stretch')

        with detail_tabs[4]:
            # Raw data sample
            st.write("Sample of raw flows (first 100):")
            st.dataframe(ip_flows.head(100), width='stretch', hide_index=True)


def render_ip_comparison(ip_stats: pd.DataFrame, df: pd.DataFrame):
    """Compare multiple IPs side by side."""
    st.subheader("IP Comparison")

    # Multi-select for IPs
    top_ips = ip_stats.nlargest(100, 'total_flows')['ipsrc'].tolist()

    selected_ips = st.multiselect(
        "Select IPs to compare (max 5)",
        top_ips,
        default=top_ips[:2] if len(top_ips) >= 2 else top_ips,
        max_selections=5
    )

    if len(selected_ips) < 2:
        st.info("Select at least 2 IPs to compare.")
        return

    # Get stats for selected IPs
    comparison = ip_stats[ip_stats['ipsrc'].isin(selected_ips)]

    # Bar chart comparison
    fig = go.Figure()

    metrics = ['total_flows', 'permits', 'denies', 'unique_dst_ports']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=comparison['ipsrc'],
            y=comparison[metric],
            marker_color=colors[i]
        ))

    fig.update_layout(
        title="IP Comparison",
        barmode='group',
        xaxis_title="Source IP",
        yaxis_title="Count"
    )
    st.plotly_chart(fig, width='stretch')

    # Radar chart
    st.subheader("Behavioral Profile")

    # Normalize metrics for radar
    radar_metrics = ['total_flows', 'permits', 'denies', 'unique_dst_ports', 'unique_dst_ips']
    normalized = comparison[['ipsrc'] + radar_metrics].copy()

    for col in radar_metrics:
        max_val = normalized[col].max()
        if max_val > 0:
            normalized[col] = normalized[col] / max_val * 100

    fig = go.Figure()

    for _, row in normalized.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in radar_metrics],
            theta=[m.replace('_', ' ').title() for m in radar_metrics],
            fill='toself',
            name=row['ipsrc']
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Normalized Behavioral Profile"
    )
    st.plotly_chart(fig, width='stretch')

    # Detailed comparison table
    st.subheader("Detailed Statistics")
    st.dataframe(
        comparison.rename(columns={
            'ipsrc': 'Source IP',
            'total_flows': 'Total Flows',
            'permits': 'PERMIT',
            'denies': 'DENY',
            'unique_dst_ips': 'Unique Dests',
            'unique_dst_ports': 'Unique Ports',
            'unique_rules': 'Unique Rules',
            'permit_ratio': 'PERMIT %',
            'deny_ratio': 'DENY %'
        }),
        width='stretch',
        hide_index=True
    )
