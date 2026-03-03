# -*- coding: utf-8 -*-
"""Statistics Dashboard Page - Key metrics and top entities"""

import streamlit as st
import pandas as pd
import plotly.express as px

from app.state import get_state
from utils.helpers import normalize_log_columns


# Internal network prefix (destination server range)
# This is the IP range of the protected server(s) in the firewall logs
INTERNAL_PREFIX = "159.84."


def is_external_ip(ip: str) -> bool:
    """Check if an IP is external (not in the internal network range)."""
    return not str(ip).startswith(INTERNAL_PREFIX)


def get_data_from_state():
    """Get data from AppState (uploaded via Data Upload page)."""
    state = get_state()
    if not state.has_raw_data():
        return None
    return normalize_log_columns(state.raw_data.copy())


def render():
    """Render statistics dashboard page."""
    st.title("Statistics Dashboard")
    st.markdown("Key metrics, top entities, and summary statistics from firewall logs.")

    # Get data from AppState (uploaded via Data Upload page)
    df = get_data_from_state()

    if df is None:
        st.warning("No data loaded. Please upload data in the **Data Upload** page first.")
        return

    # Convert port columns to numeric
    if 'portdst' in df.columns:
        df['portdst'] = pd.to_numeric(df['portdst'], errors='coerce')
    if 'portsrc' in df.columns:
        df['portsrc'] = pd.to_numeric(df['portsrc'], errors='coerce')

    st.info(f"Analyzing {len(df):,} log entries")

    # Summary metrics
    render_summary_metrics(df)

    st.markdown("---")

    # Main content in columns
    col1, col2 = st.columns(2)

    with col1:
        render_top_source_ips(df)
        st.markdown("---")
        render_top_ports_permit(df)

    with col2:
        render_external_ips(df)
        st.markdown("---")
        render_top_rules(df)

    st.markdown("---")
    render_additional_stats(df)


def render_summary_metrics(df: pd.DataFrame):
    """Render summary metric cards."""
    st.header("Summary Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    total_flows = len(df)
    unique_src_ips = df['ipsrc'].nunique()
    unique_dst_ips = df['ipdst'].nunique()
    unique_ports = df['portdst'].nunique() if 'portdst' in df.columns else 0
    unique_rules = df['regle'].nunique() if 'regle' in df.columns else 0

    col1.metric("Total Flows", f"{total_flows:,}")
    col2.metric("Unique Source IPs", f"{unique_src_ips:,}")
    col3.metric("Unique Dest IPs", f"{unique_dst_ips:,}")
    col4.metric("Unique Dest Ports", f"{unique_ports:,}")
    col5.metric("Unique Rules", f"{unique_rules:,}")

    # Second row
    col1, col2, col3, col4, col5 = st.columns(5)

    permits = (df['action'] == 'PERMIT').sum()
    denies = (df['action'] == 'DENY').sum()
    tcp_flows = (df['proto'] == 'TCP').sum()
    udp_flows = (df['proto'] == 'UDP').sum()
    _ext_mask = ~df['ipsrc'].astype(str).str.startswith(INTERNAL_PREFIX)
    external_ips = _ext_mask.sum()

    col1.metric("PERMIT", f"{permits:,}")
    col2.metric("DENY", f"{denies:,}")
    col3.metric("TCP Flows", f"{tcp_flows:,}")
    col4.metric("UDP Flows", f"{udp_flows:,}")
    col5.metric("External Src IPs", f"{df.loc[_ext_mask, 'ipsrc'].nunique():,}")


def render_top_source_ips(df: pd.DataFrame):
    """Render TOP 5 most active source IPs."""
    st.subheader("TOP 5 Most Active Source IPs")

    top_ips = df.groupby('ipsrc').agg(
        total_flows=('ipsrc', 'count'),
        permits=('action', lambda x: (x == 'PERMIT').sum()),
        denies=('action', lambda x: (x == 'DENY').sum()),
        unique_dst_ports=('portdst', 'nunique')
    ).nlargest(5, 'total_flows').reset_index()

    # Display as table
    st.dataframe(
        top_ips.rename(columns={
            'ipsrc': 'Source IP',
            'total_flows': 'Total Flows',
            'permits': 'PERMIT',
            'denies': 'DENY',
            'unique_dst_ports': 'Unique Ports'
        }),
        width='stretch',
        hide_index=True
    )

    # Bar chart
    fig = px.bar(
        top_ips,
        x='ipsrc',
        y='total_flows',
        title="Top 5 Source IPs by Flow Count",
        color='total_flows',
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_title="Source IP", yaxis_title="Flow Count", showlegend=False)
    st.plotly_chart(fig, width='stretch')


def render_top_ports_permit(df: pd.DataFrame):
    """Render TOP 10 ports < 1024 with PERMIT action."""
    st.subheader("TOP 10 Ports < 1024 with PERMIT")

    # Filter for well-known ports and PERMIT action
    well_known = df[(df['portdst'] < 1024) & (df['action'] == 'PERMIT')]

    top_ports = well_known.groupby('portdst').agg(
        count=('portdst', 'count'),
        unique_src_ips=('ipsrc', 'nunique')
    ).nlargest(10, 'count').reset_index()

    # Add common port names
    port_names = {
        21: 'FTP', 22: 'SSH', 23: 'Telnet', 25: 'SMTP', 53: 'DNS',
        80: 'HTTP', 110: 'POP3', 143: 'IMAP', 443: 'HTTPS', 445: 'SMB',
        993: 'IMAPS', 995: 'POP3S', 3389: 'RDP'
    }
    top_ports['service'] = top_ports['portdst'].map(lambda p: port_names.get(int(p), ''))

    # Display as table
    display_df = top_ports.rename(columns={
        'portdst': 'Port',
        'count': 'PERMIT Count',
        'unique_src_ips': 'Unique Sources',
        'service': 'Service'
    })
    st.dataframe(display_df, width='stretch', hide_index=True)

    # Bar chart
    fig = px.bar(
        top_ports,
        x='portdst',
        y='count',
        title="Top 10 Well-Known Ports with PERMIT",
        color='count',
        color_continuous_scale='Greens',
        text='service'
    )
    fig.update_layout(xaxis_title="Destination Port", yaxis_title="PERMIT Count", showlegend=False)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, width='stretch')


@st.fragment
def render_external_ips(df: pd.DataFrame):
    """Render list of external IPs (not in the internal network range)."""
    st.subheader(f"External Source IPs (not {INTERNAL_PREFIX}x.x)")

    # Filter external IPs
    external_df = df[~df['ipsrc'].astype(str).str.startswith(INTERNAL_PREFIX)]

    external_stats = external_df.groupby('ipsrc').agg(
        total_flows=('ipsrc', 'count'),
        permits=('action', lambda x: (x == 'PERMIT').sum()),
        denies=('action', lambda x: (x == 'DENY').sum()),
        unique_dst_ips=('ipdst', 'nunique'),
        unique_ports=('portdst', 'nunique')
    ).reset_index()

    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["total_flows", "denies", "unique_ports"],
        format_func=lambda x: {"total_flows": "Total Flows", "denies": "DENY Count", "unique_ports": "Unique Ports"}[x]
    )

    external_stats = external_stats.nlargest(20, sort_by)

    st.write(f"Showing top 20 of {(~df['ipsrc'].astype(str).str.startswith(INTERNAL_PREFIX)).sum():,} external flows / {df[~df['ipsrc'].astype(str).str.startswith(INTERNAL_PREFIX)]['ipsrc'].nunique():,} unique IPs")

    # Display as table
    st.dataframe(
        external_stats.rename(columns={
            'ipsrc': 'External IP',
            'total_flows': 'Total Flows',
            'permits': 'PERMIT',
            'denies': 'DENY',
            'unique_dst_ips': 'Unique Dests',
            'unique_ports': 'Unique Ports'
        }),
        width='stretch',
        hide_index=True
    )

    # Scatter plot: flows vs unique ports (potential scanners)
    fig = px.scatter(
        external_stats,
        x='total_flows',
        y='unique_ports',
        size='denies',
        hover_name='ipsrc',
        title="External IPs: Flows vs Unique Ports",
        color='denies',
        color_continuous_scale='Reds'
    )
    fig.update_layout(xaxis_title="Total Flows", yaxis_title="Unique Destination Ports")
    st.plotly_chart(fig, width='stretch')


def render_top_rules(df: pd.DataFrame):
    """Render top firewall rules."""
    st.subheader("Top 10 Firewall Rules")

    top_rules = df.groupby('regle').agg(
        total=('regle', 'count'),
        permits=('action', lambda x: (x == 'PERMIT').sum()),
        denies=('action', lambda x: (x == 'DENY').sum())
    ).nlargest(10, 'total').reset_index()

    # Determine dominant action
    top_rules['dominant_action'] = top_rules.apply(
        lambda r: 'PERMIT' if r['permits'] > r['denies'] else 'DENY', axis=1
    )

    # Display as table
    st.dataframe(
        top_rules.rename(columns={
            'regle': 'Rule',
            'total': 'Total Hits',
            'permits': 'PERMIT',
            'denies': 'DENY',
            'dominant_action': 'Dominant'
        }),
        width='stretch',
        hide_index=True
    )

    # Bar chart
    fig = px.bar(
        top_rules,
        x='regle',
        y='total',
        color='dominant_action',
        title="Top 10 Rules by Hit Count",
        color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
    )
    fig.update_layout(xaxis_title="Rule Number", yaxis_title="Total Hits")
    st.plotly_chart(fig, width='stretch')


def render_additional_stats(df: pd.DataFrame):
    """Render additional statistics."""
    st.header("Additional Statistics")

    tabs = st.tabs(["Port Distribution", "Protocol Breakdown", "Interface Stats"])

    with tabs[0]:
        # Port distribution histogram
        col1, col2 = st.columns(2)

        with col1:
            # Well-known ports breakdown
            well_known = df[df['portdst'] < 1024]
            port_counts = well_known['portdst'].value_counts().head(20)

            fig = px.bar(
                x=port_counts.index.astype(str),
                y=port_counts.values,
                title="Top 20 Well-Known Destination Ports (< 1024)",
                labels={'x': 'Port', 'y': 'Count'}
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Port range distribution
            df['port_range'] = pd.cut(
                df['portdst'],
                bins=[0, 1023, 49151, 65535],
                labels=['Well-known (0-1023)', 'Registered (1024-49151)', 'Dynamic (49152-65535)']
            )
            range_counts = df['port_range'].value_counts()

            fig = px.pie(
                values=range_counts.values,
                names=range_counts.index,
                title="Traffic by Port Range",
                hole=0.4
            )
            st.plotly_chart(fig, width='stretch')

    with tabs[1]:
        # Protocol breakdown
        proto_action = df.groupby(['proto', 'action'], observed=False).size().reset_index(name='count')

        fig = px.sunburst(
            proto_action,
            path=['proto', 'action'],
            values='count',
            title="Protocol and Action Hierarchy",
            color='action',
            color_discrete_map={'PERMIT': '#2ecc71', 'DENY': '#e74c3c'}
        )
        st.plotly_chart(fig, width='stretch')

    with tabs[2]:
        # Interface statistics
        if 'interface_in' in df.columns:
            interface_stats = df.groupby('interface_in').agg(
                total=('interface_in', 'count'),
                permits=('action', lambda x: (x == 'PERMIT').sum()),
                denies=('action', lambda x: (x == 'DENY').sum())
            ).reset_index()

            st.dataframe(
                interface_stats.rename(columns={
                    'interface_in': 'Interface',
                    'total': 'Total Flows',
                    'permits': 'PERMIT',
                    'denies': 'DENY'
                }),
                width='stretch',
                hide_index=True
            )

            fig = px.bar(
                interface_stats,
                x='interface_in',
                y='total',
                title="Flows by Interface",
                color='total',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, width='stretch')
