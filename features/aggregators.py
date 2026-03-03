# -*- coding: utf-8 -*-
"""
Feature Aggregators
-------------------
Aggregate raw log data into features grouped by various keys.
"""

import pandas as pd
from typing import Set

from core.config import get_config
from core.exceptions import FeatureExtractionError


class IPAggregator:
    """
    Aggregate log data by IP address.
    Creates features matching the course format (Seance 3a).
    """

    def __init__(
        self,
        group_by: str = None,
        admin_ports: Set[int] = None,
        port_threshold: int = None
    ):
        config = get_config()
        self.group_by = group_by or config.features.default_group_by
        self.admin_ports = admin_ports or config.features.admin_ports
        self.port_threshold = port_threshold or config.features.port_threshold

    def aggregate(
        self,
        df: pd.DataFrame,
        ip_col: str = "ipsrc",
        dst_col: str = "ipdst",
        port_col: str = "portdst",
        action_col: str = "action"
    ) -> pd.DataFrame:
        """
        Aggregate raw logs by IP.

        Args:
            df: Raw log DataFrame
            ip_col: Source IP column
            dst_col: Destination IP column
            port_col: Port column
            action_col: Action column (Permit/Deny)

        Returns:
            Aggregated DataFrame with features per IP
        """
        if ip_col not in df.columns:
            raise FeatureExtractionError(
                f"Column '{ip_col}' not found in DataFrame",
                feature_name=ip_col
            )

        gb = df.groupby(ip_col)

        # Build result DataFrame
        result = pd.DataFrame(index=gb.groups.keys())
        result.index.name = ip_col

        # Basic counts
        result['total_flows'] = gb.size()

        if dst_col in df.columns:
            result['unique_dst_ips'] = gb[dst_col].nunique()

        if port_col in df.columns:
            result['unique_dst_ports'] = gb[port_col].nunique()

        # Action-based features
        if action_col in df.columns:
            result = self._add_action_features(
                result, df, gb, ip_col, port_col, action_col
            )

        return result.fillna(0).astype(int)

    def _add_action_features(
        self,
        result: pd.DataFrame,
        df: pd.DataFrame,
        gb,
        ip_col: str,
        port_col: str,
        action_col: str
    ) -> pd.DataFrame:
        """Add permit/deny based features (fully vectorized)."""
        # Normalise action strings to uppercase so 'Permit', 'PERMIT', 'permit'
        # all map correctly (normalize_log_columns uppercases, but raw data may not).
        action_upper = df[action_col].str.upper()

        # Boolean masks aligned to df's row index — no Python-level loops.
        is_permit = action_upper.isin({'PERMIT', 'ALLOW', 'ACCEPT'})
        is_deny   = action_upper.isin({'DENY', 'DROP', 'REJECT', 'BLOCK'})

        groups = df[ip_col]

        result['permit'] = is_permit.groupby(groups).sum().astype(int)
        result['deny']   = is_deny.groupby(groups).sum().astype(int)

        # Port-based features — all vectorized via boolean Series + groupby.sum()
        if port_col in df.columns:
            port  = pd.to_numeric(df[port_col], errors='coerce').fillna(0)
            lo    = port <= self.port_threshold
            hi    = port >  self.port_threshold
            admin = port.isin(self.admin_ports)

            def _gsum(mask: pd.Series) -> pd.Series:
                return mask.groupby(groups).sum().astype(int)

            result['permit_low_port']  = _gsum(is_permit & lo)
            result['permit_high_port'] = _gsum(is_permit & hi)
            result['permit_admin']     = _gsum(is_permit & admin)
            result['deny_low_port']    = _gsum(is_deny & lo)
            result['deny_high_port']   = _gsum(is_deny & hi)
            result['deny_admin']       = _gsum(is_deny & admin)
        else:
            for col in ('permit_low_port', 'permit_high_port', 'permit_admin',
                        'deny_low_port',   'deny_high_port',   'deny_admin'):
                result[col] = 0

        return result


class TimeAggregator:
    """
    Aggregate time-based features from log data.
    """

    def aggregate(
        self,
        df: pd.DataFrame,
        group_col: str = "ipsrc",
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Create time-based features.

        Args:
            df: DataFrame with datetime column
            group_col: Column to group by
            date_col: Datetime column

        Returns:
            DataFrame with time features
        """
        if date_col not in df.columns:
            raise FeatureExtractionError(
                f"Date column '{date_col}' not found",
                feature_name=date_col
            )

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        gb = df.groupby(group_col)
        result = pd.DataFrame(index=gb.groups.keys())
        result.index.name = group_col

        # Time span — vectorized (max - min per group, then .dt.total_seconds())
        result['time_span_seconds'] = (
            (gb[date_col].max() - gb[date_col].min()).dt.total_seconds()
        )

        # Access rate
        counts = gb.size()
        result['access_rate'] = counts / (result['time_span_seconds'] + 1)

        # Hour-based features — avoid per-element .apply()
        hour = df[date_col].dt.hour
        result['unique_hours'] = hour.groupby(df[group_col]).nunique()

        # Night activity (22h-6h) — vectorized boolean
        is_night = (hour >= 22) | (hour < 6)
        result['night_ratio'] = is_night.groupby(df[group_col]).mean()

        # Weekend activity — vectorized
        is_weekend = df[date_col].dt.dayofweek >= 5
        result['weekend_ratio'] = is_weekend.groupby(df[group_col]).mean()

        return result


class StatisticalAggregator:
    """
    Compute statistical features from numeric columns.
    """

    def aggregate(
        self,
        df: pd.DataFrame,
        group_col: str = "ipsrc",
        numeric_col: str = "portdst"
    ) -> pd.DataFrame:
        """
        Create statistical features.

        Args:
            df: Input DataFrame
            group_col: Column to group by
            numeric_col: Numeric column for statistics

        Returns:
            DataFrame with statistical features
        """
        gb = df.groupby(group_col)
        result = pd.DataFrame(index=gb.groups.keys())
        result.index.name = group_col

        prefix = numeric_col

        result[f'avg_dst_port'] = gb[numeric_col].mean()
        result[f'port_std'] = gb[numeric_col].std().fillna(0)
        result[f'port_min'] = gb[numeric_col].min()
        result[f'port_max'] = gb[numeric_col].max()
        result[f'port_range'] = result[f'port_max'] - result[f'port_min']

        return result
