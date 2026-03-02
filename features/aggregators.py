# -*- coding: utf-8 -*-
"""
Feature Aggregators
-------------------
Aggregate raw log data into features grouped by various keys.
"""

import pandas as pd
import numpy as np
from typing import List, Set, Optional, Dict, Any

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
        result['nombre'] = gb.size()

        if dst_col in df.columns:
            result['cnbripdst'] = gb[dst_col].nunique()

        if port_col in df.columns:
            result['cnportdst'] = gb[port_col].nunique()

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
        """Add permit/deny based features."""
        # Permit/Deny counts
        action_counts = df.groupby([ip_col, action_col]).size().unstack(fill_value=0)

        if 'Permit' in action_counts.columns:
            result['permit'] = action_counts['Permit']
        else:
            result['permit'] = 0

        if 'Deny' in action_counts.columns:
            result['deny'] = action_counts['Deny']
        else:
            result['deny'] = 0

        # Port-based features
        if port_col in df.columns:
            result['inf1024permit'] = gb.apply(
                lambda x: ((x[action_col] == 'Permit') &
                          (x[port_col] <= self.port_threshold)).sum(),
                include_groups=False
            )
            result['sup1024permit'] = gb.apply(
                lambda x: ((x[action_col] == 'Permit') &
                          (x[port_col] > self.port_threshold)).sum(),
                include_groups=False
            )
            result['adminpermit'] = gb.apply(
                lambda x: ((x[action_col] == 'Permit') &
                          (x[port_col].isin(self.admin_ports))).sum(),
                include_groups=False
            )
            result['inf1024deny'] = gb.apply(
                lambda x: ((x[action_col] == 'Deny') &
                          (x[port_col] <= self.port_threshold)).sum(),
                include_groups=False
            )
            result['sup1024deny'] = gb.apply(
                lambda x: ((x[action_col] == 'Deny') &
                          (x[port_col] > self.port_threshold)).sum(),
                include_groups=False
            )
            result['admindeny'] = gb.apply(
                lambda x: ((x[action_col] == 'Deny') &
                          (x[port_col].isin(self.admin_ports))).sum(),
                include_groups=False
            )

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

        # Time span
        result['time_span_seconds'] = gb[date_col].apply(
            lambda x: (x.max() - x.min()).total_seconds()
        )

        # Access rate
        counts = gb.size()
        result['access_rate'] = counts / (result['time_span_seconds'] + 1)

        # Hour-based features
        df['hour'] = df[date_col].dt.hour
        result['unique_hours'] = df.groupby(group_col)['hour'].nunique()

        # Night activity (22h-6h)
        df['is_night'] = df['hour'].apply(lambda h: 1 if h >= 22 or h < 6 else 0)
        result['night_ratio'] = df.groupby(group_col)['is_night'].mean()

        # Weekend activity
        df['is_weekend'] = df[date_col].dt.dayofweek >= 5
        result['weekend_ratio'] = df.groupby(group_col)['is_weekend'].mean()

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

        result[f'{prefix}_mean'] = gb[numeric_col].mean()
        result[f'{prefix}_std'] = gb[numeric_col].std().fillna(0)
        result[f'{prefix}_min'] = gb[numeric_col].min()
        result[f'{prefix}_max'] = gb[numeric_col].max()
        result[f'{prefix}_range'] = result[f'{prefix}_max'] - result[f'{prefix}_min']

        return result
