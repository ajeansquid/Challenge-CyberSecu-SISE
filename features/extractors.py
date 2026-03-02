# -*- coding: utf-8 -*-
"""
Feature Extractors
------------------
High-level feature extraction pipelines.
"""

import pandas as pd
from typing import List, Optional, Tuple
import logging

from core.interfaces import FeatureExtractor, FeatureSet
from core.config import get_config
from .aggregators import IPAggregator, TimeAggregator, StatisticalAggregator
from .transformers import RatioTransformer

logger = logging.getLogger(__name__)


def _remove_constant_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with zero variance (constant values).

    Args:
        df: Feature DataFrame

    Returns:
        Tuple of (cleaned DataFrame, list of removed column names)
    """
    variances = df.var()
    constant_cols = variances[variances == 0].index.tolist()

    if constant_cols:
        logger.warning(f"Removing constant features (zero variance): {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df, constant_cols


class CourseFeatureExtractor(FeatureExtractor):
    """
    Extract features matching the course format (11 features).
    """

    FEATURE_NAMES = [
        'total_flows', 'unique_dst_ips', 'unique_dst_ports',
        'permit', 'permit_low_port', 'permit_high_port', 'permit_admin',
        'deny', 'deny_low_port', 'deny_high_port', 'deny_admin'
    ]

    def __init__(self):
        self._aggregator = IPAggregator()

    @property
    def name(self) -> str:
        return "course_features"

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(
        self,
        df: pd.DataFrame,
        ip_col: str = "ipsrc",
        dst_col: str = "ipdst",
        port_col: str = "portdst",
        action_col: str = "action"
    ) -> FeatureSet:
        """
        Extract course-standard features.

        Args:
            df: Raw log DataFrame
            ip_col: Source IP column
            dst_col: Destination IP column
            port_col: Port column
            action_col: Action column

        Returns:
            FeatureSet with 11 course features (minus any constant features)
        """
        features = self._aggregator.aggregate(
            df, ip_col, dst_col, port_col, action_col
        )

        # Ensure we have all expected columns
        for col in self.FEATURE_NAMES:
            if col not in features.columns:
                features[col] = 0

        # Reorder columns
        features = features[self.FEATURE_NAMES]

        # Remove constant features
        features, removed = _remove_constant_features(features)
        final_feature_names = [f for f in self.FEATURE_NAMES if f not in removed]

        return FeatureSet(
            data=features,
            feature_names=final_feature_names,
            index_column=ip_col,
            metadata={
                'extractor': self.name,
                'removed_constant_features': removed
            }
        )


class FullFeatureExtractor(FeatureExtractor):
    """
    Extract comprehensive feature set including time and statistical features.
    """

    def __init__(
        self,
        include_time: bool = True,
        include_ratios: bool = True,
        include_stats: bool = True
    ):
        self._ip_aggregator = IPAggregator()
        self._time_aggregator = TimeAggregator()
        self._stat_aggregator = StatisticalAggregator()
        self._ratio_transformer = RatioTransformer()

        self.include_time = include_time
        self.include_ratios = include_ratios
        self.include_stats = include_stats

        self._feature_names: List[str] = []

    @property
    def name(self) -> str:
        return "full_features"

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    def extract(
        self,
        df: pd.DataFrame,
        ip_col: str = "ipsrc",
        dst_col: str = "ipdst",
        port_col: str = "portdst",
        action_col: str = "action",
        date_col: Optional[str] = "date"
    ) -> FeatureSet:
        """
        Extract full feature set.

        Args:
            df: Raw log DataFrame
            ip_col: Source IP column
            dst_col: Destination IP column
            port_col: Port column
            action_col: Action column
            date_col: Date column (optional)

        Returns:
            FeatureSet with all features
        """
        # Base features
        result = self._ip_aggregator.aggregate(
            df, ip_col, dst_col, port_col, action_col
        )

        # Time features
        if self.include_time and date_col and date_col in df.columns:
            time_features = self._time_aggregator.aggregate(
                df, ip_col, date_col
            )
            result = result.join(time_features)

        # Ratio features
        if self.include_ratios:
            result = self._ratio_transformer.transform(result)

        # Statistical features
        if self.include_stats and port_col in df.columns:
            stat_features = self._stat_aggregator.aggregate(
                df, ip_col, port_col
            )
            result = result.join(stat_features)

        # Remove constant features
        result, removed = _remove_constant_features(result)
        self._feature_names = list(result.columns)

        return FeatureSet(
            data=result,
            feature_names=self._feature_names,
            index_column=ip_col,
            metadata={
                'extractor': self.name,
                'include_time': self.include_time,
                'include_ratios': self.include_ratios,
                'include_stats': self.include_stats,
                'removed_constant_features': removed
            }
        )


class SimpleFeatureExtractor(FeatureExtractor):
    """
    Extract simple features (3 features only).
    As recommended in course for cleaner models.
    """

    FEATURE_NAMES = ['total_flows', 'unique_dst_ips', 'unique_dst_ports']

    def __init__(self):
        self._aggregator = IPAggregator()

    @property
    def name(self) -> str:
        return "simple_features"

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(
        self,
        df: pd.DataFrame,
        ip_col: str = "ipsrc",
        dst_col: str = "ipdst",
        port_col: str = "portdst",
        **kwargs
    ) -> FeatureSet:
        """Extract simple features."""
        features = self._aggregator.aggregate(
            df, ip_col, dst_col, port_col, action_col=kwargs.get('action_col', 'action')
        )

        # Keep only simple features
        simple = features[['total_flows', 'unique_dst_ips', 'unique_dst_ports']]

        # Remove constant features
        simple, removed = _remove_constant_features(simple)
        final_feature_names = [f for f in self.FEATURE_NAMES if f not in removed]

        return FeatureSet(
            data=simple,
            feature_names=final_feature_names,
            index_column=ip_col,
            metadata={
                'extractor': self.name,
                'removed_constant_features': removed
            }
        )
