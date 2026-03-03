# -*- coding: utf-8 -*-
"""
Feature Extractors
------------------
High-level feature extraction pipelines.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

from core.interfaces import FeatureExtractor, FeatureSet
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


def _remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features to reduce multicollinearity.

    For each pair of features with correlation > threshold, removes the one
    with lower variance (less information).

    Args:
        df: Feature DataFrame
        threshold: Correlation threshold (default 0.95)

    Returns:
        Tuple of (cleaned DataFrame, list of removed column names)
    """
    corr_matrix = df.corr().abs()

    # Get upper triangle (avoid duplicates)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation above threshold
    to_drop = set()
    for column in upper.columns:
        # Find correlated features
        correlated = upper[column][upper[column] > threshold].index.tolist()
        if correlated:
            # For each correlated pair, drop the one with lower variance
            col_var = df[column].var()
            for corr_col in correlated:
                if corr_col not in to_drop:
                    corr_var = df[corr_col].var()
                    # Keep the one with higher variance
                    if col_var < corr_var:
                        to_drop.add(column)
                    else:
                        to_drop.add(corr_col)

    removed = list(to_drop)
    if removed:
        logger.warning(
            f"Removing {len(removed)} highly correlated features "
            f"(|r| > {threshold}): {removed}"
        )
        df = df.drop(columns=removed)

    return df, removed


class CourseFeatureExtractor(FeatureExtractor):
    """
    Extract features matching the course format (11 features).
    """

    FEATURE_NAMES = [
        'total_flows', 'unique_dst_ips', 'unique_dst_ports',
        'permit', 'permit_low_port', 'permit_high_port', 'permit_admin',
        'deny', 'deny_low_port', 'deny_high_port', 'deny_admin'
    ]

    def __init__(self, remove_correlated: bool = False, corr_threshold: float = 0.95):
        """
        Args:
            remove_correlated: Whether to remove highly correlated features
            corr_threshold: Correlation threshold (only used if remove_correlated=True)
        """
        self._aggregator = IPAggregator()
        self.remove_correlated = remove_correlated
        self.corr_threshold = corr_threshold

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

        # Always remove constant features (objectively useless)
        features, removed_const = _remove_constant_features(features)

        # Optionally remove highly correlated features
        removed_corr = []
        if self.remove_correlated:
            features, removed_corr = _remove_correlated_features(features, threshold=self.corr_threshold)

        all_removed = removed_const + removed_corr
        final_feature_names = [f for f in self.FEATURE_NAMES if f not in all_removed]

        return FeatureSet(
            data=features,
            feature_names=final_feature_names,
            index_column=ip_col,
            metadata={
                'extractor': self.name,
                'removed_constant_features': removed_const,
                'removed_correlated_features': removed_corr
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
        include_stats: bool = True,
        remove_correlated: bool = False,
        corr_threshold: float = 0.95
    ):
        self._ip_aggregator = IPAggregator()
        self._time_aggregator = TimeAggregator()
        self._stat_aggregator = StatisticalAggregator()
        self._ratio_transformer = RatioTransformer()

        self.include_time = include_time
        self.include_ratios = include_ratios
        self.include_stats = include_stats
        self.remove_correlated = remove_correlated
        self.corr_threshold = corr_threshold

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

        # Always remove constant features
        result, removed_const = _remove_constant_features(result)

        # Optionally remove highly correlated features
        removed_corr = []
        if self.remove_correlated:
            result, removed_corr = _remove_correlated_features(result, threshold=self.corr_threshold)

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
                'removed_constant_features': removed_const,
                'removed_correlated_features': removed_corr
            }
        )


class SimpleFeatureExtractor(FeatureExtractor):
    """
    Extract simple features (3 features only).
    As recommended in course for cleaner models.
    """

    FEATURE_NAMES = ['total_flows', 'unique_dst_ips', 'unique_dst_ports']

    def __init__(self, remove_correlated: bool = False, corr_threshold: float = 0.95):
        """
        Args:
            remove_correlated: Whether to remove highly correlated features
            corr_threshold: Correlation threshold (only used if remove_correlated=True)
        """
        self._aggregator = IPAggregator()
        self.remove_correlated = remove_correlated
        self.corr_threshold = corr_threshold

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

        # Always remove constant features
        simple, removed_const = _remove_constant_features(simple)

        # Optionally remove highly correlated features
        removed_corr = []
        if self.remove_correlated:
            simple, removed_corr = _remove_correlated_features(simple, threshold=self.corr_threshold)

        all_removed = removed_const + removed_corr
        final_feature_names = [f for f in self.FEATURE_NAMES if f not in all_removed]

        return FeatureSet(
            data=simple,
            feature_names=final_feature_names,
            index_column=ip_col,
            metadata={
                'extractor': self.name,
                'removed_constant_features': removed_const,
                'removed_correlated_features': removed_corr
            }
        )
