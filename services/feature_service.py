# -*- coding: utf-8 -*-
"""
Feature Service
---------------
High-level feature extraction service.
"""

import pandas as pd
from typing import List, Optional, Dict, Any

from core.interfaces import FeatureSet
from core.config import get_config
from features import (
    CourseFeatureExtractor,
    FullFeatureExtractor,
    FeatureStore,
    Scaler,
    FeatureSelector
)


class FeatureService:
    """
    Service for feature extraction and management.
    """

    def __init__(self):
        self._store = FeatureStore()
        self._scaler: Optional[Scaler] = None

    @property
    def store(self) -> FeatureStore:
        """Get feature store."""
        return self._store

    def extract_course_features(
        self,
        df: pd.DataFrame,
        ip_col: str = "ipsrc",
        dst_col: str = "ipdst",
        port_col: str = "portdst",
        action_col: str = "action",
        save_as: str = None
    ) -> FeatureSet:
        """
        Extract course-standard features (11 features).

        Args:
            df: Raw log DataFrame
            ip_col: Source IP column
            dst_col: Destination IP column
            port_col: Port column
            action_col: Action column
            save_as: Optional name to save in store

        Returns:
            FeatureSet
        """
        extractor = CourseFeatureExtractor()
        feature_set = extractor.extract(
            df, ip_col, dst_col, port_col, action_col
        )

        if save_as:
            self._store.save(save_as, feature_set)

        return feature_set

    def extract_full_features(
        self,
        df: pd.DataFrame,
        ip_col: str = "ipsrc",
        dst_col: str = "ipdst",
        port_col: str = "portdst",
        action_col: str = "action",
        date_col: str = "date",
        include_time: bool = True,
        include_ratios: bool = True,
        include_stats: bool = True,
        remove_correlated: bool = False,
        corr_threshold: float = 0.95,
        save_as: str = None
    ) -> FeatureSet:
        """
        Extract full feature set.

        Args:
            df: Raw log DataFrame
            ip_col: Source IP column
            dst_col: Destination IP column
            port_col: Port column
            action_col: Action column
            date_col: Date column
            include_time: Include time features
            include_ratios: Include ratio features
            include_stats: Include statistical features
            remove_correlated: Remove highly correlated features
            corr_threshold: Correlation threshold for removal
            save_as: Optional name to save in store

        Returns:
            FeatureSet
        """
        extractor = FullFeatureExtractor(
            include_time=include_time,
            include_ratios=include_ratios,
            include_stats=include_stats,
            remove_correlated=remove_correlated,
            corr_threshold=corr_threshold
        )

        feature_set = extractor.extract(
            df, ip_col, dst_col, port_col, action_col,
            date_col if include_time else None
        )

        if save_as:
            self._store.save(save_as, feature_set)

        return feature_set

    def scale_features(
        self,
        df: pd.DataFrame,
        method: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features.

        Args:
            df: Feature DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit scaler

        Returns:
            Scaled DataFrame
        """
        if fit or self._scaler is None:
            self._scaler = Scaler(method)
            return self._scaler.fit_transform(df)
        else:
            return self._scaler.transform(df)

    def select_features(
        self,
        df: pd.DataFrame,
        preset: str = None,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Select subset of features.

        Args:
            df: Feature DataFrame
            preset: Preset name ('course', 'simple')
            features: List of feature names

        Returns:
            Selected features DataFrame
        """
        selector = FeatureSelector(features=features, preset=preset)
        return selector.select(df)

    def get_from_store(self, name: str) -> pd.DataFrame:
        """Get feature data from store."""
        return self._store.get_data(name)

    def list_stored(self) -> List[str]:
        """List stored feature sets."""
        return self._store.list()

    def export_features(
        self,
        name: str,
        path: str,
        format: str = 'csv'
    ) -> None:
        """
        Export features to file.

        Args:
            name: Feature set name in store
            path: Output path
            format: Output format ('csv', 'xlsx')
        """
        if format == 'csv':
            self._store.export_csv(name, path)
        elif format == 'xlsx':
            self._store.export_excel(name, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_feature_info(self, name: str) -> Dict[str, Any]:
        """Get metadata for stored features."""
        return self._store.get_metadata(name)
