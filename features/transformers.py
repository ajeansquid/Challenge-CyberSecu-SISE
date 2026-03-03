# -*- coding: utf-8 -*-
"""
Feature Transformers
--------------------
Transform and select features for ML.
"""

import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from core.exceptions import FeatureExtractionError


class Scaler:
    """
    Feature scaling transformer.
    Wraps sklearn scalers with DataFrame support.
    """

    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }

    def __init__(self, method: str = 'standard'):
        if method not in self.SCALERS:
            raise FeatureExtractionError(
                f"Unknown scaling method: {method}. "
                f"Available: {list(self.SCALERS.keys())}"
            )
        self.method = method
        self._scaler = self.SCALERS[method]()
        self._fitted = False
        self._columns: List[str] = []

    def fit(self, df: pd.DataFrame) -> 'Scaler':
        """Fit scaler on data."""
        self._columns = list(df.columns)
        self._scaler.fit(df.values)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self._fitted:
            raise FeatureExtractionError("Scaler not fitted. Call fit() first.")

        scaled = self._scaler.transform(df[self._columns].values)
        return pd.DataFrame(scaled, index=df.index, columns=self._columns)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data."""
        if not self._fitted:
            raise FeatureExtractionError("Scaler not fitted.")

        original = self._scaler.inverse_transform(df.values)
        return pd.DataFrame(original, index=df.index, columns=self._columns)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class FeatureSelector:
    """
    Select subsets of features.
    """

    # Predefined feature sets from course
    COURSE_FEATURES = [
        'nombre', 'cnbripdst', 'cnportdst',
        'permit', 'inf1024permit', 'sup1024permit', 'adminpermit',
        'deny', 'inf1024deny', 'sup1024deny', 'admindeny'
    ]

    SIMPLE_FEATURES = ['nombre', 'cnbripdst', 'cnportdst']

    def __init__(self, features: List[str] = None, preset: str = None):
        """
        Initialize selector.

        Args:
            features: List of feature names to select
            preset: Preset name ('course', 'simple')
        """
        if preset:
            if preset == 'course':
                self.features = self.COURSE_FEATURES
            elif preset == 'simple':
                self.features = self.SIMPLE_FEATURES
            else:
                raise FeatureExtractionError(f"Unknown preset: {preset}")
        else:
            self.features = features or []

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features from DataFrame."""
        available = [f for f in self.features if f in df.columns]
        missing = [f for f in self.features if f not in df.columns]

        if missing:
            # Warning but continue with available features
            print(f"Warning: Missing features: {missing}")

        if not available:
            raise FeatureExtractionError(
                "No requested features found in DataFrame"
            )

        return df[available]

    def add_feature(self, name: str) -> None:
        """Add a feature to selection."""
        if name not in self.features:
            self.features.append(name)

    def remove_feature(self, name: str) -> None:
        """Remove a feature from selection."""
        if name in self.features:
            self.features.remove(name)


class RatioTransformer:
    """
    Create ratio-based features.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ratio features to DataFrame.

        Args:
            df: Input DataFrame with basic features

        Returns:
            DataFrame with additional ratio features
        """
        result = df.copy()

        # Deny ratio
        if 'permit' in df.columns and 'deny' in df.columns:
            total = df['permit'] + df['deny']
            result['deny_ratio'] = df['deny'] / (total + 1e-10)

        # Port diversity
        if 'cnportdst' in df.columns and 'nombre' in df.columns:
            result['port_diversity'] = df['cnportdst'] / (df['nombre'] + 1e-10)

        # Admin port ratio
        admin_cols = ['adminpermit', 'admindeny']
        if all(c in df.columns for c in admin_cols):
            admin_total = df['adminpermit'] + df['admindeny']
            if 'nombre' in df.columns:
                result['admin_ratio'] = admin_total / (df['nombre'] + 1e-10)

        # System port ratio
        sys_cols = ['inf1024permit', 'inf1024deny']
        if all(c in df.columns for c in sys_cols):
            sys_total = df['inf1024permit'] + df['inf1024deny']
            if 'nombre' in df.columns:
                result['system_port_ratio'] = sys_total / (df['nombre'] + 1e-10)

        return result
