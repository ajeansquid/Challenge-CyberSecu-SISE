# -*- coding: utf-8 -*-
"""
Clustering Models
-----------------
Clustering implementations for behavioral analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

from core.exceptions import ModelNotFittedError
from core.config import get_config


class KMeansModel:
    """
    K-Means clustering (as in course).
    """

    def __init__(
        self,
        n_clusters: int = 2,
        scale_features: bool = True,
        random_state: int = None
    ):
        config = get_config()
        self._random_state = random_state or config.model.random_state

        self._model = KMeans(
            n_clusters=n_clusters,
            random_state=self._random_state
        )
        self._scaler = StandardScaler() if scale_features else None
        self._scale_features = scale_features
        self._fitted = False
        self._feature_names: List[str] = []

    @property
    def name(self) -> str:
        return "kmeans"

    @property
    def n_clusters(self) -> int:
        return self._model.n_clusters

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def cluster_centers(self) -> np.ndarray:
        """Get cluster centers."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        centers = self._model.cluster_centers_
        if self._scaler:
            centers = self._scaler.inverse_transform(centers)
        return centers

    @property
    def inertia(self) -> float:
        """Get inertia (within-cluster sum of squares)."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.inertia_

    def fit(self, X: np.ndarray) -> 'KMeansModel':
        """Fit clustering model."""
        if self._scaler:
            X = self._scaler.fit_transform(X)
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign samples to clusters."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        if self._scaler:
            X = self._scaler.transform(X)
        return self._model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        if self._scaler:
            X = self._scaler.fit_transform(X)
        labels = self._model.fit_predict(X)
        self._fitted = True
        return labels

    def get_cluster_stats(
        self,
        X: np.ndarray,
        feature_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Get statistics per cluster.

        Args:
            X: Feature data
            feature_names: Optional feature names

        Returns:
            DataFrame with cluster statistics
        """
        labels = self.predict(X)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = labels

        return df.groupby('cluster')[feature_names].agg(['mean', 'std'])

    @staticmethod
    def find_optimal_k(
        X: np.ndarray,
        k_range: Tuple[int, int] = (1, 11),
        random_state: int = 42
    ) -> Dict[int, float]:
        """
        Find optimal K using elbow method.

        Args:
            X: Feature data
            k_range: Range of K values to try
            random_state: Random state

        Returns:
            Dict mapping k to inertia
        """
        inertias = {}
        for k in range(k_range[0], k_range[1]):
            km = KMeans(n_clusters=k, random_state=random_state)
            km.fit(X)
            inertias[k] = km.inertia_
        return inertias

    def save(self, path: str) -> None:
        """Save model to file."""
        state = {
            'model': self._model,
            'scaler': self._scaler,
            'fitted': self._fitted,
            'feature_names': self._feature_names
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'KMeansModel':
        """Load model from file."""
        state = joblib.load(path)
        instance = cls.__new__(cls)
        instance._model = state['model']
        instance._scaler = state['scaler']
        instance._fitted = state['fitted']
        instance._feature_names = state['feature_names']
        instance._scale_features = state['scaler'] is not None
        return instance


class DBSCANModel:
    """
    DBSCAN clustering for density-based anomaly detection.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        scale_features: bool = True
    ):
        self._model = DBSCAN(eps=eps, min_samples=min_samples)
        self._scaler = StandardScaler() if scale_features else None
        self._scale_features = scale_features
        self._fitted = False
        self._labels: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "dbscan"

    @property
    def labels(self) -> np.ndarray:
        """Get cluster labels (-1 = noise/anomaly)."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._labels

    @property
    def n_clusters(self) -> int:
        """Get number of clusters (excluding noise)."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return len(set(self._labels)) - (1 if -1 in self._labels else 0)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        if self._scaler:
            X = self._scaler.fit_transform(X)
        self._labels = self._model.fit_predict(X)
        self._fitted = True
        return self._labels

    def get_anomalies_mask(self) -> np.ndarray:
        """Get mask where True = noise point (potential anomaly)."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._labels == -1
