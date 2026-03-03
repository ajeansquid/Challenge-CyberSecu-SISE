# -*- coding: utf-8 -*-
"""
Anomaly Detection Models
------------------------
Unsupervised anomaly detection implementations.
"""

import numpy as np
from typing import List
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from core.interfaces import AnomalyDetector as AnomalyDetectorInterface
from core.exceptions import ModelNotFittedError
from core.config import get_config


class IsolationForestModel(AnomalyDetectorInterface):
    """
    Isolation Forest for anomaly detection (as in course).
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: int = None
    ):
        config = get_config()
        self._random_state = random_state or config.model.random_state

        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=self._random_state
        )
        self._fitted = False
        self._feature_names: List[str] = []

    @property
    def name(self) -> str:
        return "isolation_forest"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray) -> 'IsolationForestModel':
        """Fit on data (assumed to be mostly normal)."""
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Returns:
            Array where -1 = anomaly, 1 = normal
        """
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.
        Lower scores indicate more anomalous samples.
        """
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.decision_function(X)

    def get_anomaly_mask(self, X: np.ndarray) -> np.ndarray:
        """Get boolean mask where True = anomaly."""
        return self.predict(X) == -1

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X)
        return self.predict(X)

    def save(self, path: str) -> None:
        """Save model to file."""
        state = {
            'model': self._model,
            'fitted': self._fitted,
            'feature_names': self._feature_names,
            'name': self.name
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'IsolationForestModel':
        """Load model from file."""
        state = joblib.load(path)
        instance = cls.__new__(cls)
        instance._model = state['model']
        instance._fitted = state['fitted']
        instance._feature_names = state['feature_names']
        return instance


class OneClassSVMModel(AnomalyDetectorInterface):
    """
    One-Class SVM for anomaly detection.
    Alternative to Isolation Forest.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.1,
        gamma: str = 'scale'
    ):
        from sklearn.svm import OneClassSVM
        self._model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )
        self._fitted = False

    @property
    def name(self) -> str:
        return "one_class_svm"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray) -> 'OneClassSVMModel':
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.decision_function(X)


class LocalOutlierFactorModel(AnomalyDetectorInterface):
    """
    Local Outlier Factor for anomaly detection.
    Flags IPs anomalous relative to their local neighbourhood even when
    globally rare behaviour is not present.  Complements IsolationForest.

    Uses novelty=True so the model can score unseen data after fitting.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = 'minkowski',
    ):
        self._model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            novelty=True,   # required for predict/score on unseen data
        )
        self._fitted = False
        self._feature_names: List[str] = []

    @property
    def name(self) -> str:
        return "local_outlier_factor"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray) -> 'LocalOutlierFactorModel':
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns -1 for anomalies, 1 for normal points."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.predict(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns negative LOF scores; lower = more anomalous (consistent with IsolationForest)."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.score_samples(X)

    def get_anomaly_mask(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X) == -1
