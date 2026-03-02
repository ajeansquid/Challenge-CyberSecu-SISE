# -*- coding: utf-8 -*-
"""
Core Interfaces
---------------
Abstract base classes defining contracts for all modules.
Implement these to create new parsers, models, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PredictionResult:
    """Container for prediction results."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    labels: Optional[List[str]] = None


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    extra_metrics: Optional[Dict[str, float]] = None


@dataclass
class FeatureSet:
    """Container for extracted features."""
    data: pd.DataFrame
    feature_names: List[str]
    index_column: str
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Parser Interface
# ============================================================================

class Parser(ABC):
    """
    Abstract base class for log parsers.
    Implement this to add support for new log formats.
    """

    @abstractmethod
    def parse(self, source: str) -> pd.DataFrame:
        """
        Parse log data from source.

        Args:
            source: File path or data string

        Returns:
            Parsed DataFrame
        """
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate parsed data.

        Args:
            df: Parsed DataFrame

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        pass

    @property
    @abstractmethod
    def expected_columns(self) -> List[str]:
        """Return list of expected column names."""
        pass


# ============================================================================
# Feature Extractor Interface
# ============================================================================

class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction.
    Implement this to create custom feature engineering pipelines.
    """

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> FeatureSet:
        """
        Extract features from raw data.

        Args:
            df: Raw input DataFrame

        Returns:
            FeatureSet containing extracted features
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name for identification."""
        pass


# ============================================================================
# Classifier Interface
# ============================================================================

class Classifier(ABC):
    """
    Abstract base class for classification models.
    Implement this to add new ML classifiers.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Classifier':
        """
        Fit the model on training data.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return whether model has been fitted."""
        pass


# ============================================================================
# Anomaly Detector Interface
# ============================================================================

class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection.
    Implement this to add unsupervised anomaly detection methods.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        """Fit on normal data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Returns:
            Array where -1 = anomaly, 1 = normal
        """
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.

        Returns:
            Anomaly scores (lower = more anomalous)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return detector name."""
        pass


# ============================================================================
# Evaluator Interface
# ============================================================================

class Evaluator(ABC):
    """
    Abstract base class for model evaluation.
    Implement this to add custom evaluation strategies.
    """

    @abstractmethod
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> EvaluationResult:
        """
        Evaluate predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            EvaluationResult with metrics
        """
        pass

    @abstractmethod
    def cross_validate(
        self,
        model: Classifier,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Classifier to evaluate
            X: Feature matrix
            y: Target labels
            cv: Number of folds

        Returns:
            Dict with CV results
        """
        pass


# ============================================================================
# Data Service Interface
# ============================================================================

class DataService(ABC):
    """
    Abstract base class for data operations.
    Implement this to add custom data backends.
    """

    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from source."""
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame, destination: str, **kwargs) -> None:
        """Save data to destination."""
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data structure."""
        pass
