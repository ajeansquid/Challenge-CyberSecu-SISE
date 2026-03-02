# -*- coding: utf-8 -*-
"""
Evaluation Service
------------------
High-level model evaluation service.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt

from core.config import get_config
from evaluation import MetricsCalculator, EvaluationPlotter, ModelComparator


class EvaluationService:
    """
    Service for model evaluation and comparison.
    """

    def __init__(self, positive_label: str = None):
        config = get_config()
        self.positive_label = positive_label or config.model.positive_label

        self._metrics = MetricsCalculator(self.positive_label)
        self._plotter = EvaluationPlotter(self.positive_label)
        self._comparator = ModelComparator(self.positive_label)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictions.

        Args:
            y_true: True labels
            y_pred: Predictions
            y_proba: Probabilities

        Returns:
            Dict of metrics
        """
        metrics = self._metrics.calculate(y_true, y_pred, y_proba)
        return metrics.to_dict()

    def evaluate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Full model evaluation with CV.

        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv: CV folds

        Returns:
            Dict with all evaluation results
        """
        # Resubstitution
        model.fit(X, y)
        y_pred = model.predict(X)

        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]

        resub_metrics = self._metrics.calculate(y, y_pred, y_proba)

        # Cross-validation
        cv_results = self._metrics.cross_validate_metrics(model, X, y, cv)

        return {
            'resubstitution': resub_metrics.to_dict(),
            'cross_validation': cv_results,
            'confusion_matrix': resub_metrics.confusion_matrix.tolist()
        }

    def compare_models(
        self,
        model_keys: List[str],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_keys: Model identifiers
            X: Features
            y: Labels
            cv: CV folds

        Returns:
            Comparison DataFrame
        """
        return self._comparator.compare_models(model_keys, X, y, cv)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None
    ) -> plt.Figure:
        """Plot confusion matrix."""
        return self._plotter.plot_confusion_matrix(y_true, y_pred, labels)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'ROC Curve'
    ) -> plt.Figure:
        """Plot ROC curve."""
        return self._plotter.plot_roc_curve(y_true, y_proba, title)

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10
    ) -> plt.Figure:
        """Plot feature importance."""
        return self._plotter.plot_feature_importance(importance_df, top_n)

    def plot_elbow(
        self,
        inertias: Dict[int, float]
    ) -> plt.Figure:
        """Plot elbow curve for clustering."""
        return self._plotter.plot_elbow(inertias)

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """Get text classification report."""
        return self._metrics.get_classification_report(y_true, y_pred)

    def add_to_comparison(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> None:
        """Add model result to comparator."""
        self._comparator.add_result(name, y_true, y_pred, y_proba)

    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison table."""
        return self._comparator.get_comparison_table()

    def plot_comparison(self, metrics: List[str] = None) -> plt.Figure:
        """Plot model comparison."""
        return self._comparator.plot_comparison(metrics)

    def clear_comparison(self) -> None:
        """Clear comparison results."""
        self._comparator.clear()
