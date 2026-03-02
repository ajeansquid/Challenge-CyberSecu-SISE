# -*- coding: utf-8 -*-
"""
Evaluation Metrics
------------------
Classification metrics and evaluation utilities.
"""

import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import cross_val_score, LeaveOneOut

from core.interfaces import EvaluationResult
from core.config import get_config


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None
    specificity: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc': self.auc,
            'specificity': self.specificity
        }


class MetricsCalculator:
    """
    Calculate classification metrics.
    """

    def __init__(self, positive_label: str = None):
        config = get_config()
        self.positive_label = positive_label or config.model.positive_label

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """
        Calculate all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            ClassificationMetrics object
        """
        acc = accuracy_score(y_true, y_pred)

        prec = precision_score(
            y_true, y_pred,
            pos_label=self.positive_label,
            zero_division=0
        )

        rec = recall_score(
            y_true, y_pred,
            pos_label=self.positive_label,
            zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred,
            pos_label=self.positive_label,
            zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred)

        # Calculate specificity (TN / (TN + FP))
        specificity = None
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            if (tn + fp) > 0:
                specificity = tn / (tn + fp)

        # AUC if probabilities provided
        auc = None
        if y_proba is not None:
            try:
                auc = roc_auc_score(
                    y_true == self.positive_label,
                    y_proba
                )
            except:
                pass

        return ClassificationMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            auc=auc,
            specificity=specificity,
            confusion_matrix=cm
        )

    def get_roc_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data.

        Args:
            y_true: True labels
            y_proba: Positive class probabilities

        Returns:
            Dict with fpr, tpr, thresholds
        """
        fpr, tpr, thresholds = roc_curve(
            y_true == self.positive_label,
            y_proba
        )
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }

    def get_pr_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get Precision-Recall curve data.

        Args:
            y_true: True labels
            y_proba: Positive class probabilities

        Returns:
            Dict with precision, recall, thresholds, ap
        """
        precision, recall, thresholds = precision_recall_curve(
            y_true == self.positive_label,
            y_proba
        )
        ap = average_precision_score(
            y_true == self.positive_label,
            y_proba
        )
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': ap
        }

    def cross_validate_metrics(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics using cross-validation.

        Args:
            model: Sklearn-compatible model
            X: Features
            y: Labels
            cv: Number of folds
            metrics: Metrics to calculate

        Returns:
            Dict mapping metric name to {mean, std}
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        results = {}
        for metric in metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }

        return results

    def leave_one_out_evaluate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate using Leave-One-Out CV (as in course).

        Args:
            model: Sklearn-compatible model
            X: Features
            y: Labels

        Returns:
            Dict with accuracy metrics
        """
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

        return {
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores.std(),
            'n_samples': len(y)
        }

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """Get sklearn classification report."""
        return classification_report(y_true, y_pred)
