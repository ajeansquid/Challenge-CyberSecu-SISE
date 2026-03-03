# -*- coding: utf-8 -*-
"""
Evaluation Visualizations
-------------------------
Plotting utilities for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

from .metrics import MetricsCalculator


class EvaluationPlotter:
    """
    Create evaluation visualizations.
    """

    def __init__(self, positive_label: str = 'positive'):
        self.positive_label = positive_label
        self._metrics = MetricsCalculator(positive_label)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6),
        cmap: str = 'Blues'
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Normalize values
            figsize: Figure size
            cmap: Colormap

        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=figsize)

        if labels is None:
            labels = np.unique(y_true)

        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()
        return fig

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'ROC Curve',
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot ROC curve (styled like course).

        Args:
            y_true: True labels
            y_proba: Positive class probabilities
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        roc_data = self._metrics.get_roc_data(y_true, y_proba)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true == self.positive_label, y_proba)

        fig, ax = plt.subplots(figsize=figsize)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Random')

        # ROC curve
        ax.plot(
            roc_data['fpr'],
            roc_data['tpr'],
            linestyle='-',
            color='orange',
            label=f'ROC (AUC = {auc:.3f})'
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')

        plt.tight_layout()
        return fig

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'Precision-Recall Curve',
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_proba: Positive class probabilities
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        pr_data = self._metrics.get_pr_data(y_true, y_proba)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            pr_data['recall'],
            pr_data['precision'],
            color='blue',
            label=f'AP = {pr_data["average_precision"]:.3f}'
        )

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance'
            top_n: Number of top features
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        df = importance_df.head(top_n).sort_values('importance')

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(df['feature'], df['importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance')

        plt.tight_layout()
        return fig

    def plot_metrics_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot metrics comparison across models.

        Args:
            results: Dict of {model_name: {metric: value}}
            metrics: Metrics to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        models = list(results.keys())
        x = np.arange(len(models))
        width = 0.8 / len(metrics)

        fig, ax = plt.subplots(figsize=figsize)

        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in models]
            ax.bar(x + i * width, values, width, label=metric.capitalize())

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        return fig

    def plot_elbow(
        self,
        inertias: Dict[int, float],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot elbow curve for K-Means.

        Args:
            inertias: Dict of {k: inertia}
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        ks = sorted(inertias.keys())
        values = [inertias[k] for k in ks]

        ax.plot(ks, values, 'bo-')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method')

        plt.tight_layout()
        return fig
