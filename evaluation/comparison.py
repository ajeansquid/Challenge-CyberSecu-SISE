# -*- coding: utf-8 -*-
"""
Model Comparison
----------------
Compare multiple models systematically.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .metrics import MetricsCalculator
from .visualizations import EvaluationPlotter
from models.registry import ModelRegistry


class ModelComparator:
    """
    Compare multiple models on the same dataset.
    """

    def __init__(self, positive_label: str = 'positif'):
        self.positive_label = positive_label
        self._metrics = MetricsCalculator(positive_label)
        self._plotter = EvaluationPlotter(positive_label)
        self._results: Dict[str, Dict[str, Any]] = {}

    def add_result(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        cv_results: Optional[Dict] = None
    ) -> None:
        """
        Add evaluation result for a model.

        Args:
            name: Model name
            y_true: True labels
            y_pred: Predictions
            y_proba: Probabilities
            cv_results: Cross-validation results
        """
        metrics = self._metrics.calculate(y_true, y_pred, y_proba)

        self._results[name] = {
            'metrics': metrics.to_dict(),
            'cv_results': cv_results,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

    def compare_models(
        self,
        model_keys: List[str],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.

        Args:
            model_keys: List of model keys from registry
            X: Features
            y: Labels
            cv: Number of CV folds

        Returns:
            DataFrame with comparison results
        """
        results = []

        for key in model_keys:
            model = ModelRegistry.create(key)

            # Cross-validate
            cv_results = self._metrics.cross_validate_metrics(
                model._model, X, y, cv
            )

            results.append({
                'model': key,
                'accuracy': cv_results['accuracy']['mean'],
                'accuracy_std': cv_results['accuracy']['std'],
                'precision': cv_results.get('precision', {}).get('mean', np.nan),
                'recall': cv_results.get('recall', {}).get('mean', np.nan),
                'f1': cv_results.get('f1', {}).get('mean', np.nan)
            })

        return pd.DataFrame(results).sort_values('accuracy', ascending=False)

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table from added results.

        Returns:
            DataFrame with all metrics
        """
        rows = []
        for name, result in self._results.items():
            row = {'model': name}
            row.update(result['metrics'])

            if result['cv_results']:
                for metric, values in result['cv_results'].items():
                    if isinstance(values, dict) and 'mean' in values:
                        row[f'{metric}_cv'] = values['mean']

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_comparison(
        self,
        metrics: List[str] = None,
        figsize: tuple = (10, 6)
    ):
        """
        Plot comparison chart.

        Args:
            metrics: Metrics to compare
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not self._results:
            raise ValueError("No results to compare. Add results first.")

        results_dict = {
            name: result['metrics']
            for name, result in self._results.items()
        }

        return self._plotter.plot_metrics_comparison(
            results_dict, metrics, figsize
        )

    def get_best_model(self, metric: str = 'accuracy') -> str:
        """
        Get name of best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Name of best model
        """
        df = self.get_comparison_table()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found")

        return df.loc[df[metric].idxmax(), 'model']

    def clear(self) -> None:
        """Clear all stored results."""
        self._results.clear()
