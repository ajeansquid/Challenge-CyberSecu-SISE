# -*- coding: utf-8 -*-
"""
Model Service
-------------
High-level model training and prediction service.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

from models.io import save_model_file, load_model_file

from core.interfaces import PredictionResult
from core.config import get_config
from core.exceptions import ServiceError
from models import ModelRegistry, ModelPipeline
from models.anomaly import IsolationForestModel
from models.clustering import KMeansModel


class ModelService:
    """
    Service for model training, prediction, and management.
    """

    def __init__(self):
        config = get_config()
        self.models_dir = config.models_dir
        self._active_pipeline: Optional[ModelPipeline] = None
        self._anomaly_detector: Optional[IsolationForestModel] = None
        self._clusterer: Optional[KMeansModel] = None

    @property
    def active_model(self) -> Optional[ModelPipeline]:
        """Get active model pipeline."""
        return self._active_pipeline

    @property
    def has_fitted_anomaly_detector(self) -> bool:
        """True if an anomaly detector is loaded and fitted."""
        return self._anomaly_detector is not None and self._anomaly_detector.is_fitted

    @property
    def has_fitted_clusterer(self) -> bool:
        """True if a clusterer is loaded and fitted."""
        return self._clusterer is not None and self._clusterer.is_fitted

    def apply_anomaly_detector(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
    ) -> pd.DataFrame:
        """
        Score new data with the **already-fitted** anomaly detector.
        Does NOT refit — call ``detect_anomalies()`` if you want fit+predict.

        Args:
            df:           Input DataFrame.
            feature_cols: Columns to use.  Defaults to all numeric columns.

        Returns:
            DataFrame with ``is_anomaly`` (bool) and ``anomaly_score`` (float) columns.
        """
        if not self.has_fitted_anomaly_detector:
            raise ServiceError(
                "No fitted anomaly detector.  "
                "Train one in Model Training or load a saved model first.",
                "model_service",
            )
        if feature_cols is None:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)

        X = df[feature_cols].values
        output = df.copy()
        output['is_anomaly'] = self._anomaly_detector.predict(X) == -1
        raw_scores = self._anomaly_detector.score(X)
        neg = -raw_scores
        score_min, score_max = neg.min(), neg.max()
        if score_max > score_min:
            output['anomaly_score'] = (neg - score_min) / (score_max - score_min)
        else:
            output['anomaly_score'] = 0.0
        return output.sort_values('anomaly_score', ascending=False)

    def apply_clusterer(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
    ) -> pd.DataFrame:
        """
        Assign new data to clusters using the **already-fitted** clusterer.
        Does NOT refit — call ``cluster()`` if you want fit+predict.

        Args:
            df:           Input DataFrame.
            feature_cols: Columns to use.  Defaults to all numeric columns.

        Returns:
            DataFrame with a ``cluster`` column.
        """
        if not self.has_fitted_clusterer:
            raise ServiceError(
                "No fitted clusterer.  "
                "Train one in Model Training or load a saved model first.",
                "model_service",
            )
        if feature_cols is None:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)

        X = df[feature_cols].values
        output = df.copy()
        output['cluster'] = self._clusterer.predict(X)
        return output

    def list_available_models(
        self,
        model_type: str = None
    ) -> List[Dict[str, str]]:
        """
        List available models.

        Args:
            model_type: Filter by type ('classifier', 'anomaly', 'clustering')

        Returns:
            List of model info dicts
        """
        keys = ModelRegistry.list_models(model_type)
        return [
            {
                'key': key,
                'name': ModelRegistry.get_info(key).name,
                'description': ModelRegistry.get_info(key).description,
                'type': ModelRegistry.get_info(key).model_type
            }
            for key in keys
        ]

    def train(
        self,
        df: pd.DataFrame,
        model_key: str = 'logistic_regression',
        feature_cols: List[str] = None,
        target_col: str = None,
        scale_features: bool = True,
        **model_params
    ) -> Dict[str, Any]:
        """
        Train a classification model.

        Args:
            df: Training DataFrame with features and target
            model_key: Model identifier
            feature_cols: Feature columns
            target_col: Target column
            scale_features: Whether to scale features
            **model_params: Model parameters

        Returns:
            Dict with training results
        """
        config = get_config()
        target = target_col or config.model.target_column

        if target not in df.columns:
            raise ServiceError(
                f"Target column '{target}' not found",
                "model_service"
            )

        pipeline = ModelPipeline(
            model_key=model_key,
            target_col=target,
            scale_features=scale_features,
            **model_params
        )

        pipeline.fit(df, feature_cols=feature_cols)
        self._active_pipeline = pipeline

        return {
            'model': model_key,
            'features': pipeline.feature_names,
            'classes': pipeline.classes,
            'fitted': True
        }

    def cross_validate(
        self,
        df: pd.DataFrame,
        model_key: str = 'logistic_regression',
        feature_cols: List[str] = None,
        cv: int = 5,
        use_loo: bool = False,
        scoring: str = 'accuracy',
        **model_params
    ) -> Dict[str, Any]:
        """
        Cross-validate a model.

        Args:
            df: Training DataFrame
            model_key: Model identifier
            feature_cols: Feature columns
            cv: Number of folds
            use_loo: Use Leave-One-Out
            **model_params: Model parameters

        Returns:
            CV results
        """
        pipeline = ModelPipeline(
            model_key=model_key,
            **model_params
        )

        return pipeline.cross_validate(
            df,
            feature_cols=feature_cols,
            cv=cv,
            use_loo=use_loo,
            scoring=scoring
        )

    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = True
    ) -> PredictionResult:
        """
        Make predictions with active model.

        Args:
            df: Input DataFrame
            return_proba: Include probabilities

        Returns:
            PredictionResult
        """
        if self._active_pipeline is None:
            raise ServiceError(
                "No active model. Train a model first.",
                "model_service"
            )

        return self._active_pipeline.predict_full(df)

    def predict_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions and return DataFrame with results.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with predictions and probabilities
        """
        result = self.predict(df)

        output = df.copy()
        output['prediction'] = result.predictions

        if result.probabilities is not None:
            output['probability'] = result.probabilities

        return output.sort_values('probability', ascending=False)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from active model."""
        if self._active_pipeline is None:
            raise ServiceError(
                "No active model.",
                "model_service"
            )
        return self._active_pipeline.get_feature_importance()

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        contamination: float = 0.1,
        model_key: str = 'isolation_forest',
        **model_params,
    ) -> pd.DataFrame:
        """
        Fit an anomaly detector and flag anomalies.

        Args:
            df:           Input DataFrame.
            feature_cols: Feature columns (defaults to all numeric).
            contamination: Expected anomaly rate.
            model_key:    One of 'isolation_forest', 'one_class_svm', 'local_outlier_factor'.
            **model_params: Extra kwargs forwarded to the model constructor.

        Returns:
            DataFrame with ``is_anomaly`` (bool) and ``anomaly_score`` (float) columns.
        """
        if feature_cols is None:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)

        X = df[feature_cols].values

        # OneClassSVM does not accept contamination — only pass it for models that do
        _SUPPORTS_CONTAMINATION = {'isolation_forest', 'local_outlier_factor'}
        if model_key in _SUPPORTS_CONTAMINATION:
            model_params['contamination'] = contamination

        self._anomaly_detector = ModelRegistry.create(
            model_key, **model_params
        )
        self._anomaly_detector.fit(X)

        output = df.copy()
        output['is_anomaly'] = self._anomaly_detector.predict(X) == -1
        raw_scores = self._anomaly_detector.score(X)  # lower = more anomalous for all models
        # Normalize to [0, 1] where 1 = most anomalous (invert + min-max scale)
        neg = -raw_scores
        score_min, score_max = neg.min(), neg.max()
        if score_max > score_min:
            output['anomaly_score'] = (neg - score_min) / (score_max - score_min)
        else:
            output['anomaly_score'] = 0.0

        return output.sort_values('anomaly_score', ascending=False)

    def compute_shap_values(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        max_rows: int = 500,
    ) -> Optional[Any]:
        """
        Compute SHAP values for the fitted anomaly detector.

        Uses ``TreeExplainer`` for IsolationForest (fast),
        ``KernelExplainer`` for all other models (slower, sampled).

        Args:
            df:           Feature DataFrame.
            feature_cols: Columns to explain.
            max_rows:     Cap on rows for KernelExplainer (performance).

        Returns:
            (shap_values, df_slice, feature_cols) tuple, or (None, error_msg) on failure.
        """
        try:
            import shap
        except ImportError as _ie:
            import sys as _sys
            return None, f"shap import failed: {_ie} (Python: {_sys.executable})"

        if not self.has_fitted_anomaly_detector:
            return None, "No fitted anomaly detector found. Train a model first."

        X = df[feature_cols].values
        model = self._anomaly_detector._model
        tree_error = None

        try:
            # IsolationForest supports TreeExplainer directly
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if hasattr(shap_values, 'values'):
                # shap.Explanation object — extract raw ndarray
                shap_values = shap_values.values
            explainer_used = "TreeExplainer"
        except Exception as exc:
            tree_error = str(exc)
            # Fallback: KernelExplainer on a small background sample
            try:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X), size=min(100, len(X)), replace=False)
                bg = X[idx]
                predict_fn = lambda x: self._anomaly_detector.score(x)  # noqa: E731
                explainer = shap.KernelExplainer(predict_fn, bg)
                X_sample = X[:max_rows]
                shap_values = explainer.shap_values(X_sample)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                df = df.iloc[:max_rows]
                explainer_used = f"KernelExplainer (TreeExplainer failed: {tree_error})"
            except Exception as exc2:
                return None, f"TreeExplainer: {tree_error} | KernelExplainer: {exc2}"

        shap_arr = np.asarray(shap_values)
        return shap_arr, df[feature_cols].iloc[:len(shap_arr)], feature_cols, explainer_used

    def cluster(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        n_clusters: int = 2
    ) -> pd.DataFrame:
        """
        Cluster data using K-Means.

        Args:
            df: Input DataFrame
            feature_cols: Feature columns
            n_clusters: Number of clusters

        Returns:
            DataFrame with cluster assignments
        """
        if feature_cols is None:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)

        X = df[feature_cols].values

        self._clusterer = KMeansModel(n_clusters=n_clusters)
        labels = self._clusterer.fit_predict(X)

        output = df.copy()
        output['cluster'] = labels

        return output

    def find_optimal_clusters(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        k_range: tuple = (1, 11)
    ) -> Dict[int, float]:
        """
        Find optimal number of clusters.

        Args:
            df: Input DataFrame
            feature_cols: Feature columns
            k_range: Range of K values

        Returns:
            Dict mapping K to inertia
        """
        if feature_cols is None:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)

        X = df[feature_cols].values

        return KMeansModel.find_optimal_k(X, k_range)

    def save_model(self, name: str, fmt: str = 'joblib') -> Path:
        """
        Save the active supervised pipeline to disk.

        Args:
            name: Filename stem (e.g. ``'my_classifier'``).
            fmt:  Serialization format.

                  * ``'skops'``  — Secure JSON-based format.  Recommended for
                    any model you share, deploy to production, or keep in git.
                    Requires ``skops`` (``uv add skops``).
                  * ``'joblib'`` — Fast binary pickle.  Good for local caches
                    you control end-to-end.  **Cannot be safely loaded from an
                    untrusted source** (pickle executes arbitrary code).
                  * ``'pkl'``    — Standard pickle.  Avoid unless required by
                    an external tool.

        Returns:
            ``Path`` to the saved file.
        """
        if self._active_pipeline is None:
            raise ServiceError("No active model to save.", "model_service")

        ext = fmt.lstrip('.')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / f"{name}.{ext}"

        self._active_pipeline.save(str(path))
        return path

    def load_model(self, name: str) -> None:
        """
        Load a supervised pipeline from disk.

        Probes for ``.skops``, ``.joblib``, and ``.pkl`` in that order so you
        can upgrade to a more secure format without changing call sites.
        """
        for ext in ('.skops', '.joblib', '.pkl'):
            path = self.models_dir / f"{name}{ext}"
            if path.exists():
                self._active_pipeline = ModelPipeline.load(str(path))
                return
        raise ServiceError(
            f"No saved model named '{name}' found in {self.models_dir}",
            "model_service"
        )

    def save_unsupervised(self, name: str, fmt: str = 'joblib') -> Path:
        """
        Save the active anomaly detector or clusterer to disk.

        Args:
            name: Filename stem.
            fmt:  ``'skops'``, ``'joblib'``, or ``'pkl'``.

        Returns:
            ``Path`` to the saved file.
        """
        model = self._anomaly_detector or self._clusterer
        if model is None:
            raise ServiceError(
                "No fitted unsupervised model to save.  "
                "Run detect_anomalies() or cluster() first.",
                "model_service"
            )

        ext = fmt.lstrip('.')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / f"{name}.{ext}"

        state = {
            'model_type': model.name,
            'model_object': model,
        }
        save_model_file(state, path)
        return path

    def load_unsupervised(self, name: str) -> None:
        """
        Load an anomaly detector or clusterer saved with
        :meth:`save_unsupervised`.

        Probes for ``.skops``, ``.joblib``, and ``.pkl`` in that order.
        """
        for ext in ('.skops', '.joblib', '.pkl'):
            path = self.models_dir / f"{name}{ext}"
            if path.exists():
                state = load_model_file(path)
                model_type = state.get('model_type', '')
                obj = state['model_object']
                if model_type in ('isolation_forest', 'one_class_svm'):
                    self._anomaly_detector = obj
                else:
                    self._clusterer = obj
                return
        raise ServiceError(
            f"No saved unsupervised model named '{name}' in {self.models_dir}",
            "model_service"
        )

    def list_saved_models(self) -> List[Dict[str, str]]:
        """
        List all saved model files in the models directory.

        Returns:
            List of dicts with keys ``'name'``, ``'format'``, and ``'path'``.
        """
        if not self.models_dir.exists():
            return []

        results = []
        for ext in ('.skops', '.joblib', '.pkl'):
            for p in sorted(self.models_dir.glob(f"*{ext}")):
                results.append({
                    'name': p.stem,
                    'format': ext.lstrip('.'),
                    'path': str(p),
                })
        return results
