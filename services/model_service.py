# -*- coding: utf-8 -*-
"""
Model Service
-------------
High-level model training and prediction service.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from core.interfaces import PredictionResult
from core.config import get_config
from core.exceptions import ServiceError
from models import ModelRegistry, ModelPipeline, get_model
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
        contamination: float = 0.1
    ) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest.

        Args:
            df: Input DataFrame
            feature_cols: Feature columns
            contamination: Expected anomaly rate

        Returns:
            DataFrame with anomaly flags
        """
        if feature_cols is None:
            feature_cols = list(df.select_dtypes(include=[np.number]).columns)

        X = df[feature_cols].values

        self._anomaly_detector = IsolationForestModel(
            contamination=contamination
        )
        self._anomaly_detector.fit(X)

        output = df.copy()
        output['is_anomaly'] = self._anomaly_detector.predict(X) == -1
        output['anomaly_score'] = self._anomaly_detector.score(X)

        return output.sort_values('anomaly_score')

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

    def save_model(self, name: str) -> Path:
        """
        Save active model to file.

        Args:
            name: Model name

        Returns:
            Path to saved model
        """
        if self._active_pipeline is None:
            raise ServiceError(
                "No active model to save.",
                "model_service"
            )

        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / f"{name}.joblib"

        self._active_pipeline.save(str(path))
        return path

    def load_model(self, name: str) -> None:
        """
        Load model from file.

        Args:
            name: Model name
        """
        path = self.models_dir / f"{name}.joblib"

        if not path.exists():
            raise ServiceError(
                f"Model not found: {path}",
                "model_service"
            )

        self._active_pipeline = ModelPipeline.load(str(path))

    def list_saved_models(self) -> List[str]:
        """List saved models."""
        if not self.models_dir.exists():
            return []

        return [
            p.stem for p in self.models_dir.glob("*.joblib")
        ]
