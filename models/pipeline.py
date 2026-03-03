# -*- coding: utf-8 -*-
"""
Model Pipeline
--------------
End-to-end pipeline for training and prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .io import save_model_file, load_model_file

from sklearn.preprocessing import StandardScaler, LabelEncoder

from core.interfaces import Classifier, PredictionResult
from core.exceptions import ModelError, ModelNotFittedError
from core.config import get_config
from .registry import ModelRegistry


class ModelPipeline:
    """
    Complete ML pipeline with preprocessing, training, and prediction.
    """

    def __init__(
        self,
        model_key: str = 'logistic_regression',
        target_col: str = None,
        positive_label: str = None,
        scale_features: bool = True,
        **model_params
    ):
        """
        Initialize pipeline.

        Args:
            model_key: Model identifier from registry
            target_col: Target column name
            positive_label: Label for positive class
            scale_features: Whether to scale features
            **model_params: Additional model parameters
        """
        config = get_config()

        self.model_key = model_key
        self.target_col = target_col or config.model.target_column
        self.positive_label = positive_label or config.model.positive_label
        self.scale_features = scale_features

        self._model = ModelRegistry.create(model_key, **model_params)
        self._scaler = StandardScaler() if scale_features else None
        self._label_encoder = LabelEncoder()
        self._feature_names: List[str] = []
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def classes(self) -> List[str]:
        """Get class labels."""
        return list(self._label_encoder.classes_)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> 'ModelPipeline':
        """
        Fit pipeline on training data.

        Args:
            df: Training DataFrame
            feature_cols: Feature columns to use

        Returns:
            self
        """
        X, y = self._prepare_data(df, feature_cols, fit=True)

        if self._scaler:
            X = self._scaler.fit_transform(X)

        y_encoded = self._label_encoder.fit_transform(y)

        self._model.fit(X, y_encoded)
        self._fitted = True

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            df: Input DataFrame

        Returns:
            Predicted labels
        """
        if not self._fitted:
            raise ModelNotFittedError("pipeline")

        X = df[self._feature_names].values

        if self._scaler:
            X = self._scaler.transform(X)

        predictions = self._model.predict(X)
        return self._label_encoder.inverse_transform(predictions)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            df: Input DataFrame

        Returns:
            Probability matrix
        """
        if not self._fitted:
            raise ModelNotFittedError("pipeline")

        X = df[self._feature_names].values

        if self._scaler:
            X = self._scaler.transform(X)

        return self._model.predict_proba(X)

    def predict_positive_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get probability of positive class.

        Args:
            df: Input DataFrame

        Returns:
            Array of positive class probabilities
        """
        probas = self.predict_proba(df)
        pos_idx = list(self._label_encoder.classes_).index(self.positive_label)
        return probas[:, pos_idx]

    def predict_full(self, df: pd.DataFrame) -> PredictionResult:
        """
        Get full prediction results.

        Args:
            df: Input DataFrame

        Returns:
            PredictionResult with predictions and probabilities
        """
        predictions = self.predict(df)

        try:
            probabilities = self.predict_positive_proba(df)
        except:
            probabilities = None

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            labels=self.classes
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        cv: int = None,
        use_loo: bool = False,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            df: Training DataFrame
            feature_cols: Feature columns
            cv: Number of folds
            use_loo: Use Leave-One-Out CV
            scoring: Scoring metric ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')

        Returns:
            Dict with CV results
        """
        from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
        from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

        config = get_config()
        cv_folds = cv or config.model.default_cv_folds

        X, y = self._prepare_data(df, feature_cols, fit=True)

        if self._scaler:
            X = self._scaler.fit_transform(X)

        y_encoded = self._label_encoder.fit_transform(y)

        # Get fresh model instance
        model = ModelRegistry.create(self.model_key)

        # Build a safe scorer that won't raise on single-class folds
        _safe_scorers = {
            'f1':        make_scorer(f1_score,        average='weighted', zero_division=0),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall':    make_scorer(recall_score,    average='weighted', zero_division=0),
        }
        scorer = _safe_scorers.get(scoring, scoring)

        # Set up CV
        if use_loo:
            cv_obj = LeaveOneOut()
        else:
            cv_obj = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # roc_auc needs binary pos-class proba; fall back to accuracy on error
        try:
            scores = cross_val_score(
                model._model, X, y_encoded, cv=cv_obj,
                scoring=scorer
            )
        except Exception:
            scores = cross_val_score(
                model._model, X, y_encoded, cv=cv_obj, scoring='accuracy'
            )
            scoring = 'accuracy (fallback)'

        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist(),
            'scoring': scoring,
            'cv_method': 'loo' if use_loo else f'{cv_folds}-fold'
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with feature importances
        """
        if not self._fitted:
            raise ModelNotFittedError("pipeline")

        if hasattr(self._model, 'feature_importances'):
            importance = self._model.feature_importances
        elif hasattr(self._model, 'coefficients'):
            importance = np.abs(self._model.coefficients)
        else:
            raise ModelError(
                f"Model {self.model_key} doesn't provide feature importance",
                self.model_key
            )

        return pd.DataFrame({
            'feature': self._feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def _prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for training/prediction."""
        if feature_cols is None:
            # Use all numeric columns except target
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != self.target_col
            ]

        if fit:
            self._feature_names = feature_cols

        X = df[feature_cols].values

        y = None
        if self.target_col in df.columns:
            y = df[self.target_col].values

        return X, y

    def save(self, path: str) -> None:
        """
        Save pipeline to file.

        The format is chosen by the file extension:
          .skops   — secure, recommended for sharing / deployment
          .joblib  — fast binary cache (default when called from ModelService)
          .pkl     — standard pickle

        See ``models.io`` for a full explanation of the security trade-offs.
        """
        state = {
            'model': self._model,
            'scaler': self._scaler,
            'label_encoder': self._label_encoder,
            'feature_names': self._feature_names,
            'model_key': self.model_key,
            'target_col': self.target_col,
            'positive_label': self.positive_label,
            'scale_features': self.scale_features,
            'fitted': self._fitted
        }
        save_model_file(state, path)

    @classmethod
    def load(cls, path: str) -> 'ModelPipeline':
        """Load pipeline from file (supports .skops, .joblib, .pkl)."""
        state = load_model_file(path)

        pipeline = cls.__new__(cls)
        pipeline._model = state['model']
        pipeline._scaler = state['scaler']
        pipeline._label_encoder = state['label_encoder']
        pipeline._feature_names = state['feature_names']
        pipeline.model_key = state['model_key']
        pipeline.target_col = state['target_col']
        pipeline.positive_label = state['positive_label']
        pipeline.scale_features = state['scale_features']
        pipeline._fitted = state['fitted']

        return pipeline
