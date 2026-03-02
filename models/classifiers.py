# -*- coding: utf-8 -*-
"""
Classification Models
---------------------
Implementations of classification models.
"""

import numpy as np
from typing import Optional, Dict, Any, List
import joblib
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from core.interfaces import Classifier
from core.exceptions import ModelNotFittedError, ModelError
from core.config import get_config


class BaseClassifier(Classifier):
    """Base implementation for sklearn-based classifiers."""

    def __init__(self, **kwargs):
        config = get_config()
        self._random_state = kwargs.pop('random_state', config.model.random_state)
        self._model = None
        self._fitted = False
        self._feature_names: List[str] = []
        self._classes: List[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def classes(self) -> List[str]:
        return self._classes

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        if self._model is None:
            raise ModelError("Model not initialized", self.name)

        self._model.fit(X, y)
        self._fitted = True
        self._classes = list(self._model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        if not hasattr(self._model, 'predict_proba'):
            raise ModelError(
                f"Model {self.name} doesn't support probability predictions",
                self.name
            )
        return self._model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self._model:
            return self._model.get_params()
        return {}

    def save(self, path: str) -> None:
        """Save model to file."""
        state = {
            'model': self._model,
            'fitted': self._fitted,
            'feature_names': self._feature_names,
            'classes': self._classes,
            'name': self.name
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'BaseClassifier':
        """Load model from file."""
        state = joblib.load(path)
        instance = cls.__new__(cls)
        instance._model = state['model']
        instance._fitted = state['fitted']
        instance._feature_names = state['feature_names']
        instance._classes = state['classes']
        return instance


class DecisionTreeModel(BaseClassifier):
    """Decision Tree classifier (as in course)."""

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self._random_state
        )

    @property
    def name(self) -> str:
        return "decision_tree"

    @property
    def feature_importances(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.feature_importances_

    def get_rules(self, feature_names: List[str] = None) -> str:
        """Get tree rules as text."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return export_text(self._model, feature_names=feature_names)

    def get_n_leaves(self) -> int:
        """Get number of leaves (rules)."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.get_n_leaves()


class LogisticRegressionModel(BaseClassifier):
    """Logistic Regression classifier (as in course)."""

    def __init__(
        self,
        solver: str = 'liblinear',
        max_iter: int = 500,
        C: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            C=C,
            random_state=self._random_state
        )

    @property
    def name(self) -> str:
        return "logistic_regression"

    @property
    def coefficients(self) -> np.ndarray:
        """Get model coefficients."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.coef_[0]

    @property
    def intercept(self) -> float:
        """Get model intercept."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.intercept_[0]

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get coefficients mapped to feature names."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)

        return {
            name: abs(coef)
            for name, coef in zip(feature_names, self.coefficients)
        }


class RandomForestModel(BaseClassifier):
    """Random Forest classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self._random_state
        )

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def feature_importances(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.feature_importances_


class GradientBoostingModel(BaseClassifier):
    """Gradient Boosting classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self._random_state
        )

    @property
    def name(self) -> str:
        return "gradient_boosting"

    @property
    def feature_importances(self) -> np.ndarray:
        if not self._fitted:
            raise ModelNotFittedError(self.name)
        return self._model.feature_importances_


class SVMModel(BaseClassifier):
    """Support Vector Machine classifier."""

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=self._random_state
        )

    @property
    def name(self) -> str:
        return "svm"


class KNNModel(BaseClassifier):
    """K-Nearest Neighbors classifier."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )

    @property
    def name(self) -> str:
        return "knn"
