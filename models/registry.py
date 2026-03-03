# -*- coding: utf-8 -*-
"""
Model Registry
--------------
Registry for model types with factory functionality.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from core.exceptions import ConfigurationError

from .classifiers import (
    DecisionTreeModel,
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
    KNNModel
)
from .anomaly import IsolationForestModel, OneClassSVMModel
from .clustering import KMeansModel, DBSCANModel


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    model_class: type
    description: str
    model_type: str  # 'classifier', 'anomaly', 'clustering'
    default_params: Dict[str, Any] = None


class ModelRegistry:
    """
    Registry of available models.
    Provides factory methods and model information.
    """

    _models: Dict[str, ModelInfo] = {
        # Classifiers
        'decision_tree': ModelInfo(
            name='Decision Tree',
            model_class=DecisionTreeModel,
            description='Simple interpretable tree-based classifier',
            model_type='classifier'
        ),
        'logistic_regression': ModelInfo(
            name='Logistic Regression',
            model_class=LogisticRegressionModel,
            description='Linear classifier with probability outputs',
            model_type='classifier'
        ),
        'random_forest': ModelInfo(
            name='Random Forest',
            model_class=RandomForestModel,
            description='Ensemble of decision trees',
            model_type='classifier'
        ),
        'gradient_boosting': ModelInfo(
            name='Gradient Boosting',
            model_class=GradientBoostingModel,
            description='Sequential ensemble with gradient descent',
            model_type='classifier'
        ),
        'svm': ModelInfo(
            name='Support Vector Machine',
            model_class=SVMModel,
            description='Kernel-based classifier',
            model_type='classifier'
        ),
        'knn': ModelInfo(
            name='K-Nearest Neighbors',
            model_class=KNNModel,
            description='Instance-based learning',
            model_type='classifier'
        ),

        # Anomaly detectors
        'isolation_forest': ModelInfo(
            name='Isolation Forest',
            model_class=IsolationForestModel,
            description='Anomaly detection via isolation',
            model_type='anomaly'
        ),
        'one_class_svm': ModelInfo(
            name='One-Class SVM',
            model_class=OneClassSVMModel,
            description='Boundary-based anomaly detection',
            model_type='anomaly'
        ),

        # Clustering
        'kmeans': ModelInfo(
            name='K-Means',
            model_class=KMeansModel,
            description='Centroid-based clustering',
            model_type='clustering'
        ),
        'dbscan': ModelInfo(
            name='DBSCAN',
            model_class=DBSCANModel,
            description='Density-based clustering',
            model_type='clustering'
        ),
    }

    @classmethod
    def register(cls, key: str, info: ModelInfo) -> None:
        """
        Register a new model type.

        Args:
            key: Model identifier
            info: ModelInfo with model details
        """
        cls._models[key] = info

    @classmethod
    def create(cls, model_key: str, **kwargs) -> Any:
        """
        Create a model instance.

        Args:
            model_key: Model identifier
            **kwargs: Model parameters

        Returns:
            Model instance
        """
        if model_key not in cls._models:
            available = list(cls._models.keys())
            raise ConfigurationError(
                f"Unknown model: {model_key}. Available: {available}"
            )

        info = cls._models[model_key]
        return info.model_class(**kwargs)

    @classmethod
    def get_info(cls, model_key: str) -> ModelInfo:
        """Get model information."""
        if model_key not in cls._models:
            raise ConfigurationError(f"Unknown model: {model_key}")
        return cls._models[model_key]

    @classmethod
    def list_models(
        cls,
        model_type: str = None
    ) -> List[str]:
        """
        List available models.

        Args:
            model_type: Filter by type ('classifier', 'anomaly', 'clustering')

        Returns:
            List of model keys
        """
        if model_type:
            return [
                k for k, v in cls._models.items()
                if v.model_type == model_type
            ]
        return list(cls._models.keys())

    @classmethod
    def list_classifiers(cls) -> List[str]:
        """List available classifiers."""
        return cls.list_models('classifier')

    @classmethod
    def list_anomaly_detectors(cls) -> List[str]:
        """List available anomaly detectors."""
        return cls.list_models('anomaly')

    @classmethod
    def list_clustering(cls) -> List[str]:
        """List available clustering models."""
        return cls.list_models('clustering')


# Convenience function
def get_model(model_key: str, **kwargs) -> Any:
    """
    Get a model instance.

    Args:
        model_key: Model identifier
        **kwargs: Model parameters

    Returns:
        Model instance
    """
    return ModelRegistry.create(model_key, **kwargs)
