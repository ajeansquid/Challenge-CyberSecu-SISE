# -*- coding: utf-8 -*-
"""
Custom Exceptions
-----------------
Domain-specific exceptions for better error handling.
"""


class ToolkitError(Exception):
    """Base exception for all toolkit errors."""
    pass


class ParsingError(ToolkitError):
    """Raised when log parsing fails."""

    def __init__(self, message: str, source: str = None, line: int = None):
        self.source = source
        self.line = line
        super().__init__(message)


class FeatureExtractionError(ToolkitError):
    """Raised when feature extraction fails."""

    def __init__(self, message: str, feature_name: str = None):
        self.feature_name = feature_name
        super().__init__(message)


class ModelError(ToolkitError):
    """Raised when model operations fail."""

    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        super().__init__(message)


class ModelNotFittedError(ModelError):
    """Raised when trying to predict with unfitted model."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' has not been fitted. Call fit() first.",
            model_name
        )


class ValidationError(ToolkitError):
    """Raised when data validation fails."""

    def __init__(self, message: str, errors: list = None):
        self.errors = errors or []
        super().__init__(message)


class ConfigurationError(ToolkitError):
    """Raised when configuration is invalid."""
    pass


class ServiceError(ToolkitError):
    """Raised when a service operation fails."""

    def __init__(self, message: str, service_name: str = None):
        self.service_name = service_name
        super().__init__(message)
