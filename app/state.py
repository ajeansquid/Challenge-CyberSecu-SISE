# -*- coding: utf-8 -*-
"""
App State Management
--------------------
Centralized state management for Streamlit app.
"""

import streamlit as st
from typing import Optional, Any, Dict
import pandas as pd

from services import DataService, FeatureService, ModelService, EvaluationService


class AppState:
    """
    Centralized application state.
    Wraps Streamlit session state with typed access.
    """

    def __init__(self):
        self._init_services()
        self._init_data()

    def _init_services(self):
        """Initialize services in session state."""
        if 'data_service' not in st.session_state:
            st.session_state.data_service = DataService()
        if 'feature_service' not in st.session_state:
            st.session_state.feature_service = FeatureService()
        if 'model_service' not in st.session_state:
            st.session_state.model_service = ModelService()
        if 'eval_service' not in st.session_state:
            st.session_state.eval_service = EvaluationService()

    def _init_data(self):
        """Initialize data slots in session state."""
        defaults = {
            'raw_data': None,
            'features_data': None,
            'labeled_data': None,
            'predictions': None,
            'training_results': None,
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    # Service accessors
    @property
    def data_service(self) -> DataService:
        return st.session_state.data_service

    @property
    def feature_service(self) -> FeatureService:
        return st.session_state.feature_service

    @property
    def model_service(self) -> ModelService:
        return st.session_state.model_service

    @property
    def eval_service(self) -> EvaluationService:
        return st.session_state.eval_service

    # Data accessors
    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        return st.session_state.get('raw_data')

    @raw_data.setter
    def raw_data(self, value: pd.DataFrame):
        st.session_state.raw_data = value

    @property
    def features_data(self) -> Optional[pd.DataFrame]:
        return st.session_state.get('features_data')

    @features_data.setter
    def features_data(self, value: pd.DataFrame):
        st.session_state.features_data = value

    @property
    def labeled_data(self) -> Optional[pd.DataFrame]:
        return st.session_state.get('labeled_data')

    @labeled_data.setter
    def labeled_data(self, value: pd.DataFrame):
        st.session_state.labeled_data = value

    @property
    def predictions(self) -> Optional[pd.DataFrame]:
        return st.session_state.get('predictions')

    @predictions.setter
    def predictions(self, value: pd.DataFrame):
        st.session_state.predictions = value

    @property
    def training_results(self) -> Optional[Dict]:
        return st.session_state.get('training_results')

    @training_results.setter
    def training_results(self, value: Dict):
        st.session_state.training_results = value

    # Status checks
    def has_raw_data(self) -> bool:
        return self.raw_data is not None

    def has_features(self) -> bool:
        return self.features_data is not None

    def has_labeled_data(self) -> bool:
        return self.labeled_data is not None

    def has_trained_model(self) -> bool:
        return self.model_service.active_model is not None

    def has_predictions(self) -> bool:
        return self.predictions is not None

    def get_status(self) -> Dict[str, bool]:
        """Get status of all data/model states."""
        return {
            'raw_data': self.has_raw_data(),
            'features': self.has_features(),
            'labeled_data': self.has_labeled_data(),
            'model': self.has_trained_model(),
            'predictions': self.has_predictions()
        }


# Global state accessor
_state: Optional[AppState] = None


def get_state() -> AppState:
    """Get application state instance."""
    global _state
    if _state is None:
        _state = AppState()
    else:
        # Re-initialize session-backed services/data in case Streamlit cleared session_state
        try:
            _state._init_services()
            _state._init_data()
        except Exception:
            # If session state isn't available yet, ignore and allow AppState methods to use safe getters
            pass
    return _state
