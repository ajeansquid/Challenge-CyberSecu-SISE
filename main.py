# -*- coding: utf-8 -*-
"""
Challenge Toolkit v2 - Main Streamlit Application

A modular toolkit for cybersecurity log analysis and ML-based threat detection.
"""

import streamlit as st
from app.state import get_state
from app.pages import data_upload, feature_engineering, model_training, predictions, analysis, dim_reduction, llm_assistant


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="CyberSec ML Toolkit",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize application state
    get_state()

    # Sidebar navigation
    with st.sidebar:
        st.title("CyberSec ML Toolkit")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "Data Upload",
                "Feature Engineering",
                "Model Training",
                "Predictions",
                "Analysis Dashboard",
                "Dim. Reduction",
                "LLM Assistant"
            ]
        )

        st.markdown("---")
        render_sidebar_status()

    # Route to selected page
    if page == "Data Upload":
        data_upload.render()
    elif page == "Feature Engineering":
        feature_engineering.render()
    elif page == "Model Training":
        model_training.render()
    elif page == "Predictions":
        predictions.render()
    elif page == "Analysis Dashboard":
        analysis.render()
    elif page == "Dim. Reduction":
        dim_reduction.render()
    elif page == "LLM Assistant":
        llm_assistant.render()


def render_sidebar_status():
    """Render status indicators in sidebar."""
    from app.state import get_state
    state = get_state()

    st.subheader("Status")

    # Data status
    if state.has_raw_data():
        st.success(f"Raw data: {len(state.raw_data)} rows")
    else:
        st.info("No raw data loaded")

    if state.has_labeled_data():
        st.success(f"Labeled data: {len(state.labeled_data)} samples")
    else:
        st.info("No labeled data loaded")

    # Features status
    if state.has_features():
        st.success(f"Features: {len(state.features_data)} IPs")
    else:
        st.info("No features generated")

    # Model status
    if state.has_trained_model():
        st.success("Model trained")
    else:
        st.info("No model trained")

    # Predictions status
    if state.has_predictions():
        st.success(f"Predictions: {len(state.predictions)}")


if __name__ == "__main__":
    main()
