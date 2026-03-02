# -*- coding: utf-8 -*-
"""Predictions Page"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import tempfile

from app.state import get_state


def render():
    """Render predictions page."""
    state = get_state()

    st.title("Predictions")

    if not state.has_trained_model():
        st.warning("Please train a model first.")
        return

    tab_single, tab_batch = st.tabs(["🔍 Single Prediction", "📦 Batch Predictions"])

    with tab_single:
        render_single_prediction(state)

    with tab_batch:
        render_predict_section(state)
        if state.has_predictions():
            render_results_section(state)


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

def render_single_prediction(state):
    """Let the user type in feature values and get a one-off prediction."""
    st.header("Single Observation Prediction")

    pipeline = state.model_service.active_model
    feature_names = pipeline.feature_names if pipeline else []

    if not feature_names:
        st.info("No feature names known — train a model first.")
        return

    st.markdown("Enter values for each feature:")

    # Use reference data to show reasonable defaults / ranges
    ref_df = None
    if state.has_features():
        ref_df = state.features_data
    elif state.has_labeled_data():
        ref_df = state.labeled_data

    values = {}
    cols_per_row = 3
    rows = [feature_names[i:i+cols_per_row] for i in range(0, len(feature_names), cols_per_row)]

    for row_features in rows:
        cols = st.columns(len(row_features))
        for col, feat in zip(cols, row_features):
            default_val = 0.0
            if ref_df is not None and feat in ref_df.columns:
                default_val = float(ref_df[feat].median())
            values[feat] = col.number_input(
                feat, value=default_val, key=f"single_{feat}"
            )

    if st.button("Predict", type="primary", key="btn_single_predict"):
        try:
            input_df = pd.DataFrame([values])
            result = state.model_service.predict(input_df)
            pred = result.predictions[0]
            proba = float(result.probabilities[0]) if result.probabilities is not None else None

            positive_label = st.session_state.get('positive_label', 'positif')
            is_threat = pred == positive_label

            if is_threat:
                st.error(f"**Prediction: {pred}** {'🚨 THREAT DETECTED' if is_threat else ''}")
            else:
                st.success(f"**Prediction: {pred}** ✅ Benign")

            if proba is not None:
                st.metric("Threat probability", f"{proba:.1%}")
                st.progress(proba)

        except Exception as e:
            st.error(f"Error: {e}")


# ---------------------------------------------------------------------------
# Batch predictions (unchanged)
# ---------------------------------------------------------------------------

def render_predict_section(state):
    """Render batch prediction controls."""
    st.header("Batch Predictions")

    # Option 1: Use features data
    if state.has_features():
        st.subheader("Predict on Generated Features")
        if st.button("Predict on Features"):
            try:
                results = state.model_service.predict_dataframe(state.features_data)
                state.predictions = results
                st.success(f"Generated {len(results)} predictions!")
            except Exception as e:
                st.error(f"Error: {e}")

    # Option 2: Upload
    st.subheader("Upload New Data")
    new_file = st.file_uploader("Upload data", type=['xlsx', 'csv'])

    if new_file:
        if st.button("Predict on Upload"):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=new_file.name[-5:]
                ) as f:
                    f.write(new_file.getvalue())
                    temp_path = f.name

                df = state.data_service.load_features(temp_path)
                results = state.model_service.predict_dataframe(df)
                state.predictions = results
                st.success(f"Generated {len(results)} predictions!")

            except Exception as e:
                st.error(f"Error: {e}")


def render_results_section(state):
    """Render prediction results."""
    st.markdown("---")
    st.header("Results")

    results = state.predictions
    positive_label = st.session_state.get('positive_label', 'positif')

    # Metrics
    cols = st.columns(3)
    pos_count = (results['prediction'] == positive_label).sum()
    cols[0].metric("Positive Predictions", pos_count)
    cols[1].metric("Total", len(results))
    cols[2].metric("Positive Rate", f"{pos_count/len(results)*100:.1f}%")

    # Top risks
    st.subheader("High Risk IPs (Top 20)")
    st.dataframe(results.head(20), width='stretch')

    # Distribution
    if 'probability' in results.columns:
        st.subheader("Probability Distribution")
        fig = px.histogram(
            results, x='probability', nbins=50,
            title='Distribution of Positive Probabilities'
        )
        st.plotly_chart(fig)

    # Download
    csv = results.to_csv()
    st.download_button("Download Results", csv, "predictions.csv", "text/csv")
