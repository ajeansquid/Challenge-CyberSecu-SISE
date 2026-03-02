# -*- coding: utf-8 -*-
"""Model Training Page"""

import streamlit as st
import plotly.express as px
import numpy as np

from app.state import get_state

# ---------------------------------------------------------------------------
# Per-model hyperparameter definitions
# ---------------------------------------------------------------------------

_HYPERPARAM_DEFS = {
    'decision_tree': [
        ('max_depth',         'slider_opt', 1, 30, None,  'Max depth (None = unlimited)'),
        ('min_samples_split', 'slider',     2, 20, 2,     'Min samples to split'),
        ('min_samples_leaf',  'slider',     1, 10, 1,     'Min samples per leaf'),
    ],
    'logistic_regression': [
        ('C',        'float',  0.001, 100.0, 1.0,          'Regularisation strength (C)'),
        ('max_iter', 'slider', 100,   2000,  500,           'Max iterations'),
        ('solver',   'select', ['liblinear', 'lbfgs', 'saga', 'newton-cg'], 'liblinear', 'Solver'),
    ],
    'random_forest': [
        ('n_estimators',      'slider',     10,  500,  100, 'Number of trees'),
        ('max_depth',         'slider_opt', 1,   30,   None,'Max depth (None = unlimited)'),
        ('min_samples_split', 'slider',     2,   20,   2,   'Min samples to split'),
    ],
    'gradient_boosting': [
        ('n_estimators',   'slider', 10,   500,  100,  'Number of estimators'),
        ('learning_rate',  'float',  0.001, 1.0,  0.1,  'Learning rate'),
        ('max_depth',      'slider', 1,    10,   3,    'Max depth'),
    ],
    'svm': [
        ('C',      'float',  0.001, 100.0, 1.0,                          'Regularisation (C)'),
        ('kernel', 'select', ['rbf', 'linear', 'poly', 'sigmoid'], 'rbf', 'Kernel'),
    ],
    'knn': [
        ('n_neighbors', 'slider', 1,  50, 5,                    'Number of neighbours'),
        ('weights',     'select', ['uniform', 'distance'], 'uniform', 'Weight function'),
    ],
}

_SCORERS = {
    'accuracy':  'Accuracy',
    'f1':        'F1 Score (weighted)',
    'precision': 'Precision (weighted)',
    'recall':    'Recall (weighted)',
    'roc_auc':   'ROC-AUC',
}


def _render_hyperparams(model_key: str) -> dict:
    """Render hyperparameter widgets for given model and return param dict."""
    defs = _HYPERPARAM_DEFS.get(model_key, [])
    if not defs:
        st.caption("No configurable hyperparameters for this model.")
        return {}

    params = {}
    for item in defs:
        name, kind = item[0], item[1]
        label = item[-1]

        if kind == 'slider':
            lo, hi, default = item[2], item[3], item[4]
            params[name] = st.slider(label, lo, hi, default, key=f'hp_{model_key}_{name}')

        elif kind == 'slider_opt':
            lo, hi, default = item[2], item[3], item[4]
            use_none = st.checkbox(f"{label}: unlimited", value=(default is None),
                                   key=f'hp_{model_key}_{name}_none')
            if use_none:
                params[name] = None
            else:
                params[name] = st.slider(label, lo, hi, lo if default is None else default,
                                          key=f'hp_{model_key}_{name}')

        elif kind == 'float':
            lo, hi, default = item[2], item[3], item[4]
            params[name] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                            value=float(default), step=float(lo),
                                            key=f'hp_{model_key}_{name}')

        elif kind == 'select':
            options, default = item[2], item[3]
            idx = options.index(default) if default in options else 0
            params[name] = st.selectbox(label, options, index=idx,
                                         key=f'hp_{model_key}_{name}')
    return params


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render():
    """Render model training page."""
    state = get_state()

    st.title("Model Training")

    if not state.has_labeled_data():
        st.warning("Please upload labeled data first.")
        return

    df = state.labeled_data

    col1, col2 = st.columns(2)

    with col1:
        render_model_selection(state, df)

    with col2:
        render_feature_selection(df)

    render_training_section(state, df)


def render_model_selection(state, df):
    """Render model selection + hyperparameter panel."""
    st.subheader("Model Selection")

    models = state.model_service.list_available_models('classifier')
    model_options = {m['key']: m['name'] for m in models}

    model_key = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )

    model_info = next(m for m in models if m['key'] == model_key)
    st.caption(model_info['description'])

    target_col = st.selectbox(
        "Target Column",
        [c for c in df.columns if df[c].dtype == 'object']
    )
    positive_label = st.text_input("Positive label", value="positif")

    # Hyperparameter panel
    with st.expander("⚙️ Hyperparameters", expanded=False):
        hyperparams = _render_hyperparams(model_key)

    # Store in session
    st.session_state.model_key = model_key
    st.session_state.target_col = target_col
    st.session_state.positive_label = positive_label
    st.session_state.hyperparams = hyperparams


def render_feature_selection(df):
    """Render feature selection controls."""
    st.subheader("Feature Selection")

    target_col = st.session_state.get('target_col', 'risque')
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    preset = st.selectbox(
        "Feature Preset",
        ["All Features", "Course Features (11)", "Simple (3)", "Custom"]
    )

    if preset == "All Features":
        selected = numeric_cols
    elif preset == "Course Features (11)":
        course = ['nombre', 'cnbripdst', 'cnportdst', 'permit',
                  'inf1024permit', 'sup1024permit', 'adminpermit',
                  'deny', 'inf1024deny', 'sup1024deny', 'admindeny']
        selected = [f for f in course if f in numeric_cols]
    elif preset == "Simple (3)":
        selected = [f for f in ['nombre', 'cnbripdst', 'cnportdst']
                    if f in numeric_cols]
    else:
        selected = st.multiselect("Features", numeric_cols, default=numeric_cols[:5])

    st.write(f"Selected {len(selected)} features")
    st.session_state.selected_features = selected


def render_training_section(state, df):
    """Render training controls and results."""
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        scale = st.checkbox("Scale features", value=True)
        cv_folds = st.slider("CV folds", 2, 10, 5)

    with col2:
        eval_method = st.radio("Evaluation", ["Cross-Validation", "Leave-One-Out"])

    with col3:
        scoring = st.selectbox(
            "Optimise for",
            list(_SCORERS.keys()),
            format_func=lambda k: _SCORERS[k]
        )

    if st.button("Train Model", type="primary"):
        model_key  = st.session_state.get('model_key', 'logistic_regression')
        target_col = st.session_state.get('target_col', 'risque')
        features   = st.session_state.get('selected_features', [])
        hyperparams = st.session_state.get('hyperparams', {})

        try:
            # Cross-validate
            with st.spinner("Running cross-validation…"):
                cv_results = state.model_service.cross_validate(
                    df,
                    model_key=model_key,
                    feature_cols=features,
                    cv=cv_folds,
                    use_loo=(eval_method == "Leave-One-Out"),
                    scoring=scoring,
                    **hyperparams,
                )

            # Train full model
            with st.spinner("Training model…"):
                train_results = state.model_service.train(
                    df,
                    model_key=model_key,
                    feature_cols=features,
                    target_col=target_col,
                    scale_features=scale,
                    **hyperparams,
                )

            state.training_results = {'cv': cv_results, 'train': train_results}
            st.success("Model trained!")
            st.toast("Training complete!", icon="✅")

            metric_label = _SCORERS.get(cv_results.get('scoring', scoring), scoring.capitalize())
            cols = st.columns(4)
            cols[0].metric(metric_label, f"{cv_results['mean']:.3f}")
            cols[1].metric("Std Dev", f"{cv_results['std']:.3f}")
            cols[2].metric("CV method", cv_results.get('cv_method', ''))
            cols[3].metric("Features used", len(features))

            # Per-fold breakdown
            with st.expander("Per-fold scores"):
                fold_scores = cv_results.get('scores', [])
                for i, s in enumerate(fold_scores, 1):
                    st.write(f"Fold {i}: {s:.4f}")

            # Feature importance
            try:
                importance = state.model_service.get_feature_importance()
                st.subheader("Feature Importance")
                fig = px.bar(
                    importance.head(10),
                    x='importance', y='feature',
                    orientation='h', title='Top 10 Features'
                )
                st.plotly_chart(fig)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
