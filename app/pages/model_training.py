# -*- coding: utf-8 -*-
"""Model Training Page - Supervised and Unsupervised Learning"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from app.state import get_state

# ---------------------------------------------------------------------------
# Per-model hyperparameter definitions
# ---------------------------------------------------------------------------

_HYPERPARAM_DEFS = {
    # Supervised classifiers
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
    # Unsupervised - Anomaly detection
    'isolation_forest': [
        ('contamination', 'float', 0.01, 0.5, 0.1, 'Expected anomaly rate'),
        ('n_estimators',  'slider', 50, 500, 100,  'Number of trees'),
    ],
    'one_class_svm': [
        ('nu',     'float',  0.01, 0.5, 0.1,                              'Nu (anomaly fraction bound)'),
        ('kernel', 'select', ['rbf', 'linear', 'poly', 'sigmoid'], 'rbf', 'Kernel'),
    ],
    # Unsupervised - Clustering
    'kmeans': [
        ('n_clusters', 'slider', 2, 20, 5,    'Number of clusters'),
        ('n_init',     'slider', 1, 20, 10,   'Number of initializations'),
    ],
    'dbscan': [
        ('eps',         'float',  0.1, 10.0, 0.5, 'Epsilon (neighborhood radius)'),
        ('min_samples', 'slider', 2, 20, 5,       'Min samples per cluster'),
    ],
}

_SCORERS = {
    'accuracy':  'Accuracy',
    'f1':        'F1 Score (weighted)',
    'precision': 'Precision (weighted)',
    'recall':    'Recall (weighted)',
    'roc_auc':   'ROC-AUC',
}


def _render_hyperparams(model_key: str, key_prefix: str = 'hp') -> dict:
    """Render hyperparameter widgets for given model and return param dict."""
    defs = _HYPERPARAM_DEFS.get(model_key, [])
    if not defs:
        st.caption("No configurable hyperparameters for this model.")
        return {}

    params = {}
    for item in defs:
        name, kind = item[0], item[1]
        label = item[-1]
        widget_key = f'{key_prefix}_{model_key}_{name}'

        if kind == 'slider':
            lo, hi, default = item[2], item[3], item[4]
            params[name] = st.slider(label, lo, hi, default, key=widget_key)

        elif kind == 'slider_opt':
            lo, hi, default = item[2], item[3], item[4]
            use_none = st.checkbox(f"{label}: unlimited", value=(default is None),
                                   key=f'{widget_key}_none')
            if use_none:
                params[name] = None
            else:
                params[name] = st.slider(label, lo, hi, lo if default is None else default,
                                          key=widget_key)

        elif kind == 'float':
            lo, hi, default = item[2], item[3], item[4]
            params[name] = st.number_input(label, min_value=float(lo), max_value=float(hi),
                                            value=float(default), step=float(lo),
                                            key=widget_key)

        elif kind == 'select':
            options, default = item[2], item[3]
            idx = options.index(default) if default in options else 0
            params[name] = st.selectbox(label, options, index=idx, key=widget_key)
    return params


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------

def render():
    """Render model training page with Supervised and Unsupervised tabs."""
    state = get_state()

    st.title("Model Training")

    # Tabs for supervised vs unsupervised
    tab_supervised, tab_unsupervised = st.tabs([
        "🎯 Supervised (requires labels)",
        "🔍 Unsupervised (no labels needed)"
    ])

    with tab_supervised:
        render_supervised_tab(state)

    with tab_unsupervised:
        render_unsupervised_tab(state)


# ---------------------------------------------------------------------------
# Supervised Tab
# ---------------------------------------------------------------------------

def render_supervised_tab(state):
    """Render supervised learning tab."""
    st.markdown("Train classification models on labeled data with a target column (e.g., 'risk').")

    if not state.has_labeled_data():
        st.warning(
            "**Labeled data required.** Upload a dataset with a target column (e.g., 'risk') via Data Upload."
        )
        return

    df = state.labeled_data

    col1, col2 = st.columns(2)

    with col1:
        render_supervised_model_selection(state, df)

    with col2:
        render_supervised_feature_selection(df)

    render_supervised_training_section(state, df)


def render_supervised_model_selection(state, df):
    """Render model selection + hyperparameter panel for supervised."""
    st.subheader("Model Selection")

    models = state.model_service.list_available_models('classifier')
    model_options = {m['key']: m['name'] for m in models}

    model_key = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        key='sup_model_key'
    )

    model_info = next(m for m in models if m['key'] == model_key)
    st.caption(model_info['description'])

    target_col = st.selectbox(
        "Target Column",
        [c for c in df.columns if df[c].dtype == 'object'],
        key='sup_target_col'
    )
    positive_label = st.text_input("Positive label", value="positive", key='sup_positive_label')

    # Hyperparameter panel
    with st.expander("⚙️ Hyperparameters", expanded=False):
        hyperparams = _render_hyperparams(model_key, key_prefix='sup_hp')

    # Store in session
    st.session_state.sup_model_key = model_key
    st.session_state.sup_target_col = target_col
    st.session_state.sup_positive_label = positive_label
    st.session_state.sup_hyperparams = hyperparams


def render_supervised_feature_selection(df):
    """Render feature selection controls for supervised."""
    st.subheader("Feature Selection")

    target_col = st.session_state.get('sup_target_col', 'risk')
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    preset = st.selectbox(
        "Feature Preset",
        ["All Features", "Course Features (11)", "Simple (3)", "Custom"],
        key='sup_preset'
    )

    if preset == "All Features":
        selected = numeric_cols
    elif preset == "Course Features (11)":
        course = ['total_flows', 'unique_dst_ips', 'unique_dst_ports', 'permit',
                  'permit_low_port', 'permit_high_port', 'permit_admin',
                  'deny', 'deny_low_port', 'deny_high_port', 'deny_admin']
        selected = [f for f in course if f in numeric_cols]
    elif preset == "Simple (3)":
        selected = [f for f in ['total_flows', 'unique_dst_ips', 'unique_dst_ports']
                    if f in numeric_cols]
    else:
        selected = st.multiselect("Features", numeric_cols, default=numeric_cols[:5], key='sup_custom_feats')

    st.write(f"Selected {len(selected)} features")
    st.session_state.sup_selected_features = selected


def render_supervised_training_section(state, df):
    """Render training controls and results for supervised."""
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        scale = st.checkbox("Scale features", value=True, key='sup_scale')
        cv_folds = st.slider("CV folds", 2, 10, 5, key='sup_cv_folds')

    with col2:
        eval_method = st.radio("Evaluation", ["Cross-Validation", "Leave-One-Out"], key='sup_eval')

    with col3:
        scoring = st.selectbox(
            "Optimise for",
            list(_SCORERS.keys()),
            format_func=lambda k: _SCORERS[k],
            key='sup_scoring'
        )

    if st.button("Train Model", type="primary", key='sup_train_btn'):
        model_key  = st.session_state.get('sup_model_key', 'logistic_regression')
        target_col = st.session_state.get('sup_target_col', 'risk')
        features   = st.session_state.get('sup_selected_features', [])
        hyperparams = st.session_state.get('sup_hyperparams', {})

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

            state.training_results = {'cv': cv_results, 'train': train_results, 'type': 'supervised'}
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


# ---------------------------------------------------------------------------
# Unsupervised Tab
# ---------------------------------------------------------------------------

def render_unsupervised_tab(state):
    """Render unsupervised learning tab."""
    st.markdown("Train anomaly detection or clustering models without labeled data.")

    # Check for features or labeled data (both work for unsupervised)
    if not state.has_features() and not state.has_labeled_data():
        st.warning(
            "**Feature data required.** Generate features via Feature Engineering, "
            "or upload labeled data with numeric features."
        )
        st.info("**Workflow:** Data Upload → Feature Engineering → Model Training (Unsupervised)")
        return

    # Use features_data if available, otherwise labeled_data
    if state.has_features():
        df = state.features_data
        st.info(f"Using generated features: {len(df):,} IPs × {len(df.columns)} features")
    else:
        df = state.labeled_data
        st.info(f"Using labeled data: {len(df):,} rows")

    # Model type selection
    model_type = st.radio(
        "Model Type",
        ["Anomaly Detection", "Clustering"],
        horizontal=True,
        key='unsup_model_type'
    )

    col1, col2 = st.columns(2)

    with col1:
        render_unsupervised_model_selection(state, model_type)

    with col2:
        render_unsupervised_feature_selection(df)

    st.markdown("---")

    if model_type == "Anomaly Detection":
        render_anomaly_training_section(state, df)
    else:
        render_clustering_training_section(state, df)


def render_unsupervised_model_selection(state, model_type: str):
    """Render model selection for unsupervised."""
    st.subheader("Model Selection")

    if model_type == "Anomaly Detection":
        models = state.model_service.list_available_models('anomaly')
    else:
        models = state.model_service.list_available_models('clustering')

    model_options = {m['key']: m['name'] for m in models}

    model_key = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
        key='unsup_model_key'
    )

    model_info = next(m for m in models if m['key'] == model_key)
    st.caption(model_info['description'])

    # Hyperparameter panel
    with st.expander("⚙️ Hyperparameters", expanded=True):
        hyperparams = _render_hyperparams(model_key, key_prefix='unsup_hp')

    st.session_state.unsup_hyperparams = hyperparams


def render_unsupervised_feature_selection(df):
    """Render feature selection for unsupervised."""
    st.subheader("Feature Selection")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    preset = st.selectbox(
        "Feature Preset",
        ["All Features", "Course Features (11)", "Simple (3)", "Custom"],
        key='unsup_preset'
    )

    if preset == "All Features":
        selected = numeric_cols
    elif preset == "Course Features (11)":
        course = ['total_flows', 'unique_dst_ips', 'unique_dst_ports', 'permit',
                  'permit_low_port', 'permit_high_port', 'permit_admin',
                  'deny', 'deny_low_port', 'deny_high_port', 'deny_admin']
        selected = [f for f in course if f in numeric_cols]
    elif preset == "Simple (3)":
        selected = [f for f in ['total_flows', 'unique_dst_ips', 'unique_dst_ports']
                    if f in numeric_cols]
    else:
        selected = st.multiselect("Features", numeric_cols, default=numeric_cols[:5], key='unsup_custom_feats')

    st.write(f"Selected {len(selected)} features")
    st.session_state.unsup_selected_features = selected


def render_anomaly_training_section(state, df):
    """Render anomaly detection training."""
    st.subheader("Anomaly Detection")

    model_key = st.session_state.get('unsup_model_key', 'isolation_forest')
    features = st.session_state.get('unsup_selected_features', [])
    hyperparams = st.session_state.get('unsup_hyperparams', {})

    if not features:
        st.warning("Select at least one feature.")
        return

    col1, col2 = st.columns(2)
    with col1:
        scale_features = st.checkbox("Scale features", value=True, key='unsup_scale')

    if st.button("Train Anomaly Detector", type="primary", key='unsup_anomaly_btn'):
        try:
            with st.spinner("Training anomaly detector…"):
                contamination = hyperparams.get('contamination', 0.1)

                # Use the model service to detect anomalies
                result_df = state.model_service.detect_anomalies(
                    df,
                    feature_cols=features,
                    contamination=contamination
                )

                # Store results
                state.unsupervised_results = {
                    'type': 'anomaly',
                    'model_key': model_key,
                    'features': features,
                    'result_df': result_df,
                }

            n_anomalies = result_df['is_anomaly'].sum()
            n_total = len(result_df)

            st.success(f"Anomaly detection complete! Found {n_anomalies:,} anomalies ({n_anomalies/n_total*100:.1f}%)")

            # Display metrics
            cols = st.columns(4)
            cols[0].metric("Total IPs", f"{n_total:,}")
            cols[1].metric("Anomalies", f"{n_anomalies:,}")
            cols[2].metric("Normal", f"{n_total - n_anomalies:,}")
            cols[3].metric("Anomaly Rate", f"{n_anomalies/n_total*100:.1f}%")

            # Score distribution
            st.subheader("Anomaly Score Distribution")
            fig = px.histogram(
                result_df, x='anomaly_score', color='is_anomaly',
                nbins=50, title='Anomaly Score Distribution',
                color_discrete_map={True: '#e74c3c', False: '#2ecc71'}
            )
            st.plotly_chart(fig)

            # Top anomalies
            st.subheader("Top Anomalous IPs")
            anomalies = result_df[result_df['is_anomaly']].sort_values('anomaly_score').head(20)
            st.dataframe(anomalies[features + ['anomaly_score', 'is_anomaly']], hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())


def render_clustering_training_section(state, df):
    """Render clustering training."""
    st.subheader("Clustering")

    model_key = st.session_state.get('unsup_model_key', 'kmeans')
    features = st.session_state.get('unsup_selected_features', [])
    hyperparams = st.session_state.get('unsup_hyperparams', {})

    if not features:
        st.warning("Select at least one feature.")
        return

    col1, col2 = st.columns(2)
    with col1:
        show_elbow = st.checkbox("Show elbow plot (K-Means only)", value=True, key='unsup_elbow')

    if st.button("Train Clustering Model", type="primary", key='unsup_cluster_btn'):
        try:
            # Elbow plot for K-Means
            if model_key == 'kmeans' and show_elbow:
                with st.spinner("Computing elbow plot…"):
                    inertias = state.model_service.find_optimal_clusters(df, features)

                st.subheader("Elbow Plot")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(inertias.keys()),
                    y=list(inertias.values()),
                    mode='lines+markers'
                ))
                fig.update_layout(xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
                st.plotly_chart(fig)

            with st.spinner("Training clustering model…"):
                n_clusters = hyperparams.get('n_clusters', 5)

                # Use the model service to cluster
                result_df = state.model_service.cluster(
                    df,
                    feature_cols=features,
                    n_clusters=n_clusters
                )

                # Store results
                state.unsupervised_results = {
                    'type': 'clustering',
                    'model_key': model_key,
                    'features': features,
                    'result_df': result_df,
                    'n_clusters': n_clusters,
                }

            st.success(f"Clustering complete! Found {result_df['cluster'].nunique()} clusters")

            # Cluster distribution
            cluster_counts = result_df['cluster'].value_counts().sort_index()
            cols = st.columns(min(len(cluster_counts), 5))
            for i, (cluster, count) in enumerate(cluster_counts.items()):
                if i < 5:
                    cols[i].metric(f"Cluster {cluster}", f"{count:,}")

            # Visualization (2D projection if more than 2 features)
            st.subheader("Cluster Visualization")
            if len(features) >= 2:
                fig = px.scatter(
                    result_df, x=features[0], y=features[1],
                    color=result_df['cluster'].astype(str),
                    title=f'Clusters: {features[0]} vs {features[1]}',
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig)

            # Cluster statistics
            st.subheader("Cluster Statistics")
            cluster_stats = result_df.groupby('cluster')[features].mean()
            st.dataframe(cluster_stats, width='stretch')

            # Cluster profile (radar chart)
            if len(features) >= 3:
                st.subheader("Cluster Profiles")
                # Normalize for radar
                normalized = cluster_stats.copy()
                for col in features:
                    max_val = normalized[col].max()
                    if max_val > 0:
                        normalized[col] = normalized[col] / max_val * 100

                fig = go.Figure()
                for cluster_id in normalized.index:
                    fig.add_trace(go.Scatterpolar(
                        r=[normalized.loc[cluster_id, f] for f in features],
                        theta=features,
                        fill='toself',
                        name=f'Cluster {cluster_id}'
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Normalized Cluster Profiles"
                )
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
