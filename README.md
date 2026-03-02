# Challenge Toolkit v2

A modular, layered toolkit for **ML-driven firewall log analysis** and **network intrusion detection**. Built as a reusable base project for the cybersecurity challenge module, where data/ML/SWE teams collaborate with cybersecurity partners to analyze network traffic and detect suspicious IP behavior.

---

## Academic Context

This project bridges two student profiles:

| Team | Role |
|------|------|
| **Cybersecurity students** | Provide raw network logs (firewall logs, syslog, etc.) from a monitored application or lab environment |
| **Data science / ML / SWE students** | Parse logs, engineer features, train classification models, build an interactive dashboard, and present findings |

The final deliverable is a working **web application or dashboard** (this toolkit, or an ELK-based alternative) analyzed network traffic, classifying source IPs as `positif` (suspicious/threatening) or `negatif` (benign).

---

## What This Toolkit Does

1. **Parses** raw firewall or syslog files into structured DataFrames
2. **Engineers** 11 course-standard features per source IP (connection counts, port patterns, permit/deny ratios)
3. **Trains** supervised classifiers (or runs unsupervised anomaly detectors) on labeled data
4. **Predicts** the risk level of IPs in new/unlabeled log files
5. **Visualizes** results (ROC curves, confusion matrices, feature importances) through a Streamlit dashboard

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit web app
python run.py

# Optional flags
python run.py --port 8502 --host 0.0.0.0 --debug
```

Navigate to `http://localhost:8501` in your browser.

---

## The 5-Step Workflow (UI)

```
[1] Data Upload  →  [2] Feature Engineering  →  [3] Model Training  →  [4] Predictions  →  [5] Analysis
```

1. **Data Upload** — Load raw firewall logs (CSV) and/or labeled data (Excel/CSV with a `risque` column)
2. **Feature Engineering** — Aggregate per-IP features from raw logs; choose course (11), full, or simple extractor
3. **Model Training** — Pick a classifier, tune hyperparameters, run cross-validation
4. **Predictions** — Upload new unlabeled logs and get risk predictions + probabilities
5. **Analysis** — Explore data distributions, evaluation metrics, and compare models

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web UI | Streamlit |
| ML | scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Serialization | joblib, PyYAML |
| Notebook examples | Jupyter (see `docs/courses/`) |

---

## Architecture

Strict 7-layer dependency order (each layer only imports from the layer below):

```
App (Streamlit UI)
  └── Services (orchestration)
        └── Models / Evaluation / Features
              └── Parsers
                    └── Core (interfaces, config, exceptions)
```

See [GUIDE.md](GUIDE.md) for a complete architecture reference.

---

## Expected Log Format

The default parser expects CSV firewall logs with these columns:

```
ipsrc, ipdst, portdst, proto, action, date, regle
```

- `action` values: `Permit` / `Deny` (or variants — the parser normalizes them)
- `portdst`: destination port (integer)

See [GUIDE.md — Data Format](GUIDE.md#data-format) for other supported formats and how to add new parsers.

---

## Available Models

| Type | Models |
|------|--------|
| Classifiers | Decision Tree, Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN |
| Anomaly Detection | Isolation Forest, One-Class SVM |
| Clustering | K-Means, DBSCAN |

When you have labeled data (`risque` column), use a **classifier**. When you only have raw traffic and want to flag outliers, use an **anomaly detector**.

---

## Course Materials

Jupyter notebooks and exercises for each session are in `docs/courses/`:

| Session | Topic |
|---------|-------|
| Seance 1.a | Resampling, cross-validation, evaluation metrics |
| Seance 1.b | ROC curves, AUC interpretation |
| Seance 2.a | Anomaly detection, Isolation Forest |
| Seance 2.b | Advanced exercises |
| Seance 3.a | Firewall log parsing (`parse_log.py` reference) |
| Seance 3.b | Attack detection with resampling |
| ElasticSearch | Docker setup and ELK stack alternative |

---

## Programmatic API

The toolkit can be used without the UI:

```python
from services import DataService, FeatureService, ModelService

data_svc = DataService()
feature_svc = FeatureService()
model_svc = ModelService()

raw_df    = data_svc.load_raw_logs('logs.csv', parser_type='firewall')
features  = feature_svc.extract_course_features(raw_df, ip_col='ipsrc', dst_col='ipdst',
                                                 port_col='portdst', action_col='action')
model_svc.train(labeled_df, model_key='random_forest',
                feature_cols=features.feature_names, target_col='risque')
result    = model_svc.predict(new_df, features.feature_names)
```

---

## Documentation

| File | Contents |
|------|---------|
| [GUIDE.md](GUIDE.md) | Full reference: architecture, data format, features, model selection, extension points |
| [CLAUDE.md](CLAUDE.md) | Instructions for Claude Code AI assistant |
| `docs/courses/` | Session notebooks and course exercises |
