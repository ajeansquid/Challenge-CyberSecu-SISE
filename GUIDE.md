# Projet SISE-OPSIE 2026 — Developer & User Guide

**Audience**: DS / ML / SWE students building the ML pipeline; cybersecurity partners providing log data.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Architecture](#2-architecture)
3. [Data Format](#3-data-format)
4. [Workflow Walkthrough](#4-workflow-walkthrough)
5. [The 11 Course Features](#5-the-11-course-features)
6. [Model Selection Guide](#6-model-selection-guide)
7. [Evaluation & Interpreting Results](#7-evaluation--interpreting-results)
8. [Configuration Reference](#8-configuration-reference)
9. [Programmatic API](#9-programmatic-api)
10. [Extension Points](#10-extension-points)
11. [ELK Stack Alternative](#11-elk-stack-alternative)
12. [Project Structure](#12-project-structure)

---

## 1. Project Context

### The Challenge

Teams of **cybersecurity students** capture network traffic from a monitored application or lab environment and export the logs (typically firewall rules hit logs or syslog output). Teams of **data science / ML / SWE students** then:

1. Parse and clean the logs
2. Aggregate per-IP behavioral features
3. Train a binary classifier: is an IP `positif` (suspicious/attacking) or `negatif` (normal)?
4. Build and present a working dashboard showing model performance and live predictions

This app is the ML team's deliverable — a modular, layered codebase where every component (parser, feature extractor, model, evaluation) can be swapped or extended without touching the rest.

### Team Responsibilities

**Cybersecurity partners should provide:**
- Raw log files in a known format (see [Data Format](#3-data-format))
- A labeled sample if possible: a CSV/Excel file where each row is a **source IP** with a `risque` column (`positif` or `negatif`)
- Documentation on the test environment: what traffic is normal, what attacks were simulated, which ports matter

**ML/DS team responsibilities:**
- Adapt the parser to the actual log format (or use the existing `firewall` / `syslog` parsers)
- Validate feature semantics with the cybersec team (does `admindeny` on port 22 actually mean an SSH brute-force attempt in their setup?)
- Train, evaluate, and tune models on the labeled data
- Build the final presentation showing precision, recall, ROC-AUC, and example flagged IPs

---

## 2. Architecture

The codebase enforces a **strict 7-layer dependency order**. Each layer may only import from layers below it. This ensures that, for example, adding a new parser never requires touching the UI, and swapping models never requires changing feature logic.

```
┌─────────────────────────────────────────┐
│  Layer 7 — App (Streamlit UI)           │  app/pages/*.py, app/state.py
├─────────────────────────────────────────┤
│  Layer 6 — Services (orchestration)     │  services/
├─────────────────────────────────────────┤
│  Layer 5 — Evaluation                  │  evaluation/
├──────────────────┬──────────────────────┤
│  Layer 4 — Models│  Layer 3 — Features  │  models/ , features/
├──────────────────┴──────────────────────┤
│  Layer 2 — Parsers                      │  parsers/
├─────────────────────────────────────────┤
│  Layer 1 — Core (interfaces, config)    │  core/
└─────────────────────────────────────────┘
```

### Key Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Factory** | `parsers/factory.py` | Register/create parsers by string key |
| **Registry** | `models/registry.py` | Register/create ML models by string key |
| **Pipeline** | `models/pipeline.py` | Wrap any model with preprocessing (scaling, label encoding) |
| **Typed session state** | `app/state.py` | Single `AppState` object wraps all Streamlit session state |
| **Service layer** | `services/` | All UI pages go through services, never call models/parsers directly |

### State Management

All cross-page data in the Streamlit UI flows through `AppState` (`app/state.py`). Never read from or write to `st.session_state` directly in page code. Use:

```python
from app.state import get_state
state = get_state()
state.raw_data          # the loaded DataFrame
state.features_data     # the FeatureSet
state.training_results  # dict of metrics from last training run
```

---

## 3. Data Format

### Firewall Logs (primary format)

The `FirewallParser` (parser key: `"firewall"`) expects a CSV with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `ipsrc` | string | Source IP address |
| `ipdst` | string | Destination IP address |
| `portdst` | integer | Destination port |
| `proto` | string | Protocol (TCP, UDP, ICMP…) |
| `action` | string | Firewall decision: `Permit` or `Deny` |
| `date` | datetime | Timestamp of the event |
| `regle` | string | Firewall rule that matched |

The parser tolerates variant action values (`permit`, `PERMIT`, `allow`, `Accept` → normalized to `Permit`; `deny`, `DENY`, `block`, `Drop` → normalized to `Deny`).

**Minimal required columns**: `ipsrc`, `ipdst`, `portdst`, `action`. Other columns are optional.

**Example rows:**
```
ipsrc,ipdst,portdst,proto,action,date,regle
192.168.1.45,10.0.0.1,22,TCP,Deny,2024-01-15 08:23:11,RULE-SSH-BLOCK
192.168.1.45,10.0.0.1,80,TCP,Permit,2024-01-15 08:23:15,RULE-HTTP-ALLOW
10.0.0.99,10.0.0.1,3306,TCP,Deny,2024-01-15 08:24:00,RULE-DB-BLOCK
```

### Labeled Data (for supervised training)

A separate file where each row represents a **source IP** (already aggregated) with a label column:

| Column | Type | Description |
|--------|------|-------------|
| `ipsrc` | string | Source IP (index) |
| `nombre` | int | Total connections |
| ... | ... | The 11 course features (see Section 5) |
| `risque` | string | Label: `"positif"` or `"negatif"` |

This file is typically produced by running feature extraction on a batch of logs for which the cybersec team has confirmed which IPs were attackers.

### Syslog Format

The `SyslogParser` (parser key: `"syslog"`) handles standard RFC 3164 / RFC 5424 syslog output:

```
Jan 15 08:23:11 hostname process[pid]: message content
```

Extracted fields: `timestamp`, `hostname`, `process`, `pid`, `message`. Note that syslog does not always carry the structured `portdst`/`action` fields needed for the 11 course features — you may need to parse the message field further or write a custom parser.

### Generic CSV

The `GenericCSVParser` (parser key: `"csv"`) accepts any CSV with configurable column remapping. Useful if your cybersec partner exports logs in a different schema.

---

## 4. Workflow Walkthrough

### Step 1 — Data Upload

**Page**: *Data Upload*

Upload two files:
- **Raw logs** (CSV): the full firewall log from the cybersec partner. Use parser type `firewall` unless the format differs.
- **Labeled data** (CSV or Excel): contains the 11 features + `risque` label for the training set.

The page shows a preview and basic statistics (row count, null values, column types). Confirm the parser detected actions correctly in the summary.

**Tip**: If the page shows 0 `Permit` or 0 `Deny` after parsing, the action column uses unexpected values. Check the raw file and either adjust the data or add the value to `FirewallParser._action_mapping` in `parsers/firewall.py`.

---

### Step 2 — Feature Engineering

**Page**: *Feature Engineering*

Choose an extractor:

| Extractor | Features | Use When |
|-----------|----------|----------|
| **Course (11)** | The 11 fixed course features | Always — required for grading |
| **Full** | 11 + time-based + ratios + stats | When you want richer features for experimentation |
| **Simple (3)** | `nombre`, `cnbripdst`, `cnportdst` | Quick sanity check or very small datasets |

Map the columns from your raw logs to the expected roles (`ip_col`, `dst_col`, `port_col`, `action_col`). The extractor groups all log lines by source IP and computes aggregated features.

Optionally apply **scaling** (Standard, MinMax, Robust) before training. Scaling is required if you plan to use SVM or KNN.

---

### Step 3 — Model Training

**Page**: *Model Training*

Select a model from the registry and configure hyperparameters. Click **Train**. The pipeline:

1. Loads the labeled feature DataFrame
2. Encodes the label column (`positif` → 1, `negatif` → 0)
3. Optionally scales features
4. Fits the model
5. Runs k-fold cross-validation (default: 5 folds)
6. Reports accuracy, precision, recall, F1, AUC

The sidebar shows current status: which model is trained, when.

**Saving the model**: Use the export button on this page, or via API: `model_svc.pipeline.save('my_model.pkl')`.

---

### Step 4 — Predictions

**Page**: *Predictions*

Upload new, **unlabeled** firewall logs. The service:

1. Parses the log file
2. Runs feature extraction with the same settings as training
3. Applies the trained pipeline (same scaling, same feature order)
4. Returns a DataFrame with an `ipsrc` column, a `prediction` column (`positif`/`negatif`), and a `probability` column (confidence score)

Export the prediction results as CSV or Excel for inclusion in your report/presentation.

---

### Step 5 — Analysis Dashboard

**Page**: *Analysis*

Explore:
- Raw data distributions (action counts, port histograms, top source IPs)
- Model performance: ROC curve, precision-recall curve, confusion matrix
- Feature importances (tree-based models)
- Side-by-side model comparison if you've trained multiple models

---

## 5. The 11 Course Features

All features are **aggregated per source IP** from the full log file. This means even if an IP appears in 10,000 log lines, it produces exactly one row of features.

| Feature | Formula / Meaning | Cybersecurity Relevance |
|---------|------------------|------------------------|
| `nombre` | Total log entries for this IP | High volume can indicate scanning or DDoS |
| `cnbripdst` | Count of unique destination IPs | Scanning across the network → high value |
| `cnportdst` | Count of unique destination ports | Port scanning → high value |
| `permit` | Count of permitted connections | Normal traffic volume baseline |
| `inf1024permit` | Permitted connections to ports < 1024 (system ports) | Access to web (80/443), SSH (22), FTP (21)… |
| `sup1024permit` | Permitted connections to ports ≥ 1024 (ephemeral/app ports) | Application-layer traffic |
| `adminpermit` | Permitted connections to admin ports (22, 23, 3389, 3306, etc.) | Successful access to sensitive services |
| `deny` | Count of denied connections | Repeated denies suggest blocked attack attempts |
| `inf1024deny` | Denied connections to system ports | Blocked attempts on web/infrastructure ports |
| `sup1024deny` | Denied connections to high ports | Blocked application-layer attempts |
| `admindeny` | Denied connections to admin ports | Blocked brute-force / unauthorized admin access |

**Admin ports** (configurable in `config/settings.yaml`):
`21` (FTP), `22` (SSH), `23` (Telnet), `3389` (RDP), `3306` (MySQL), `5432` (PostgreSQL), `1433` (MSSQL), `445` (SMB), `139` (NetBIOS)

**Port threshold**: `1024` separates system/well-known ports from ephemeral/application ports.

**Interpretation example**: An IP with `deny=500`, `admindeny=490`, `nombre=500` is almost entirely composed of denied admin connections — a strong indicator of SSH/RDP brute-force.

---

## 6. Model Selection Guide

### When You Have Labeled Data

Use a **classifier**. The cybersec team should provide confirmed positive (attacking) IPs for training.

| Model | Strengths | Watch Out For |
|-------|-----------|---------------|
| **Random Forest** | Robust, handles imbalanced classes, built-in feature importance | Can overfit on tiny datasets |
| **Gradient Boosting** | Often best accuracy, handles nonlinear patterns | Slower to train, more hyperparameters |
| **Decision Tree** | Explainable, very fast, easy to present | Overfits without pruning (`max_depth`) |
| **Logistic Regression** | Simple baseline, probabilistic, fast | Assumes linear separability |
| **SVM** | Strong on small datasets with clear margin | Requires scaling, slow on large data |
| **KNN** | Intuitive, no training phase | Requires scaling, slow at prediction, sensitive to imbalance |

**Recommended starting point**: Random Forest with default parameters, then compare with Gradient Boosting.

### When You Only Have Raw Traffic (No Labels)

Use an **anomaly detector**. These models learn "normal" behavior and flag outliers.

| Model | Approach | Use When |
|-------|----------|----------|
| **Isolation Forest** | Randomly isolates points; anomalies are easier to isolate | Good general-purpose default |
| **One-Class SVM** | Fits a boundary around normal data | When the attack pattern is geometrically distinct |

Anomaly detection output: `1` = normal, `-1` = anomaly (following scikit-learn convention). The service maps this to `negatif`/`positif`.

### Class Imbalance

In real firewall logs, attacking IPs are much rarer than normal ones (e.g., 5% positive, 95% negative). This will hurt naive classifiers. Strategies:

- Use `class_weight='balanced'` (available for LR, SVM, RF, Decision Tree) — can set via hyperparameter UI
- Use **F1-score and AUC** as primary metrics, not accuracy
- Consider the **precision/recall trade-off**: for intrusion detection, high recall (catch all attacks) is usually more important than high precision

---

## 7. Evaluation & Interpreting Results

### Key Metrics

| Metric | Meaning | Priority For IDS |
|--------|---------|-----------------|
| **Accuracy** | % of all predictions correct | Low priority (misleading with imbalanced data) |
| **Precision** | Of all IPs flagged as `positif`, how many truly are | Important (avoid false accusations) |
| **Recall (Sensitivity)** | Of all true attacks, how many did we catch | **High priority** — missing an attack is costly |
| **F1 Score** | Harmonic mean of precision and recall | Good single-number summary |
| **AUC-ROC** | Area under the ROC curve; model's discrimination ability | Target > 0.85 for a usable model |
| **Specificity** | Of normal IPs, how many did we correctly clear | Also important — avoid alarming normal users |

### Confusion Matrix

```
                Predicted Positif  | Predicted Negatif
Actual Positif  True Positive (TP) | False Negative (FN)  ← missed attacks
Actual Negatif  False Positive (FP)| True Negative (TN)
```

For intrusion detection: **minimize FN** (missed attacks) even at the cost of more FP (false alarms), unless the cost of a false alarm is very high in your environment.

### Cross-Validation

With small labeled datasets (< 500 IPs), 5-fold cross-validation gives a more realistic estimate than a single train/test split. The app runs CV automatically during training and reports mean ± std for each metric.

If your dataset is very small (< 50 labeled IPs), enable **Leave-One-Out CV** in the training page options.

### Presenting Results

Good things to include in the final presentation:
1. ROC curve across multiple models (shows relative performance)
2. Confusion matrix for the best model (shows failure modes)
3. Feature importance plot (shows which network behaviors matter most)
4. A table of the top 10 predicted `positif` IPs with their key feature values
5. Cross-validation table: mean F1 and AUC per model

---

## 8. Configuration Reference

All defaults live in `config/settings.yaml`.

```yaml
parser:
  default_separator: ","
  encoding: "utf-8"
  # Columns expected in firewall logs
  firewall_columns: ["ipsrc", "ipdst", "portdst", "proto", "action", "date", "regle"]

features:
  admin_ports: [21, 22, 23, 3389, 3306, 5432, 1433, 445, 139]
  port_threshold: 1024

model:
  random_state: 42          # Reproducibility seed
  positive_label: "positif"
  negative_label: "negatif"
  target_column: "risque"
  cv_folds: 5

app:
  title: "SISE-OPSIE 2026"
  layout: "wide"
  max_upload_size_mb: 200
```

Load/access from code:

```python
from core.config import get_config

config = get_config()
admin_ports = set(config.features.admin_ports)   # {21, 22, 23, ...}
port_threshold = config.features.port_threshold  # 1024
```

---

## 9. Programmatic API

All services can be used independently of Streamlit — useful for scripts, notebooks, or batch processing.

### Full Supervised Pipeline

```python
from services import DataService, FeatureService, ModelService, EvaluationService

data_svc  = DataService()
feat_svc  = FeatureService()
model_svc = ModelService()
eval_svc  = EvaluationService()

# 1. Parse raw logs
raw_df = data_svc.load_raw_logs("firewall_logs.csv", parser_type="firewall")

# 2. Extract the 11 course features
feature_set = feat_svc.extract_course_features(
    raw_df,
    ip_col="ipsrc",
    dst_col="ipdst",
    port_col="portdst",
    action_col="action"
)

# 3. Load labeled data (features + risque column)
labeled_df = data_svc.load_labeled_data("labeled_ips.csv")

# 4. Train a model
model_svc.train(
    labeled_df,
    model_key="random_forest",
    feature_cols=feature_set.feature_names,
    target_col="risque",
    scale_features=False,
    n_estimators=200,
    class_weight="balanced"
)

# 5. Predict on new data
result = model_svc.predict(feature_set.data, feature_set.feature_names)
# result.predictions  → array of "positif" / "negatif"
# result.probabilities → array of floats [0, 1]

# 6. Evaluate
metrics = eval_svc.evaluate(labeled_df["risque"], result.predictions, result.probabilities)
print(metrics)  # {"accuracy": 0.94, "f1": 0.88, "auc": 0.96, ...}
```

### Anomaly Detection (No Labels)

```python
# Fit on "normal" traffic only, then score all IPs
model_svc.anomaly_detect(feature_set.data)
# Returns PredictionResult with predictions: 1 (normal) or -1 (anomaly)
```

### Saving and Loading Models

```python
# Save after training
model_svc.pipeline.save("models/rf_v1.pkl")

# Load in another session / script
from models.pipeline import ModelPipeline
pipeline = ModelPipeline.load("models/rf_v1.pkl")
result = pipeline.predict_full(new_feature_df)
```

---

## 10. Extension Points

### Adding a New Parser

Use this when the cybersec partner provides logs in a format other than the default firewall CSV or syslog.

```python
# parsers/my_parser.py
import pandas as pd
from parsers.base import BaseParser

class MyParser(BaseParser):
    @property
    def expected_columns(self):
        return ["src_ip", "dst_port", "verdict"]

    def _parse_impl(self, source: str) -> pd.DataFrame:
        df = pd.read_csv(source, sep="|")
        # Rename to standard names
        df = df.rename(columns={"src_ip": "ipsrc", "dst_port": "portdst", "verdict": "action"})
        return df
```

Register in `parsers/factory.py`:

```python
from parsers.my_parser import MyParser
ParserFactory._parsers["my_format"] = MyParser
```

Use: `data_svc.load_raw_logs("file.log", parser_type="my_format")`

---

### Adding a New Model

```python
# models/my_model.py
from sklearn.ensemble import ExtraTreesClassifier
from models.classifiers import BaseClassifier

class ExtraTreesModel(BaseClassifier):
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__(ExtraTreesClassifier(n_estimators=n_estimators, **kwargs))
```

Register in `models/registry.py`:

```python
from models.my_model import ExtraTreesModel

ModelRegistry._models["extra_trees"] = ModelInfo(
    key="extra_trees",
    name="Extra Trees",
    model_class=ExtraTreesModel,
    model_type="classifier",
    description="Extremely randomized trees — faster than Random Forest, often similar accuracy"
)
```

---

### Adding a New Streamlit Page

```python
# app/pages/my_page.py
import streamlit as st
from app.state import get_state

def render():
    state = get_state()
    st.title("My Page")
    if not state.has_features():
        st.warning("Extract features first.")
        return
    # ... page logic using state.features_data
```

Add to the router in `main.py`:

```python
from app.pages import my_page

# In the page switch block:
elif page == "My Page":
    my_page.render()
```

---

### Custom Error Handling

Always raise and catch from `core/exceptions.py`:

```python
from core.exceptions import ParsingError, FeatureExtractionError, ModelError

# Raise:
raise ParsingError("Expected column 'action' not found", source="logs.csv")

# Catch at service level:
try:
    result = parser.parse(file_path)
except ParsingError as e:
    st.error(f"Failed to parse file: {e}")
```

---

## 11. ELK Stack Alternative

`docs/courses/ElasticSearch/` contains a Docker Compose setup for running the ELK stack (Elasticsearch, Logstash, Kibana) as an alternative dashboard.

| Approach | Strengths | Weaknesses |
|----------|-----------|-----------|
| **This app (Streamlit)** | Native ML, custom models, full Python control | No real-time ingestion |
| **ELK** | Real-time log ingestion, powerful search, Kibana dashboards | Less flexible for custom ML models |
| **Hybrid** | Use Logstash for ingestion → export CSVs → train here | More setup but best of both worlds |

For the course presentation, both approaches are acceptable. The ELK setup in `docs/courses/ElasticSearch/` includes sample Logstash pipeline configs and Kibana dashboard exports.

---

## 12. Project Structure

```
sise_opsie_2026/
├── main.py                          # Streamlit entry point
├── run.py                           # Launcher (--port, --host, --debug)
├── requirements.txt                 # pip dependencies
├── pyproject.toml                   # Project metadata
│
├── config/
│   └── settings.yaml                # All configurable defaults
│
├── core/                            # Layer 1 — foundations
│   ├── interfaces.py                # Abstract base classes (Parser, Classifier, etc.)
│   ├── exceptions.py                # Custom exceptions
│   └── config.py                    # Config dataclasses + YAML loader
│
├── parsers/                         # Layer 2 — log parsing
│   ├── base.py                      # BaseParser (encoding detection, validation)
│   ├── firewall.py                  # Firewall CSV parser
│   ├── generic.py                   # Flexible CSV parser
│   ├── syslog.py                    # RFC 3164/5424 syslog parser
│   └── factory.py                   # ParserFactory registry
│
├── features/                        # Layer 3 — feature engineering
│   ├── aggregators.py               # IPAggregator (11 features), TimeAggregator, StatAggregator
│   ├── extractors.py                # CourseFeatureExtractor, FullFeatureExtractor, SimpleFeatureExtractor
│   ├── transformers.py              # Scaler, FeatureSelector, RatioTransformer
│   └── store.py                     # FeatureStore (in-memory + disk)
│
├── models/                          # Layer 4 — ML models
│   ├── registry.py                  # ModelRegistry (10 registered models)
│   ├── pipeline.py                  # ModelPipeline (preprocessing + fit/predict/CV)
│   ├── classifiers.py               # DT, LR, RF, GB, SVM, KNN
│   ├── anomaly.py                   # IsolationForest, OneClassSVM
│   └── clustering.py                # KMeans, DBSCAN
│
├── evaluation/                      # Layer 5 — evaluation
│   ├── metrics.py                   # MetricsCalculator (accuracy, F1, AUC, etc.)
│   ├── visualizations.py            # EvaluationPlotter (ROC, confusion matrix)
│   └── comparison.py                # ModelComparator
│
├── services/                        # Layer 6 — orchestration
│   ├── data_service.py              # Load / save / validate data
│   ├── feature_service.py           # Extract / scale / select features
│   ├── model_service.py             # Train / predict / evaluate
│   └── evaluation_service.py        # Full evaluation with plots
│
├── app/                             # Layer 7 — Streamlit UI
│   ├── state.py                     # AppState (typed session state)
│   └── pages/
│       ├── data_upload.py
│       ├── feature_engineering.py
│       ├── model_training.py
│       ├── predictions.py
│       └── analysis.py
│
├── utils/
│   ├── helpers.py
│   └── io.py
│
└── docs/
    └── courses/
        ├── Seance 1.a/              # Resampling, cross-validation (TD1.ipynb)
        ├── Seance 1.b/              # ROC curves (TD2.ipynb)
        ├── Seance 2.a/              # Anomaly detection, Isolation Forest (TD3.ipynb)
        ├── Seance 2.b/
        ├── Seance_3a/               # Firewall log parsing reference (parse_log.py)
        ├── Seance_3b/               # Attack detection with resampling
        └── ElasticSearch/           # Docker + ELK stack setup
```
