# 🤖 Agentic ML Workflow — Framework

> **A config-driven, multi-agent ML pipeline framework.** Define your project in a single YAML file, and the framework handles data loading, feature engineering, model training, and evaluation — with optional LLM augmentation via Gemini, OpenAI, or Claude.

---

## Quick Start

```bash
# Clone
git clone https://github.com/arthanusamn/agentic-ml-workflow.git
cd agentic-ml-workflow

# Install core deps
pip install pandas scikit-learn pyyaml rich

# Run with sample data (no API key needed)
python run.py

# Create a new project template
python run.py init --name my_project
# Then edit my_project/config.yaml and run:
python run.py --config my_project/config.yaml
```

---

## How It Works

Every ML project is defined by a **single YAML config file**. The framework reads it, initializes the right backends, and runs a DAG of agents.

```
┌──────────────────────────────────────────────────────────────────┐
│                       Config-Driven Pipeline                      │
│                                                                  │
│  project_config.yaml ──► Framework ──► Agents ──► Results        │
│                              │                                    │
│                    ┌─────────┼─────────────┐                      │
│                    ▼         ▼             ▼                      │
│               DataBackend  LLMBackend   Orchestrator             │
│               (CSV|BQ|GCS) (Gemini|     (DAG runner)            │
│                            OpenAI|                               │
│                            Claude)                               │
└──────────────────────────────────────────────────────────────────┘
```

### The Agents

| Agent | Uses LLM? | What it does | Fallback without LLM |
|-------|-----------|-------------|----------------------|
| **Data** | ✅ Optional | Loads from CSV/BigQuery/GCS, profiles data quality via LLM, validates, splits | Deterministic pandas loading |
| **Features** | ✅ Optional | LLM writes custom feature engineering code based on dataset schema, then sklearn preprocessing runs | sklearn ColumnTransformer (scaling + encoding) |
| **Train** | ❌ No | Trains multiple candidates (RF, LogReg, XGBoost, LightGBM), picks best, logs to MLflow | Same — no LLM needed |
| **Eval** | ✅ Optional | LLM writes executive summary + 3 actionable recommendations | Rule-based thresholds + canned text |

---

## Project Config

Every project starts with a YAML config. Here's a complete example:

```yaml
# config.yaml

project:
  name: customer_churn
  description: "Predict customer churn from behavioral data"

data_source:
  type: csv                               # csv | bigquery | gcs
  path: "data.csv"                        # local or gs://bucket/file
  target_column: churned
  # For BigQuery:
  # type: bigquery
  # location: "my_project.my_dataset.my_table"
  # sql: "SELECT * FROM `my_project.my_dataset.my_table` WHERE date > '2025-01-01'"
  # credentials: /path/to/service-account.json

llm:
  provider: ""                            # gemini | openai | claude | "" (disabled)
  model: ""                               # e.g., gemini-1.5-pro, gpt-4o-mini
  api_key_env: ""                         # env var name, e.g., GEMINI_API_KEY

agents:
  data:
    enabled: true
    test_size: 0.2
    llm_analysis: false                   # LLM profiles data for quality issues

  features:
    enabled: true
    llm_engineering: false                # LLM writes feature engineering code
    max_transformations: 5

  train:
    enabled: true
    algorithms: [auto]                    # auto | specific list
    hyperopt_trials: 0

  eval:
    enabled: true
    llm_report: false                     # LLM generates report + recommendations
    thresholds:
      min_accuracy: 0.5
      min_r2: 0.0
```

---

## Running with Different Backends

### Local CSV (no LLM)
```bash
python run.py --config config.yaml
```

### BigQuery + Gemini
```bash
export GEMINI_API_KEY="your-key-here"
python run.py --config config.yaml \
  --llm-provider gemini \
  --api-key-env GEMINI_API_KEY \
  --model gemini-1.5-flash
```

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
python run.py --config config.yaml \
  --llm-provider openai \
  --api-key-env OPENAI_API_KEY
```

### Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python run.py --config config.yaml \
  --llm-provider claude \
  --api-key-env ANTHROPIC_API_KEY
```

### Local LLM (OpenAI-compatible endpoint)
```bash
python run.py --config config.yaml \
  --llm-provider openai \
  --api-key-env OPENAI_API_KEY \
  --model local-model
# Also set OPENAI_BASE_URL=http://localhost:1234/v1 in your env
```

---

## Creating a New Project

```bash
# Generate a template
python run.py init --name fraud_detection --desc "Detect credit card fraud"

# Edit the config
vim fraud_detection/config.yaml

# Run it
python run.py --config fraud_detection/config.yaml
```

---

## Project Structure

```
agentic-ml-workflow/
├── run.py                    # CLI: run pipeline or init project
├── requirements.txt          # Dependencies
├── README.md
├── framework/
│   ├── __init__.py
│   ├── artifact.py           # Typed artifacts (Data, Feature, Model, Eval)
│   ├── agent_base.py         # Agent base class + Context
│   ├── config.py             # YAML config loader
│   ├── orchestrator.py       # DAG runner
│   └── backends/
│       ├── __init__.py
│       ├── llm.py            # Provider-agnostic LLM (Gemini, OpenAI, Claude)
│       └── data.py           # Provider-agnostic data (CSV, BigQuery, GCS)
├── agents/
│   ├── __init__.py
│   ├── data_agent.py         # Load, profile, validate, split
│   ├── feature_agent.py      # LLM feature engineering + sklearn prep
│   ├── train_agent.py        # Model selection, training, MLflow
│   └── eval_agent.py         # Metrics, LLM report, recommendations
└── example_projects/
    └── customer_churn/
        └── config.yaml       # Sample project config
```

---

## LLM Backend Providers

| Provider | Config Value | SDK Required | Key Env Var |
|----------|-------------|--------------|-------------|
| Gemini | `gemini` | `google-genai` | `GEMINI_API_KEY` |
| OpenAI | `openai` | `openai` | `OPENAI_API_KEY` |
| Claude | `claude` | `anthropic` | `ANTHROPIC_API_KEY` |
| Local | `openai` | `openai` | Set `base_url` too |

---

## What's Next

- [ ] **Hyperparameter tuning** — Optuna integration via `hyperopt_trials` in config
- [ ] **Model deployment** — export pipeline as joblib/ONNX, create FastAPI endpoint
- [ ] **Data drift detection** — compare new data against training distribution
- [ ] **Web UI** — dashboard to track runs, compare experiments
- [ ] **Multi-modal** — handle text (embeddings) and image (feature extractors)
- [ ] **Feedback loop** — Eval recommendations auto-trigger retraining

---

## License

MIT — use it for anything.
