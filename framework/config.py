"""Config loader — reads a YAML project config and validates it.

Each ML project is defined by a single YAML file.
This loader resolves it into a typed dict the framework can use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# ---- Defaults ----

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "unnamed_project",
        "description": "",
    },
    "data_source": {
        "type": "csv",
        "path": "",
        "target_column": "",
    },
    "llm": {
        "provider": "",        # gemini | openai | claude | "" (disabled)
        "model": "",
        "api_key_env": "",
    },
    "agents": {
        "data": {
            "enabled": True,
            "test_size": 0.2,
            "llm_analysis": False,
        },
        "features": {
            "enabled": True,
            "llm_engineering": False,
            "max_transformations": 5,
        },
        "train": {
            "enabled": True,
            "algorithms": ["auto"],
            "hyperopt_trials": 0,
        },
        "eval": {
            "enabled": True,
            "llm_report": False,
            "thresholds": {
                "min_accuracy": 0.5,
                "min_r2": 0.0,
            },
        },
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML project config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    import yaml
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping (dict)")

    # Deep-merge with defaults
    config = _deep_merge(DEFAULT_CONFIG, raw)

    # Resolve API keys from env vars
    llm = config.get("llm", {})
    env_var = llm.get("api_key_env", "")
    if env_var:
        val = os.environ.get(env_var, "")
        if val:
            llm["api_key"] = val
        else:
            # Maybe the user put the key inline
            llm["api_key"] = llm.get("api_key", "")

    # Validate basics
    ds = config["data_source"]
    if ds["type"] not in ("csv", "bigquery", "gcs"):
        raise ValueError(f"Unsupported data source: {ds['type']}")

    if ds["type"] == "csv" and not ds.get("path"):
        raise ValueError("CSV data source requires a 'path' field")

    if not ds.get("target_column"):
        raise ValueError("data_source.target_column is required")

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. override wins."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def write_default_config(path: str | Path, name: str = "my_project",
                         description: str = "",
                         data_type: str = "csv"):
    """Write a starter config to disk."""
    path = Path(path)
    cfg = {
        "project": {
            "name": name,
            "description": description,
        },
        "data_source": {
            "type": data_type,
            "path": "data.csv",
            "target_column": "target",
        },
        "llm": {
            "provider": "",
            "model": "",
            "api_key_env": "",
        },
        "agents": {
            "data": {"enabled": True, "test_size": 0.2, "llm_analysis": False},
            "features": {"enabled": True, "llm_engineering": False, "max_transformations": 5},
            "train": {"enabled": True, "algorithms": ["auto"], "hyperopt_trials": 0},
            "eval": {"enabled": True, "llm_report": False,
                     "thresholds": {"min_accuracy": 0.5, "min_r2": 0.0}},
        },
    }
    import yaml
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path
