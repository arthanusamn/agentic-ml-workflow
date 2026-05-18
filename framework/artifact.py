"""Typed artifacts passed between pipeline stages. Shared across all projects."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class DataArtifact:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    target_name: str
    n_samples: int
    n_features: int
    task_type: str  # classification | regression
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureArtifact:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: Any = None
    feature_names: list[str] = field(default_factory=list)
    transformations_applied: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ModelArtifact:
    model: Any = None
    model_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    features_used: list[str] = field(default_factory=list)
    mlflow_run_id: str = ""
    training_time_s: float = 0.0


@dataclass
class EvalArtifact:
    metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    report: str = ""
    recommendations: list[str] = field(default_factory=list)
    passes_validation: bool = False
