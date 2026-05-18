"""Train Agent — model selection, training, MLflow tracking.

Trains multiple model candidates, picks the best one by training score.
Supports XGBoost and LightGBM if installed.
Logs everything to MLflow if available.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from rich.console import Console
from rich.panel import Panel

from framework.agent_base import Agent, Context
from framework.artifact import FeatureArtifact, DataArtifact, ModelArtifact

console = Console()
warnings.filterwarnings("ignore")


class TrainAgent(Agent):
    """Selects and trains the best model based on data characteristics."""

    def __init__(self, context: Context):
        agent_cfg = context.config.get("agents", {}).get("train", {})
        super().__init__("train", agent_cfg, context)

    def run(self) -> ModelArtifact:
        feature_artifact = self.ctx.get_artifact("features")
        data_artifact = self.ctx.get_artifact("data")

        if feature_artifact:
            X_train, X_test = feature_artifact.X_train, feature_artifact.X_test
            y_train, y_test = feature_artifact.y_train, feature_artifact.y_test
            task_type = data_artifact.task_type if data_artifact else "regression"
            target_name = data_artifact.target_name if data_artifact else "target"
            feature_names = feature_artifact.feature_names
        elif data_artifact:
            X_train, X_test = data_artifact.X_train, data_artifact.X_test
            y_train, y_test = data_artifact.y_train, data_artifact.y_test
            task_type = data_artifact.task_type
            target_name = data_artifact.target_name
            feature_names = data_artifact.feature_names
        else:
            raise ValueError("No data available")

        # Determine algorithm candidates
        candidates = self._get_candidates(task_type)

        # Train each candidate, pick best by training score
        start = time.time()
        best_model, best_score, best_params = None, float("-inf"), {}
        for name, model_fn in candidates:
            try:
                m = model_fn()
                m.fit(X_train, y_train)
                score = m.score(X_train, y_train)
                if score > best_score:
                    best_model = m
                    best_score = score
                    best_params = m.get_params()
            except Exception as e:
                console.print(f"  │  [dim]  {name}: {e}[/]")

        if best_model is None:
            raise RuntimeError("No model could be trained")

        # Test score
        y_pred = best_model.predict(X_test)
        if task_type == "classification":
            test_score = accuracy_score(y_test, y_pred)
            metric_name = "Accuracy"
        else:
            test_score = r2_score(y_test, y_pred)
            metric_name = "R²"

        elapsed = time.time() - start

        # MLflow
        mlflow_run_id = self._log_mlflow(best_model, best_params, feature_names,
                                         metric_name, test_score)

    

        console.print(Panel(
            f"[bold]{best_model.__class__.__name__}[/]\n"
            f"  {metric_name}: {test_score:.4f}\n"
            f"  Train time: {elapsed:.2f}s\n"
            f"  MLflow: {mlflow_run_id or '—'}",
            title="Train Agent",
            border_style="green",
        ))

        return ModelArtifact(
            model=best_model, model_name=best_model.__class__.__name__,
            params=best_params, features_used=feature_names,
            mlflow_run_id=mlflow_run_id, training_time_s=elapsed,
        )

    def _get_candidates(self, task_type: str):
        """Build list of (name, factory_fn) candidates."""
        algorithms = self.agent_config.get("algorithms", ["auto"])

        if algorithms == ["auto"]:
            # Auto-select based on task
            if task_type == "classification":
                base = [
                    ("LogisticRegression", lambda: LogisticRegression(max_iter=1000, random_state=42)),
                    ("RandomForest", lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ]
            else:
                base = [
                    ("Ridge", lambda: Ridge(alpha=1.0, random_state=42)),
                    ("RandomForest", lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ]

            # Add XGBoost / LightGBM if installed
            try:
                import xgboost
                if task_type == "classification":
                    base.append(("XGBoost", lambda: xgboost.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)))
                else:
                    base.append(("XGBoost", lambda: xgboost.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)))
            except ImportError:
                pass
            try:
                import lightgbm
                if task_type == "classification":
                    base.append(("LightGBM", lambda: lightgbm.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)))
                else:
                    base.append(("LightGBM", lambda: lightgbm.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)))
            except ImportError:
                pass

            return base

        # Manual algorithm selection from config
        algo_map = {
            "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": lambda: (RandomForestClassifier if task_type == "classification" else RandomForestRegressor)(
                n_estimators=100, random_state=42, n_jobs=-1),
            "ridge": lambda: Ridge(alpha=1.0, random_state=42),
        }
        try:
            import xgboost
            algo_map["xgboost"] = lambda: (xgboost.XGBClassifier if task_type == "classification" else xgboost.XGBRegressor)(
                n_estimators=100, random_state=42, verbosity=0)
        except ImportError:
            pass
        try:
            import lightgbm
            algo_map["lightgbm"] = lambda: (lightgbm.LGBMClassifier if task_type == "classification" else lightgbm.LGBMRegressor)(
                n_estimators=100, random_state=42, verbose=-1)
        except ImportError:
            pass

        return [(a, algo_map[a]) for a in algorithms if a in algo_map]

    def _log_mlflow(self, model, params, features, metric_name, metric_value) -> str:
        try:
            import mlflow
            mlflow.set_experiment("agentic-ml-workflow")
            with mlflow.start_run() as run:
                mlflow.log_params(params)
                mlflow.log_metric(metric_name.lower(), metric_value)
                mlflow.sklearn.log_model(model, "model")
                if hasattr(model, "feature_importances_"):
                    for name, imp in zip(features[:50], model.feature_importances_[:50]):
                        mlflow.log_metric(f"imp_{name}", float(imp))
                return run.info.run_id
        except Exception:
            return ""
