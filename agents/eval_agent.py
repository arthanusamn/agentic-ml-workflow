"""Eval Agent — comprehensive evaluation with LLM-powered report.

Computes all relevant metrics, extracts feature importance, generates
an executive summary and actionable recommendations via the configured LLM.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
)
from rich.console import Console
from rich.panel import Panel

from framework.agent_base import Agent, Context
from framework.artifact import DataArtifact, FeatureArtifact, ModelArtifact, EvalArtifact

console = Console()


class EvalAgent(Agent):
    """Evaluates the trained model and produces a report."""

    def __init__(self, context: Context):
        agent_cfg = context.config.get("agents", {}).get("eval", {})
        super().__init__("eval", agent_cfg, context)

    def run(self) -> EvalArtifact:
        model_artifact: ModelArtifact | None = self.ctx.get_artifact("train")
        if not model_artifact or not model_artifact.model:
            raise ValueError("No trained model found — did TrainAgent run?")

        feature = self.ctx.get_artifact("features")
        data = self.ctx.get_artifact("data")

        X_test = feature.X_test if feature else data.X_test
        y_test = feature.y_test if feature else data.y_test
        y_train = feature.y_train if feature else data.y_train
        task_type = data.task_type if data else "regression"
        fnames = (feature or data).feature_names if (feature or data) else []

        model = model_artifact.model
        y_pred = model.predict(X_test)

        # Metrics
        metrics = {}
        cm = None
        if task_type == "classification":
            try:
                metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
                metrics["precision"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                metrics["recall"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                cm = confusion_matrix(y_test, y_pred).tolist()
            except Exception as e:
                console.print(f"  │  [yellow]Classification metrics error: {e}[/]")
        else:
            try:
                metrics["r2_score"] = round(float(r2_score(y_test, y_pred)), 4)
                metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
                mask = y_test != 0
                if mask.any():
                    metrics["mape"] = round(float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100), 2)
            except Exception as e:
                console.print(f"  │  [yellow]Regression metrics error: {e}[/]")

        # Feature importance
        importance = {}
        if hasattr(model, "feature_importances_") and len(fnames) > 0:
            fi = model.feature_importances_
            for name, imp in zip(fnames[:20], fi[:20]):
                importance[name] = round(float(imp), 4)
            importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

        # LLM or rule-based report
        thresholds = self.agent_config.get("thresholds", {})
        report, recommendations, passes = self._generate_report(
            task_type, metrics, importance, model_artifact.model_name, thresholds,
        )

        self._print_eval(task_type, metrics, importance, passes)

        return EvalArtifact(
            metrics=metrics, confusion_matrix=cm,
            feature_importance=importance, report=report,
            recommendations=recommendations, passes_validation=passes,
        )

    def _generate_report(self, task_type: str, metrics: dict, importance: dict,
                         model_name: str, thresholds: dict):
        if self.agent_config.get("llm_report") and self.ctx.llm_available:
            try:
                result = self.llm_chat_json(
                    "You are an ML evaluation expert. Return only JSON.",
                    f"Review this {task_type} model ({model_name}).\n\n"
                    f"Task: {self.ctx.task_description}\n\n"
                    f"Metrics:\n{json.dumps(metrics, indent=2)}\n\n"
                    f"Top features:\n{json.dumps(list(importance.items())[:5], indent=2)}\n\n"
                    'Return JSON: {"report": "executive summary (2-3 sentences)", '
                    '"recommendations": ["rec1", "rec2", "rec3"]}',
                )
                report = result.get("report", "")
                recommendations = result.get("recommendations", [])
                passes = self._check_thresholds(task_type, metrics, thresholds)
                return report, recommendations, passes
            except Exception as e:
                console.print(f"  │  [yellow]LLM report failed: {e}[/]")

        # Fallback: rule-based
        report = self._basic_report(task_type, metrics)
        recommendations = self._basic_recs(task_type, metrics)
        passes = self._check_thresholds(task_type, metrics, thresholds)
        return report, recommendations, passes

    def _check_thresholds(self, task_type: str, metrics: dict, thresholds: dict) -> bool:
        if task_type == "classification":
            min_acc = thresholds.get("min_accuracy", 0.5)
            if metrics.get("accuracy", 1) < min_acc:
                return False
        else:
            min_r2 = thresholds.get("min_r2", 0.0)
            if metrics.get("r2_score", 1) < min_r2:
                return False
        return True

    def _basic_report(self, task_type: str, metrics: dict) -> str:
        if task_type == "classification":
            return (f"Accuracy: {metrics.get('accuracy', 'N/A')}, "
                    f"F1: {metrics.get('f1_score', 'N/A')}.")
        return (f"R²: {metrics.get('r2_score', 'N/A')}, "
                f"RMSE: {metrics.get('rmse', 'N/A')}.")

    def _basic_recs(self, task_type: str, metrics: dict) -> list[str]:
        if task_type == "classification":
            recs = []
            if metrics.get("accuracy", 0) < 0.7:
                recs.append("Try XGBoost or more feature engineering")
            if metrics.get("f1_score", 0) < metrics.get("accuracy", 0) - 0.1:
                recs.append("Check class imbalance — try SMOTE or class weights")
            return recs or ["Baseline model is reasonable"]
        recs = []
        if metrics.get("r2_score", 0) < 0.6:
            recs.append("Try polynomial features or gradient boosting")
        if metrics.get("mape", 100) > 20:
            recs.append("High relative error — consider log-transforming target")
        return recs or ["Baseline model is reasonable"]

    def _print_eval(self, task_type: str, metrics: dict, importance: dict, passes: bool):
        status = "[green]✔ PASS[/]" if passes else "[red]✘ FAIL[/]"
        lines = [f"[bold]Metrics ({task_type})[/]"]
        lines.extend(f"  {k}: {v}" for k, v in metrics.items())
        if importance:
            lines.append("\n[bold]Top Features[/]")
            lines.extend(f"  {k}: {v:.4f}" for k, v in list(importance.items())[:5])
        lines.append(f"\nStatus: {status}")
        console.print(Panel("\n".join(lines), title="Eval Agent",
                            border_style="green" if passes else "red"))
