"""Data Agent — loads data, infers schema, validates, splits, profiles.

Connects to BigQuery / CSV / GCS via the DataBackend.
Optionally uses LLM to profile data and suggest anomalies.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.panel import Panel

from framework.agent_base import Agent, Context
from framework.artifact import DataArtifact

console = Console()


class DataAgent(Agent):
    """Loads data from the configured backend, validates, splits, profiles."""

    def __init__(self, context: Context):
        agent_cfg = context.config.get("agents", {}).get("data", {})
        super().__init__("data", agent_cfg, context)

    def run(self) -> DataArtifact:
        cfg = context = self.ctx.config
        ds_cfg = cfg.get("data_source", {})
        target_column = ds_cfg.get("target_column", "")

        if not target_column:
            raise ValueError("data_source.target_column not set in config")

        # 1. Load data from the configured backend
        console.print(f"  │  [dim]Loading from {ds_cfg.get('type', 'csv')}...[/]")
        df = self.ctx.data.load()
        console.print(f"  │  [dim]Loaded {len(df)} rows x {len(df.columns)} columns[/]")

        # 2. Validate target
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available: {list(df.columns)}"
            )

        y = df[target_column]
        X = df.drop(columns=[target_column])

        # 3. Handle missing values
        n_missing = X.isnull().sum().sum()
        if n_missing > 0:
            console.print(f"  │  [yellow]⚠ {n_missing} missing values — filling[/]")
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        X = X.ffill().bfill().fillna(0)

        # 4. Detect task type
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        nunique_target = y.nunique()
        if nunique_target <= 20 and y.dtype in ("object", "category", "bool", "int64"):
            task_type = "classification" if nunique_target / len(y) < 0.1 else "regression"
        else:
            task_type = "regression"

        # 5. LLM data analysis (optional)
        llm_insights = ""
        if self.agent_config.get("llm_analysis") and self.ctx.llm_available:
            llm_insights = self._llm_profile(X, y, target_column, task_type)
            if llm_insights:
                console.print(f"  │  [dim]LLM analysis: {llm_insights[:120]}...[/]")

        # 6. Split
        stratify = y if task_type == "classification" else None
        test_size = self.agent_config.get("test_size", 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify,
        )

        # 7. Summary
        summary = {
            "total_rows": len(df),
            "valid_rows": len(y),
            "numeric_features": len(numeric_cols),
            "categorical_features": len(cat_cols),
            "missing_values": int(n_missing),
            "target_distribution": y.value_counts(normalize=True).head(5).to_dict(),
            "llm_insights": llm_insights,
        }

        console.print(Panel(
            f"[bold]Data Profile (via {ds_cfg.get('type', 'csv')})[/]\n"
            f"  Rows: {summary['valid_rows']}\n"
            f"  Features: {summary['numeric_features']} num, {summary['categorical_features']} cat\n"
            f"  Task: {task_type}\n"
            f"  Train: {len(X_train)}  |  Test: {len(X_test)}",
            title="Data Agent",
            border_style="blue",
        ))

        return DataArtifact(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            feature_names=X.columns.tolist(), target_name=target_column,
            n_samples=len(y), n_features=X.shape[1],
            task_type=task_type, summary=summary,
        )

    def _llm_profile(self, X: pd.DataFrame, y: pd.Series,
                     target: str, task_type: str) -> str:
        schema_text = "\n".join(
            f"  {c}: {X[c].dtype}  unique={X[c].nunique()}  "
            f"null={X[c].isnull().sum()}  sample={X[c].dropna().head(2).tolist()}"
            for c in X.columns
        )
        target_info = (
            f"Target '{target}': {task_type}, values={y.nunique()}, "
            f"distribution={y.value_counts(normalize=True).head(5).to_dict()}"
        )
        prompt = (
            f"Analyze this {task_type} dataset ({len(X)} rows, {len(X.columns)} features).\n\n"
            f"Target: {target_info}\n\nSchema:\n{schema_text}\n\n"
            "Return JSON:\n"
            '{"observations": "brief data quality observations (2-3 sentences)", '
            '"warnings": ["list of potential issues"]}'
        )
        try:
            result = self.llm_chat_json(
                "You are a data profiling expert. Be concise.", prompt
            )
            obs = result.get("observations", "")
            warnings = result.get("warnings", [])
            combined = obs + (" | Warnings: " + "; ".join(warnings) if warnings else "")
            return combined
        except Exception as e:
            console.print(f"  │  [yellow]LLM data analysis failed: {e}[/]")
            return ""
