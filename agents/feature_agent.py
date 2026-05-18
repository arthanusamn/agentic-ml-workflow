"""Feature Agent — LLM-guided feature engineering + sklearn preprocessing.

Uses the configured LLM provider (Gemini, OpenAI, Claude) to analyze
the dataset and write custom feature engineering code. Falls back to
deterministic sklearn preprocessing if no LLM is configured.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from rich.console import Console

from framework.agent_base import Agent, Context
from framework.artifact import DataArtifact, FeatureArtifact

console = Console()


class FeatureAgent(Agent):
    """Engineers features: LLM-guided transforms + sklearn preprocessing."""

    def __init__(self, context: Context):
        agent_cfg = context.config.get("agents", {}).get("features", {})
        super().__init__("features", agent_cfg, context)

    def run(self) -> FeatureArtifact:
        # Get data from upstream
        data_artifact: DataArtifact | None = self.ctx.get_artifact("data")
        if data_artifact is None:
            raise ValueError("No DataArtifact found — did DataAgent run?")

        X_train = data_artifact.X_train.copy()
        X_test = data_artifact.X_test.copy()
        y_train = data_artifact.y_train

        console.print(f"  │  [dim]Features: {data_artifact.n_features} raw inputs[/]")

        transformations = []
        llm_applied = False

        # LLM-guided feature engineering (optional)
        if self.agent_config.get("llm_engineering") and self.ctx.llm_available:
            llm_code = self._llm_generate_features(data_artifact)
            if llm_code:
                llm_applied, X_train, X_test, descs = self._safe_exec(llm_code, X_train, X_test, y_train)
                if llm_applied:
                    transformations.extend(descs)

        # Deterministic sklearn preprocessing (always runs)
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SkPipe

        numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        pipe_num = SkPipe([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        pipe_cat = SkPipe([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        xformers = []
        if numeric_cols:
            xformers.append(("num", pipe_num, numeric_cols))
        if cat_cols:
            xformers.append(("cat", pipe_cat, cat_cols))

        preprocessor = ColumnTransformer(xformers, remainder="drop")
        preprocessor.fit(X_train)

        X_tr = preprocessor.transform(X_train)
        X_te = preprocessor.transform(X_test)

        # Build names
        fnames = list(numeric_cols)
        if cat_cols:
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            fnames.extend(encoder.get_feature_names_out(cat_cols).tolist())

        X_train_final = pd.DataFrame(X_tr, columns=fnames)
        X_test_final = pd.DataFrame(X_te, columns=fnames)

        console.print(f"  │  [green]→ {len(fnames)} features after prep[/]")
        for t in transformations:
            console.print(f"  │    [dim]• {t}[/]")

        return FeatureArtifact(
            X_train=X_train_final, X_test=X_test_final,
            y_train=data_artifact.y_train, y_test=data_artifact.y_test,
            pipeline=preprocessor, feature_names=fnames,
            transformations_applied=transformations,
            notes=f"LLM: {'yes' if llm_applied else 'no'}",
        )

    def _llm_generate_features(self, artifact: DataArtifact) -> str | None:
        """Ask LLM to propose feature engineering code."""
        schema_lines = []
        X = artifact.X_train
        for col in X.columns:
            dtype = X[col].dtype
            nunique = X[col].nunique()
            sample = X[col].dropna().head(3).tolist()
            schema_lines.append(f"  {col} ({dtype}) unique={nunique} sample={sample}")
        schema_text = "\n".join(schema_lines)

        max_trans = self.agent_config.get("max_transformations", 5)
        prompt = (
            f"Dataset: {artifact.task_type} with target '{artifact.target_name}'\n\n"
            f"Schema:\n{schema_text}\n\n"
            f"Write a Python function `engineer_features(X_train, X_test, y_train)` "
            f"that adds up to {max_trans} new feature columns using pandas and numpy only.\n"
            "Rules:\n"
            "- Return (X_train, X_test, list_of_descriptions)\n"
            "- Do NOT import anything beyond pandas, numpy\n"
            "- Add NEW columns, don't drop existing ones\n"
            "- Keep it simple — ratios, interactions, log transforms, binning\n"
            "- No filesystem, network, or subprocess access\n\n"
            'Return JSON: {"code": "def engineer_features(...):\\n    ...\\n    return ..."}'
        )
        try:
            result = self.llm_chat_json(
                "You are a feature engineering expert. Return only valid JSON.", prompt,
            )
            return result.get("code", "")
        except Exception as e:
            console.print(f"  │  [yellow]LLM feature gen failed: {e}[/]")
            return None

    def _safe_exec(self, code: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series):
        """Execute LLM-generated code in restricted scope."""
        match = re.search(r"def engineer_features\(.*?\):", code)
        if not match:
            return False, X_train, X_test, []
        scope: dict[str, Any] = {"pd": pd, "np": __import__("numpy")}
        try:
            exec(code, scope)
            fn = scope.get("engineer_features")
            if fn is None:
                return False, X_train, X_test, []
            nx_tr, nx_te, descs = fn(X_train, X_test, y_train)
            return True, nx_tr, nx_te, descs
        except Exception as e:
            console.print(f"  │  [yellow]LLM code failed at runtime: {e}[/]")
            return False, X_train, X_test, []
