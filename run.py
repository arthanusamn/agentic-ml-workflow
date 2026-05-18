#!/usr/bin/env python3
"""🤖 Agentic ML Workflow — Project Runner

Usage:
    python run.py                          # Run with sample data
    python run.py --config my_config.yaml  # Run from a project config
    python run.py init --name fraud_detection  # Generate a new project template

Examples:
    # Simple: run with auto-generated config + sample data
    python run.py

    # Bring your own CSV
    python run.py --config my_churn.yaml

    # With Gemini
    python run.py --config my_config.yaml --llm-provider gemini --api-key-env GEMINI_API_KEY

    # Create a new project
    python run.py init --name house_prices --dir ~/projects/
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

from framework.config import load_config, write_default_config
from framework.backends.llm import LLMBackend
from framework.backends.data import DataBackend
from framework.agent_base import Context
from framework.orchestrator import MasterOrchestrator
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.train_agent import TrainAgent
from agents.eval_agent import EvalAgent

console = Console()


def _generate_sample_config() -> tuple[str, str]:
    """Generate sample churn data + config, returns (config_path, data_path)."""
    import pandas as pd
    import numpy as np
    import tempfile

    np.random.seed(42)
    n = 2000

    df = pd.DataFrame({
        "age": np.random.randint(18, 75, n),
        "income": np.random.lognormal(mean=10.5, sigma=0.6, size=n),
        "education_years": np.random.randint(8, 22, n),
        "num_accounts": np.random.randint(1, 10, n),
        "tenure_months": np.random.randint(1, 120, n),
        "region": np.random.choice(["north", "south", "east", "west"], n),
        "occupation": np.random.choice(["tech", "finance", "healthcare", "education", "other"], n),
        "credit_score": np.random.randint(300, 850, n),
        "avg_transaction": np.random.exponential(scale=100, size=n),
        "support_calls": np.random.poisson(lam=2, size=n),
        "late_payments": np.random.poisson(lam=0.5, size=n),
    })

    logit = (
        -2.0
        + 0.03 * (df["age"] - 40) / 15
        + 0.15 * (df["income"] - df["income"].mean()) / df["income"].std()
        + 0.10 * (df["education_years"] - 15) / 3
        - 0.20 * (df["num_accounts"] - 5) / 2
        + 0.02 * (df["tenure_months"] - 60) / 30
        - 0.25 * (df["credit_score"] - 600) / 150
        + (df["region"] == "north") * 0.3
        - (df["region"] == "south") * 0.2
        + (df["occupation"] == "tech") * 0.4
        - 0.30 * (df["avg_transaction"] - df["avg_transaction"].mean()) / df["avg_transaction"].std()
        + 0.20 * df["support_calls"]
        + 0.40 * df["late_payments"]
        + np.random.normal(0, 2.0, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["churned"] = (np.random.random(n) < prob).astype(int)

    # Write data to temp
    data_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(data_tmp.name, index=False)

    # Write config next to it
    cfg_path = Path(tempfile.gettempdir()) / "sample_churn_config.yaml"
    cfg = {
        "project": {
            "name": "customer_churn",
            "description": "Predict customer churn from demographic and behavioral data",
        },
        "data_source": {
            "type": "csv",
            "path": data_tmp.name,
            "target_column": "churned",
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
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return str(cfg_path), data_tmp.name


def cmd_run(args):
    """Run a pipeline from a config file (or sample data)."""
    config_path = args.config

    if not config_path:
        console.print("[yellow]No --config provided. Generating sample project...[/]")
        config_path, data_path = _generate_sample_config()
        console.print(f"  [dim]Config: {config_path}[/]")
        console.print(f"  [dim]Data: {data_path}[/]")

    # Load config
    config = load_config(config_path)

    # Override LLM from CLI if provided
    if args.llm_provider:
        config["llm"]["provider"] = args.llm_provider
    if args.api_key_env:
        config["llm"]["api_key_env"] = args.api_key_env
        config["llm"]["api_key"] = os.environ.get(args.api_key_env, "")
    if args.model:
        config["llm"]["model"] = args.model

    # Init backends
    llm = LLMBackend(config.get("llm", {}))
    data = DataBackend(config.get("data_source", {}))
    ctx = Context(config, llm, data)

    if llm.available:
        console.print(f"  [dim]LLM: {config['llm']['provider']} ({config['llm']['model']})[/]")

    # Build pipeline from config
    orchestrator = MasterOrchestrator(ctx)
    agent_configs = config.get("agents", {})

    if agent_configs.get("data", {}).get("enabled", True):
        orchestrator.add_stage(DataAgent(ctx), description="Load, profile & split data")

    if agent_configs.get("features", {}).get("enabled", True):
        orchestrator.add_stage(FeatureAgent(ctx), depends_on=["data"],
                                description="Feature engineering & preprocessing")

    if agent_configs.get("train", {}).get("enabled", True):
        orchestrator.add_stage(TrainAgent(ctx), depends_on=["features"],
                                description="Model training & selection")

    if agent_configs.get("eval", {}).get("enabled", True):
        orchestrator.add_stage(EvalAgent(ctx), depends_on=["train"],
                                description="Evaluation & recommendations")

    # Run
    results = orchestrator.run()

    # Print final recommendations
    eval_result = orchestrator.get_artifact("eval")
    if eval_result and hasattr(eval_result, "recommendations"):
        recs = eval_result.recommendations
        if recs:
            console.print("\n[bold]📋 Recommendations:[/]")
            for i, rec in enumerate(recs, 1):
                console.print(f"  {i}. {rec}")

    console.print()


def cmd_init(args):
    """Generate a new project template."""
    name = args.name or "my_project"
    dest = Path(args.dir) / name if args.dir else Path.cwd() / name
    dest.mkdir(parents=True, exist_ok=True)

    # Write config
    cfg_path = dest / "config.yaml"
    write_default_config(cfg_path, name=name, description=args.desc or f"{name} ML project")
    console.print(f"[green]✔ Created project template: {dest}[/]")
    console.print(f"[green]  Config: {cfg_path}[/]")
    console.print()
    console.print("Next:")
    console.print(f"  1. Edit {cfg_path} with your data source and target")
    console.print("  2. Run: python run.py --config " + str(cfg_path))


def main():
    parser = argparse.ArgumentParser(
        description="🤖 Agentic ML Workflow — Multi-agent ML pipeline framework"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_p = sub.add_parser("run", help="Run a pipeline from config")
    run_p.add_argument("--config", help="Path to project YAML config")
    run_p.add_argument("--llm-provider", help="LLM provider (gemini, openai, claude)")
    run_p.add_argument("--api-key-env", help="Environment variable with API key")
    run_p.add_argument("--model", help="Model name override")
    run_p.set_defaults(func=cmd_run)

    # init
    init_p = sub.add_parser("init", help="Generate a new project template")
    init_p.add_argument("--name", default="my_project", help="Project name")
    init_p.add_argument("--dir", default=".", help="Parent directory")
    init_p.add_argument("--desc", default="", help="Project description")
    init_p.add_argument("--template", default="classification", help="Template type")
    init_p.set_defaults(func=cmd_init)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    elif hasattr(args, 'config'):
        args.func(args)
    else:
        # Default: run with sample data — set up a fake args for cmd_run
        class FakeArgs:
            config = None
            llm_provider = None
            api_key_env = None
            model = None
        cmd_run(FakeArgs())


if __name__ == "__main__":
    main()
