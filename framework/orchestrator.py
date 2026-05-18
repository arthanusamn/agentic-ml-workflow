"""Master Orchestrator — DAG runner with typed artifact passing.

Now works with the Agent class hierarchy: each stage is an Agent instance
whose run() method returns an artifact.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from framework.agent_base import Agent, Context
from framework import __version__

console = Console()


@dataclass
class StageResult:
    stage_name: str
    success: bool
    duration_s: float
    artifact: Any = None
    error: str | None = None


class MasterOrchestrator:
    """Orchestrates a DAG of Agent instances."""

    def __init__(self, context: Context):
        self.ctx = context
        self._stages: list[dict] = []
        self.results: dict[str, StageResult] = {}

    def add_stage(self, agent: Agent, depends_on: list[str] | None = None,
                  description: str = "") -> "MasterOrchestrator":
        self._stages.append({
            "name": agent.name,
            "agent": agent,
            "depends_on": depends_on or [],
            "description": description or agent.__class__.__name__,
        })
        return self

    def run(self) -> dict[str, StageResult]:
        self._print_header()
        order = self._resolve_order()

        for stage in order:
            name = stage["name"]
            agent = stage["agent"]
            self._print_stage_start(name, stage["description"])

            start = time.time()
            try:
                artifact = agent.run()
                elapsed = time.time() - start
                valid = agent.validate(artifact)
                self.results[name] = StageResult(
                    stage_name=name, success=valid, duration_s=elapsed,
                    artifact=artifact,
                )
                if valid:
                    self.ctx.set_artifact(name, artifact)
                    self._print_stage_ok(name, elapsed)
                else:
                    self.results[name].error = "Validation failed"
                    self._print_stage_fail(name, "Validation failed", elapsed)
            except Exception as e:
                elapsed = time.time() - start
                self.results[name] = StageResult(
                    stage_name=name, success=False, duration_s=elapsed, error=str(e),
                )
                self._print_stage_fail(name, str(e), elapsed)

        self._print_summary()
        return self.results

    def get_artifact(self, stage_name: str, artifact_type: type = object):
        result = self.results.get(stage_name)
        if result and result.success and isinstance(result.artifact, artifact_type):
            return result.artifact
        return None

    def _resolve_order(self) -> list[dict]:
        ordered = []
        visited = set()

        def visit(stage: dict):
            if stage["name"] in visited:
                return
            for dep_name in stage["depends_on"]:
                dep = next(s for s in self._stages if s["name"] == dep_name)
                visit(dep)
            visited.add(stage["name"])
            ordered.append(stage)

        for stage in self._stages:
            visit(stage)
        return ordered

    def _print_header(self):
        console.print()
        console.print(Panel(
            f"[bold cyan]🤖 Agentic ML — v{__version__}[/]\n"
            f"[white]{self.ctx.task_description}[/]",
            box=box.ROUNDED,
        ))
        console.print()

    def _print_stage_start(self, name: str, desc: str):
        indent = "  │  " if self.results else "  ├──"
        console.print(f"{indent}[yellow]▶ {name}[/] — {desc}")

    def _print_stage_ok(self, name: str, elapsed: float):
        console.print(f"  │  [green]✔ {name} done[/]  [dim]({elapsed:.2f}s)[/]")
        console.print()

    def _print_stage_fail(self, name: str, error: str, elapsed: float):
        console.print(f"  │  [red]✘ {name} FAILED[/]  [dim]({elapsed:.2f}s)[/]")
        console.print(f"  │  [red]  {error}[/]")
        console.print()

    def _print_summary(self):
        console.print("  └──" + "─" * 40)
        table = Table(title="Pipeline Summary", box=box.SIMPLE)
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Time", justify="right")
        table.add_column("Artifact")
        for r in self.results.values():
            status = "[green]✔[/]" if r.success else "[red]✘[/]"
            artifact_name = type(r.artifact).__name__ if r.artifact and r.success else "—"
            table.add_row(r.stage_name, status, f"{r.duration_s:.2f}s", artifact_name)
        console.print(table)
        console.print()
