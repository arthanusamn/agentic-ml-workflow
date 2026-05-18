"""Agent base class — all agents inherit from this.

An Agent has:
- A name and config (from project YAML)
- Access to the LLM backend (if configured)
- A run() method that returns a typed Artifact
- A validate() method for self-checking
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from framework.backends.llm import LLMBackend
from framework.backends.data import DataBackend


class Context:
    """Shared runtime context passed between agents.

    The orchestrator builds this and injects the output of each agent
    as attributes. Agents read from context to get upstream artifacts.
    """

    def __init__(self, config: dict, llm: LLMBackend, data: DataBackend):
        self.config = config
        self.llm = llm
        self.data = data
        # Upstream artifacts are set by the orchestrator after each stage
        self._artifacts: dict[str, Any] = {}
        # Also flatten into attributes for convenience
        self.task_description = config.get("project", {}).get("description", "")

    def set_artifact(self, stage_name: str, artifact: Any):
        self._artifacts[stage_name] = artifact

    def get_artifact(self, stage_name: str):
        return self._artifacts.get(stage_name)

    @property
    def llm_available(self) -> bool:
        return self.llm.available


class Agent(ABC):
    """Base class for all pipeline agents."""

    def __init__(self, name: str, agent_config: dict[str, Any], context: Context):
        self.name = name
        self.agent_config = agent_config
        self.ctx = context

    @abstractmethod
    def run(self) -> Any:
        """Execute this agent's work. Returns an artifact."""
        ...

    def validate(self, artifact: Any) -> bool:
        """Post-run validation. Override in subclasses."""
        return artifact is not None

    def llm_chat(self, system: str, user: str) -> str:
        """Convenience: send a chat to the configured LLM."""
        return self.ctx.llm.chat(system, user)

    def llm_chat_json(self, system: str, user: str) -> dict:
        """Convenience: get JSON response from LLM."""
        return self.ctx.llm.chat_json(system, user)
