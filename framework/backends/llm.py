"""LLM backend — provider-agnostic wrapper.

Supports Gemini, OpenAI, Claude, and local LLMs with a unified interface.
Each agent creates its own LLMBackend from the project config.
"""
from __future__ import annotations

import json
from typing import Any, Optional


class LLMBackend:
    """Unified LLM interface. Configure once per pipeline run."""

    def __init__(self, config: dict[str, Any]):
        self.provider = (config.get("provider") or "").lower()
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self._available = bool(self.provider and self.api_key)

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, system: str, user: str, response_format: Optional[dict] = None) -> str:
        """Send a chat message. Returns the response text."""
        if not self._available:
            raise RuntimeError("LLM not configured — set provider and api_key in config")

        if self.provider == "gemini":
            return self._gemini_chat(system, user, response_format)
        elif self.provider == "openai":
            return self._openai_chat(system, user, response_format)
        elif self.provider == "claude":
            return self._claude_chat(system, user, response_format)
        else:
            # Fallback: try OpenAI-compatible (works with local LLMs too)
            return self._openai_chat(system, user, response_format)

    def chat_json(self, system: str, user: str) -> dict:
        """Chat and parse response as JSON."""
        raw = self.chat(system, user, response_format={"type": "json_object"})
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def _openai_chat(self, system: str, user: str,
                     response_format: Optional[dict] = None) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        kwargs = {
            "model": self.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _gemini_chat(self, system: str, user: str,
                     response_format: Optional[dict] = None) -> str:
        import google.genai as genai
        client = genai.Client(api_key=self.api_key)
        model = self.model or "gemini-1.5-flash"
        contents = f"{system}\n\n{user}" if system else user
        config_args = {"temperature": 0.3}
        if response_format:
            config_args["response_mime_type"] = "application/json"
        resp = client.models.generate_content(
            model=model, contents=contents,
            config=type("cfg", (), config_args)(),
        )
        return resp.text or ""

    def _claude_chat(self, system: str, user: str,
                     response_format: Optional[dict] = None) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        model = self.model or "claude-sonnet-4-20250514"
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": 0.3,
        }
        if response_format:
            # Claude doesn't support response_format natively; just request JSON
            kwargs["messages"][0]["content"] = (
                "Respond in valid JSON only, no markdown.\n\n" + user
            )
        resp = client.messages.create(**kwargs)
        return resp.content[0].text if resp.content else ""
