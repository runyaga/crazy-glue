"""Context object shared by all handlers."""

from __future__ import annotations

import json
import typing
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

if typing.TYPE_CHECKING:
    from pydantic_ai.models import Model
    from soliplex.config import InstallationConfig


def _get_project_root() -> Path:
    """Get the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


@dataclass
class AnalysisContext:
    """Shared context for all handlers."""

    installation_config: InstallationConfig
    agent_config: typing.Any  # FactoryAgentConfig
    model: Model | None = None

    # Derived paths (set in __post_init__)
    project_root: Path = field(init=False)
    pending_tool_path: Path = field(init=False)
    managed_rooms_path: Path = field(init=False)
    prompts_path: Path = field(init=False)
    map_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = _get_project_root()
        db_path = self.project_root / "db"
        self.pending_tool_path = db_path / "pending_tool.json"
        self.managed_rooms_path = db_path / "managed_rooms.json"
        self.prompts_path = db_path / "prompts.json"

        # Map path from agent config or default
        extra = getattr(self.agent_config, "extra_config", {}) or {}
        map_rel = extra.get("map_path", "db/project_map.json")
        self.map_path = self.project_root / map_rel

    @property
    def roots(self) -> list[str]:
        """Get roots to scan for knowledge graph."""
        extra = getattr(self.agent_config, "extra_config", {}) or {}
        return extra.get("roots", ["src/crazy_glue", "rooms/"])

    @property
    def max_depth(self) -> int:
        """Get max depth for knowledge graph exploration."""
        extra = getattr(self.agent_config, "extra_config", {}) or {}
        return extra.get("max_depth", 5)

    @property
    def max_files(self) -> int:
        """Get max files for knowledge graph exploration."""
        extra = getattr(self.agent_config, "extra_config", {}) or {}
        return extra.get("max_files", 100)

    @property
    def model_name(self) -> str:
        """Model name from agent config."""
        extra = getattr(self.agent_config, "extra_config", {}) or {}
        return extra.get("model_name", "gpt-oss:latest")

    # --- Pending Tool Storage ---

    def load_pending_tool(self) -> dict | None:
        """Load pending tool from staging file."""
        if self.pending_tool_path.exists():
            try:
                return json.loads(self.pending_tool_path.read_text())
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def save_pending_tool(self, data: dict) -> None:
        """Save pending tool to staging file."""
        self.pending_tool_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_tool_path.write_text(json.dumps(data, indent=2))

    def clear_pending_tool(self) -> None:
        """Clear pending tool staging file."""
        if self.pending_tool_path.exists():
            self.pending_tool_path.unlink()

    # --- Prompts Storage ---

    def load_prompts(self) -> dict:
        """Load prompts from library file."""
        if self.prompts_path.exists():
            try:
                data = json.loads(self.prompts_path.read_text())
                return data.get("prompts", {})
            except (json.JSONDecodeError, KeyError):
                return {}
        return {}

    def save_prompts(self, prompts: dict) -> None:
        """Save prompts to library file."""
        self.prompts_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"prompts": prompts}
        self.prompts_path.write_text(json.dumps(data, indent=2))
