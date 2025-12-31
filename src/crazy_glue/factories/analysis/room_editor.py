"""
Room Configuration Editor - Read, modify, and write room_config.yaml files.

Provides a clean interface for managing room configurations with safety checks.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


# Custom representer to use block style for multi-line strings
def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
    """Use block style (|) for multi-line strings, plain style otherwise."""
    tag = "tag:yaml.org,2002:str"
    if "\n" in data:
        return dumper.represent_scalar(tag, data, style="|")
    return dumper.represent_scalar(tag, data)


# Create a custom dumper that uses block style for multi-line strings
class _BlockStyleDumper(yaml.SafeDumper):
    """YAML dumper that uses block style for multi-line strings."""

    pass


_BlockStyleDumper.add_representer(str, _str_representer)

# Tracking file for managed rooms (stored separately since RoomConfig
# doesn't support arbitrary fields)
MANAGED_ROOMS_FILE = "db/managed_rooms.json"


def _get_project_root() -> Path:
    """Get the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


class RoomConfigEditor:
    """Read, modify, and write room_config.yaml files."""

    def __init__(self, room_path: Path):
        """Initialize editor for a room directory.

        Args:
            room_path: Path to the room directory (containing room_config.yaml)
        """
        self.room_path = room_path
        self.config_path = room_path / "room_config.yaml"
        self._data: dict | None = None
        self._original: str | None = None

    def load(self) -> dict:
        """Load and parse room config."""
        if self._data is None:
            if not self.config_path.exists():
                msg = f"Config not found: {self.config_path}"
                raise FileNotFoundError(msg)
            self._original = self.config_path.read_text()
            self._data = yaml.safe_load(self._original)
        return self._data

    def save(self) -> None:
        """Write config back to YAML."""
        if self._data is None:
            raise ValueError("No config loaded")
        content = yaml.dump(
            self._data,
            Dumper=_BlockStyleDumper,
            default_flow_style=False,
            sort_keys=False,
            width=80,
            allow_unicode=True,
        )
        self.config_path.write_text(content)

    def validate(self) -> list[str]:
        """Run soliplex check-config and return errors."""
        project_root = _get_project_root()
        installation_yaml = project_root / "installation.yaml"

        if not installation_yaml.exists():
            return []

        try:
            result = subprocess.run(
                ["soliplex-cli", "check-config", str(installation_yaml)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root),
            )

            errors = []
            if result.returncode != 0:
                for line in result.stdout.split("\n"):
                    line_lower = line.lower()
                    is_val_err = "validation error" in line_lower
                    is_err = "error" in line_lower and ":" in line
                    if is_val_err or is_err:
                        errors.append(line.strip())

                if not errors and result.stderr:
                    snippet = result.stderr[:200]
                    errors.append(f"Config validation failed: {snippet}")

            return errors
        except FileNotFoundError:
            return ["soliplex-cli not found in PATH"]
        except subprocess.TimeoutExpired:
            return ["Config validation timed out"]
        except Exception as e:
            return [f"Config validation error: {e}"]

    # ----- Managed Room Tracking -----

    @staticmethod
    def _get_managed_rooms_path() -> Path:
        """Get path to managed rooms tracking file."""
        return _get_project_root() / MANAGED_ROOMS_FILE

    @staticmethod
    def _load_managed_rooms() -> set[str]:
        """Load set of managed room IDs from tracking file."""
        path = RoomConfigEditor._get_managed_rooms_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return set(data.get("managed_rooms", []))
            except (json.JSONDecodeError, KeyError):
                return set()
        return set()

    @staticmethod
    def _save_managed_rooms(room_ids: set[str]) -> None:
        """Save set of managed room IDs to tracking file."""
        path = RoomConfigEditor._get_managed_rooms_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"managed_rooms": sorted(room_ids)}
        path.write_text(json.dumps(data, indent=2))

    def is_managed(self) -> bool:
        """Check if this room is managed by the architect."""
        room_id = self.get_id()
        managed = self._load_managed_rooms()
        return room_id in managed

    def mark_as_managed(self) -> None:
        """Mark this room as managed by the architect."""
        room_id = self.get_id()
        managed = self._load_managed_rooms()
        managed.add(room_id)
        self._save_managed_rooms(managed)

    def unmark_as_managed(self) -> None:
        """Remove this room from managed tracking."""
        room_id = self.get_id()
        managed = self._load_managed_rooms()
        managed.discard(room_id)
        self._save_managed_rooms(managed)

    # ----- Room Metadata -----

    def get_id(self) -> str:
        """Get room ID."""
        return self.load().get("id", "")

    def get_name(self) -> str:
        """Get room name."""
        return self.load().get("name", "")

    def set_description(self, text: str) -> None:
        """Update room description."""
        self.load()["description"] = text

    def get_description(self) -> str:
        """Get room description."""
        return self.load().get("description", "")

    def set_welcome_message(self, text: str) -> None:
        """Update welcome message."""
        self.load()["welcome_message"] = text

    def get_welcome_message(self) -> str:
        """Get welcome message."""
        return self.load().get("welcome_message", "")

    def add_suggestion(self, text: str) -> None:
        """Add a suggestion to the room."""
        data = self.load()
        if "suggestions" not in data:
            data["suggestions"] = []
        data["suggestions"].append(text)

    def remove_suggestion(self, index: int) -> bool:
        """Remove suggestion by index. Returns True if removed."""
        data = self.load()
        suggestions = data.get("suggestions", [])
        if 0 <= index < len(suggestions):
            suggestions.pop(index)
            return True
        return False

    def get_suggestions(self) -> list[str]:
        """Get all suggestions."""
        return self.load().get("suggestions", [])

    # ----- Agent Config -----

    def _get_agent(self) -> dict:
        """Get or create agent config dict."""
        data = self.load()
        if "agent" not in data:
            data["agent"] = {}
        return data["agent"]

    def get_agent_kind(self) -> str:
        """Get agent kind (default or factory)."""
        return self._get_agent().get("kind", "default")

    def set_agent_kind(self, kind: str) -> None:
        """Set agent kind."""
        self._get_agent()["kind"] = kind

    def set_system_prompt(self, text: str) -> None:
        """Update system prompt (for default agents)."""
        self._get_agent()["system_prompt"] = text

    def get_system_prompt(self) -> str | None:
        """Get system prompt."""
        return self._get_agent().get("system_prompt")

    def set_model_name(self, name: str) -> None:
        """Change LLM model (for default agents)."""
        self._get_agent()["model_name"] = name

    def get_model_name(self) -> str | None:
        """Get model name."""
        return self._get_agent().get("model_name")

    # ----- Tools -----
    # Note: soliplex expects tools as a LIST of dicts with tool_name key:
    # tools:
    #   - tool_name: "soliplex.tools.search_documents"
    #     rag_lancedb_stem: "my_db"

    def _get_tools(self) -> list:
        """Get or create tools list."""
        data = self.load()
        if "tools" not in data:
            data["tools"] = []
        # Handle legacy dict format by converting
        if isinstance(data["tools"], dict):
            data["tools"] = []
        return data["tools"]

    def add_tool(
        self,
        tool_name: str,
        **tool_config: Any,
    ) -> None:
        """Add a tool to the room.

        Args:
            tool_name: The tool name (e.g. soliplex.tools.search_documents)
            **tool_config: Additional tool configuration (allow_mcp, etc.)
        """
        tools = self._get_tools()
        # Check if already exists
        for t in tools:
            if t.get("tool_name") == tool_name:
                return  # Already exists
        tool_entry = {"tool_name": tool_name}
        tool_entry.update(tool_config)
        tools.append(tool_entry)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool. Returns True if removed."""
        tools = self._get_tools()
        for i, t in enumerate(tools):
            if t.get("tool_name") == tool_name:
                tools.pop(i)
                return True
        return False

    def configure_tool(self, tool_name: str, **config: Any) -> None:
        """Update tool configuration."""
        tools = self._get_tools()
        for t in tools:
            if t.get("tool_name") == tool_name:
                t.update(config)
                return
        # Not found, add it
        self.add_tool(tool_name, **config)

    def list_tools(self) -> list[str]:
        """List configured tool names."""
        tools = self._get_tools()
        return [t.get("tool_name", "") for t in tools if t.get("tool_name")]

    # ----- MCP Client Toolsets -----

    def _get_mcp_toolsets(self) -> dict:
        """Get or create mcp_client_toolsets dict."""
        data = self.load()
        if "mcp_client_toolsets" not in data:
            data["mcp_client_toolsets"] = {}
        return data["mcp_client_toolsets"]

    def add_mcp_http(
        self,
        name: str,
        url: str,
        headers: dict | None = None,
        query_params: dict | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        """Add an HTTP MCP client toolset."""
        toolsets = self._get_mcp_toolsets()
        config: dict[str, Any] = {"kind": "http", "url": url}
        if headers:
            config["headers"] = headers
        if query_params:
            config["query_params"] = query_params
        if allowed_tools:
            config["allowed_tools"] = allowed_tools
        toolsets[name] = config

    def add_mcp_stdio(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        """Add a stdio MCP client toolset."""
        toolsets = self._get_mcp_toolsets()
        config: dict[str, Any] = {"kind": "stdio", "command": command}
        if args:
            config["args"] = args
        if env:
            config["env"] = env
        if allowed_tools:
            config["allowed_tools"] = allowed_tools
        toolsets[name] = config

    def remove_mcp(self, name: str) -> bool:
        """Remove an MCP toolset. Returns True if removed."""
        toolsets = self._get_mcp_toolsets()
        if name in toolsets:
            del toolsets[name]
            return True
        return False

    def list_mcp_toolsets(self) -> list[str]:
        """List configured MCP toolsets."""
        return list(self._get_mcp_toolsets().keys())

    # ----- MCP Server Mode -----

    def set_allow_mcp(self, enabled: bool) -> None:
        """Enable or disable MCP server mode."""
        self.load()["allow_mcp"] = enabled

    def get_allow_mcp(self) -> bool:
        """Check if MCP server mode is enabled."""
        return self.load().get("allow_mcp", False)
