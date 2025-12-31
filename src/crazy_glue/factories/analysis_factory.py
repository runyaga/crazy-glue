"""
Factory for Analysis Room - The System Architect.

Pattern: Domain Exploration (Cartographer)
Purpose: Build and query a knowledge graph of the codebase

Flow diagram:

```mermaid
flowchart TB
    User[User Request] --> Analyze

    subgraph Analyze[Analysis Phase]
        Refresh[Refresh Graph]
        Query[Query Codebase]
        Read[Read Source]
    end

    subgraph Scaffold[Scaffolding Phase]
        Generate[Generate Files]
        Apply[Apply Changes]
    end

    Analyze --> Scaffold
    Scaffold --> Result[New Room Created]
```
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import typing
import uuid
from collections import abc
from pathlib import Path

from agentic_patterns.domain_exploration import ExplorationBoundary
from agentic_patterns.domain_exploration import KnowledgeStore
from agentic_patterns.domain_exploration import explore_domain
from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools
from soliplex import config

from crazy_glue.factories.analysis.room_editor import RoomConfigEditor

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = (
    ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]
)

# Default paths
DEFAULT_MAP_PATH = "db/project_map.json"
DEFAULT_ROOTS = ["src/crazy_glue", "rooms/"]

# Reference implementations mapping
REFERENCE_IMPLEMENTATIONS = {
    "joker": ("soliplex.examples", "joker_agent_factory"),
    "faux": ("soliplex.examples", "FauxAgent"),
    "brainstorm": (None, "src/crazy_glue/factories/brainstorm_factory.py"),
}

SYSTEM_PROMPT = """\
You are the System Architect, an introspective agent that understands the \
codebase structure and helps scaffold new rooms.

You have access to a knowledge graph of the project built using tree-sitter \
AST parsing. Use your tools to explore the codebase and help users.

**Available Tools:**
- `refresh_knowledge_graph`: Scan the codebase and update the knowledge graph
- `query_graph`: Search for entities (classes, functions, modules) by name
- `read_entity_source`: Read the source code of a specific entity
- `read_reference_implementation`: Get source code of reference patterns
- `scaffold_room`: Generate configuration for a new room
- `apply_scaffold`: Write generated files to disk

**Reference Implementations:**
- "Joker": Agent delegation pattern (one agent calls another)
- "Faux": Mock agent for testing (no LLM calls)
- "Brainstorm": Parallelization pattern with voting

When scaffolding rooms, analyze the user's intent to determine:
1. Simple room (YAML only) - just needs system prompt and tools
2. Complex room (Factory + YAML) - needs custom Python logic
"""


def _extract_prompt(message_history: MessageHistory | None) -> str:
    """Extract the user prompt from message history."""
    if not message_history:
        return ""
    last_msg = message_history[-1]
    if isinstance(last_msg, ai_messages.ModelRequest):
        for part in last_msg.parts:
            if isinstance(part, ai_messages.UserPromptPart):
                return part.content
    return ""


def _get_project_root() -> Path:
    """Get the project root directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


@dataclasses.dataclass
class AnalysisAgent:
    """Agent that explores codebases and scaffolds new rooms."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None

    @property
    def map_path(self) -> str:
        return self.agent_config.extra_config.get("map_path", DEFAULT_MAP_PATH)

    @property
    def roots(self) -> list[str]:
        return self.agent_config.extra_config.get("roots", DEFAULT_ROOTS)

    @property
    def max_depth(self) -> int:
        return self.agent_config.extra_config.get("max_depth", 5)

    @property
    def max_files(self) -> int:
        return self.agent_config.extra_config.get("max_files", 100)

    @property
    def installation(self):
        """Get the live installation config."""
        return self.agent_config._installation_config

    def _list_rooms(self) -> list[dict]:
        """List all registered rooms in the installation."""
        rooms = []
        for room_id, room_cfg in self.installation.room_configs.items():
            agent_cfg = room_cfg.agent_config
            rooms.append({
                "id": room_id,
                "name": room_cfg.name,
                "description": room_cfg.description or "",
                "agent_kind": agent_cfg.kind,
                "factory": getattr(agent_cfg, "factory_name", None),
            })
        return sorted(rooms, key=lambda r: r["id"])

    def _list_managed_rooms(self) -> list[dict]:
        """List rooms created by the architect (managed rooms)."""
        managed = []

        for room_id, room_cfg in self.installation.room_configs.items():
            config_path = getattr(room_cfg, "_config_path", None)
            if not config_path:
                continue

            room_dir = Path(config_path).parent
            try:
                editor = RoomConfigEditor(room_dir)
                if editor.is_managed():
                    agent_cfg = room_cfg.agent_config
                    managed.append({
                        "id": room_id,
                        "name": room_cfg.name,
                        "description": room_cfg.description or "",
                        "agent_kind": agent_cfg.kind,
                        "config_path": str(config_path),
                    })
            except Exception:
                continue

        return sorted(managed, key=lambda r: r["id"])

    def _get_room_editor(
        self, room_id: str, require_managed: bool = True
    ) -> tuple[RoomConfigEditor | None, str | None]:
        """Get a RoomConfigEditor for a room.

        Args:
            room_id: The room ID to get editor for
            require_managed: If True, only allow editing managed rooms

        Returns:
            Tuple of (editor, error_message). Editor is None if error.
        """
        room_cfg = self.installation.room_configs.get(room_id)
        if not room_cfg:
            return None, f"Room `{room_id}` not found."

        config_path = getattr(room_cfg, "_config_path", None)
        if not config_path:
            return None, f"Room `{room_id}` has no config path."

        room_dir = Path(config_path).parent
        try:
            editor = RoomConfigEditor(room_dir)
        except Exception as e:
            return None, f"Failed to load room config: {e}"

        if require_managed and not editor.is_managed():
            return None, (
                f"Room `{room_id}` is not managed by the architect.\n"
                "Only rooms created with `create <name> room` can be edited."
            )

        return editor, None

    def _edit_room(
        self,
        room_id: str,
        field: str,
        value: str,
    ) -> dict:
        """Edit a room's configuration field.

        Args:
            room_id: The room to edit
            field: Field to edit (description, prompt, model, welcome)
            value: New value for the field

        Returns:
            Dict with status and any errors.
        """
        editor, error = self._get_room_editor(room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        field_map = {
            "description": editor.set_description,
            "prompt": editor.set_system_prompt,
            "model": editor.set_model_name,
            "welcome": editor.set_welcome_message,
        }

        if field not in field_map:
            valid = ", ".join(field_map.keys())
            msg = f"Unknown field: {field}. Valid: {valid}"
            return {"status": "error", "message": msg}

        try:
            field_map[field](value)
            editor.save()

            # Validate after save
            errors = editor.validate()
            if errors:
                return {
                    "status": "warning",
                    "message": f"Saved but validation issues: {errors}",
                }

            msg = f"Updated {field} for {room_id}"
            return {"status": "success", "message": msg}
        except Exception as e:
            return {"status": "error", "message": f"Failed to update: {e}"}

    def _manage_suggestion(
        self,
        room_id: str,
        action: str,
        value: str | int,
    ) -> dict:
        """Add or remove a suggestion from a room.

        Args:
            room_id: The room to modify
            action: "add" or "remove"
            value: Text to add, or index to remove

        Returns:
            Dict with status and any errors.
        """
        editor, error = self._get_room_editor(room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        try:
            if action == "add":
                editor.add_suggestion(str(value))
                editor.save()
                msg = f"Added suggestion to {room_id}"
                return {"status": "success", "message": msg}
            elif action == "remove":
                idx = int(value)
                if editor.remove_suggestion(idx):
                    editor.save()
                    msg = f"Removed suggestion {idx}"
                    return {"status": "success", "message": msg}
                else:
                    msg = f"Invalid index: {idx}"
                    return {"status": "error", "message": msg}
            else:
                msg = f"Unknown action: {action}"
                return {"status": "error", "message": msg}
        except ValueError:
            msg = "Remove requires a numeric index"
            return {"status": "error", "message": msg}
        except Exception as e:
            return {"status": "error", "message": f"Failed: {e}"}

    def _inspect_room(self, room_id: str) -> dict | None:
        """Get detailed info about a specific room."""
        room_cfg = self.installation.room_configs.get(room_id)
        if not room_cfg:
            return None

        agent_cfg = room_cfg.agent_config
        result = {
            "id": room_id,
            "name": room_cfg.name,
            "description": room_cfg.description,
            "welcome_message": room_cfg.welcome_message,
            "suggestions": room_cfg.suggestions or [],
            "agent": {
                "kind": agent_cfg.kind,
            },
            "config_path": str(room_cfg._config_path),
        }

        # Add factory-specific info
        if agent_cfg.kind == "factory":
            result["agent"]["factory_name"] = agent_cfg.factory_name
            result["agent"]["extra_config"] = agent_cfg.extra_config or {}
        elif agent_cfg.kind == "default":
            model = getattr(agent_cfg, "model_name", None)
            result["agent"]["model_name"] = model
            prompt = getattr(agent_cfg, "_system_prompt_text", None)
            if prompt:
                result["agent"]["system_prompt_preview"] = prompt[:200]

        # Add tool info
        if room_cfg.tool_configs:
            result["tools"] = list(room_cfg.tool_configs.keys())

        # Add MCP toolsets
        if room_cfg.mcp_client_toolset_configs:
            result["mcp_toolsets"] = list(
                room_cfg.mcp_client_toolset_configs.keys()
            )

        return result

    def _get_map_path(self) -> Path:
        """Get absolute path to knowledge map."""
        project_root = _get_project_root()
        return project_root / self.map_path

    def _load_store(self) -> KnowledgeStore | None:
        """Load existing knowledge store if available."""
        map_path = self._get_map_path()
        if map_path.exists():
            return KnowledgeStore.load(map_path)
        return None

    async def _refresh_knowledge_graph(self, force: bool = False) -> dict:
        """Scan codebase and build/update knowledge graph."""
        map_path = self._get_map_path()
        map_path.parent.mkdir(parents=True, exist_ok=True)

        project_root = _get_project_root()

        boundary = ExplorationBoundary(
            max_depth=self.max_depth,
            max_files=self.max_files,
            dry_run=True,
            include_patterns=["**/*.py", "**/*.yaml", "**/*.md"],
            exclude_patterns=[
                "**/__pycache__/**",
                "**/.git/**",
                "**/.venv/**",
                "**/node_modules/**",
                "**/test*/**",
            ],
        )

        results = []
        for root in self.roots:
            root_path = project_root / root
            if root_path.exists():
                km = await explore_domain(
                    root_path=str(root_path),
                    boundary=boundary,
                    storage_path=str(map_path),
                )
                results.append({
                    "root": root,
                    "entities": len(km.entities),
                    "links": len(km.links),
                    "files_processed": km.files_processed,
                })

        total_entities = sum(r["entities"] for r in results)
        total_links = sum(r["links"] for r in results)

        return {
            "status": "complete",
            "map_path": str(map_path),
            "roots_scanned": results,
            "total_entities": total_entities,
            "total_links": total_links,
        }

    def _query_graph(
        self, query: str, entity_type: str | None = None, limit: int = 10
    ) -> list[dict]:
        """Search knowledge graph for entities matching query."""
        store = self._load_store()
        if store is None:
            return [{"error": "Knowledge graph not found. Run refresh first."}]

        km = store.to_knowledge_map()
        query_lower = query.lower()
        matches = []

        for entity in km.entities:
            if entity_type and entity.entity_type != entity_type:
                continue

            score = 0
            name_lower = entity.name.lower()
            summary_lower = entity.summary.lower()

            if query_lower in name_lower:
                score += 10
                if name_lower == query_lower:
                    score += 20
            if query_lower in summary_lower:
                score += 5

            if score > 0:
                matches.append((score, entity))

        matches.sort(key=lambda x: x[0], reverse=True)
        results = []

        for _, entity in matches[:limit]:
            result = {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "summary": entity.summary,
                "location": entity.location,
            }
            if entity.metadata.get("line"):
                result["line"] = entity.metadata["line"]
            results.append(result)

        return results

    def _read_entity_source(self, entity_id: str) -> str:
        """Read source code for a specific entity."""
        store = self._load_store()
        if store is None:
            return "Error: Knowledge graph not found. Run refresh first."

        km = store.to_knowledge_map()
        entity = None
        for e in km.entities:
            if e.id == entity_id:
                entity = e
                break

        if entity is None:
            return f"Error: Entity '{entity_id}' not found in graph."

        file_path = Path(entity.location)
        if not file_path.exists():
            return f"Error: Source file not found: {entity.location}"

        try:
            content = file_path.read_text()
            lines = content.split("\n")

            line_num = entity.metadata.get("line", 1)
            start = max(0, line_num - 1)
            end = min(len(lines), start + 50)

            snippet = "\n".join(
                f"{i+1:4}: {line}"
                for i, line in enumerate(lines[start:end], start)
            )

            header = f"# {entity.name} ({entity.entity_type})"
            location = f"# {entity.location}:{line_num}"
            return f"{header}\n{location}\n\n{snippet}"
        except Exception as e:
            return f"Error reading file: {e}"

    def _read_reference_implementation(self, name: str) -> str:
        """Get source code for a reference implementation."""
        name_lower = name.lower()
        if name_lower not in REFERENCE_IMPLEMENTATIONS:
            available = ", ".join(REFERENCE_IMPLEMENTATIONS.keys())
            return f"Unknown reference: '{name}'. Available: {available}"

        module_name, target = REFERENCE_IMPLEMENTATIONS[name_lower]

        if module_name is None:
            file_path = _get_project_root() / target
            if not file_path.exists():
                return f"Error: File not found: {target}"
            return f"# {target}\n\n{file_path.read_text()}"

        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, target)
            source = inspect.getsource(obj)
            return f"# {module_name}.{target}\n\n{source}"
        except Exception as e:
            return f"Error loading reference: {e}"

    def _scaffold_room(self, name: str, intent: str) -> dict[str, str]:
        """Generate room configuration files based on intent."""
        name_slug = name.lower().replace(" ", "-").replace("_", "-")

        # Get model name from installation config or use default
        model_name = "gpt-oss:latest"
        try:
            installation = self.agent_config._installation_config
            if installation.agent_configs:
                # Use first configured agent's model
                first_agent = list(installation.agent_configs.values())[0]
                model_name = first_agent.model_name or model_name
        except Exception:
            pass

        yaml_content = f"""id: "{name_slug}"
name: "{name}"
description: "{intent}"
welcome_message: |
  Welcome to the {name} room!

  {intent}

suggestions:
  - "What can you help me with?"
  - "Show me an example"

agent:
  kind: "default"
  model_name: "{model_name}"
  system_prompt: |
    You are a helpful assistant in the {name} room.
    Your purpose: {intent}
"""

        files = {
            f"rooms/{name_slug}/room_config.yaml": yaml_content,
        }

        return files

    def _apply_scaffold(self, files: dict[str, str]) -> dict:
        """Write scaffold files to disk with safety checks."""
        project_root = _get_project_root()
        allowed_prefixes = ["rooms/", "src/crazy_glue/factories/"]
        written = []
        errors = []

        for rel_path, content in files.items():
            if not any(rel_path.startswith(p) for p in allowed_prefixes):
                errors.append(f"Rejected: {rel_path} (not in allowed paths)")
                continue

            file_path = project_root / rel_path
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
                written.append(rel_path)
            except Exception as e:
                errors.append(f"Failed to write {rel_path}: {e}")

        # Validate config after writing
        if written:
            validation_errors = self._validate_config()
            if validation_errors:
                errors.extend(validation_errors)

            # Mark rooms as managed (only if validation passed)
            if not errors:
                for rel_path in written:
                    if rel_path.startswith("rooms/") and rel_path.endswith(
                        "/room_config.yaml"
                    ):
                        room_dir = project_root / Path(rel_path).parent
                        try:
                            editor = RoomConfigEditor(room_dir)
                            editor.mark_as_managed()
                        except Exception:
                            pass  # Non-critical, room still created

        return {
            "written": written,
            "errors": errors,
            "status": "complete" if not errors else "partial",
        }

    def _validate_config(self) -> list[str]:
        """Run soliplex config validation and return errors."""
        import subprocess

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
                # Parse output for validation errors
                for line in result.stdout.split("\n"):
                    line_lower = line.lower()
                    is_validation_err = "validation error" in line_lower
                    is_other_err = "error" in line_lower and ":" in line
                    if is_validation_err or is_other_err:
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

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Stream the analysis session."""
        user_prompt = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        if emitter:
            emitter.update_activity("analysis", {
                "status": "starting",
                "prompt": user_prompt[:60],
            }, activity_id)

        preview = user_prompt[:40]
        think_part = ai_messages.ThinkingPart(f"Analyzing: {preview}...")
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        response_parts = []
        prompt_lower = user_prompt.lower()

        if "refresh" in prompt_lower or "scan" in prompt_lower:
            delta = "\nRefreshing knowledge graph..."
            think_part.content += delta
            delta_event = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=delta_event)

            result = await self._refresh_knowledge_graph()
            response_parts.append("## Knowledge Graph Refreshed\n\n")
            response_parts.append(f"- **Map Path**: {result['map_path']}\n")
            entities = result['total_entities']
            links = result['total_links']
            response_parts.append(f"- **Total Entities**: {entities}\n")
            response_parts.append(f"- **Total Links**: {links}\n\n")
            for root_info in result['roots_scanned']:
                response_parts.append(f"### {root_info['root']}\n")
                ents = root_info['entities']
                files = root_info['files_processed']
                response_parts.append(f"- Entities: {ents}\n")
                response_parts.append(f"- Files: {files}\n\n")

        elif any(k in prompt_lower for k in ("find", "search", "query")):
            words = user_prompt.split()
            query = words[-1] if words else ""
            for i, w in enumerate(words):
                if w.lower() in ("find", "search", "query", "for"):
                    query = " ".join(words[i+1:])
                    break

            delta = f"\nSearching for: {query}..."
            think_part.content += delta
            delta_event = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=delta_event)

            results = self._query_graph(query.strip())
            query_clean = query.strip()
            response_parts.append(f"## Search Results for '{query_clean}'\n\n")

            if results and "error" not in results[0]:
                for r in results:
                    loc = f"{r['location']}:{r.get('line', '?')}"
                    response_parts.append(f"### {r['name']} ({r['type']})\n")
                    response_parts.append(f"- **Location**: `{loc}`\n")
                    response_parts.append(f"- **Summary**: {r['summary']}\n")
                    response_parts.append(f"- **ID**: `{r['id']}`\n\n")
            else:
                response_parts.append("No results found.\n")

        elif "rooms" in prompt_lower or "list room" in prompt_lower:
            delta = "\nListing registered rooms..."
            think_part.content += delta
            tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

            rooms = self._list_rooms()
            response_parts.append("## Registered Rooms\n\n")
            response_parts.append(f"**Total**: {len(rooms)} rooms\n\n")

            # Group by agent kind
            factories = [r for r in rooms if r["agent_kind"] == "factory"]
            defaults = [r for r in rooms if r["agent_kind"] != "factory"]

            if factories:
                response_parts.append("### Factory Agents\n\n")
                for r in factories:
                    desc = r["description"][:50] if r["description"] else ""
                    response_parts.append(f"- **{r['id']}**: {desc}\n")
                response_parts.append("\n")

            if defaults:
                response_parts.append("### Default Agents\n\n")
                for r in defaults:
                    desc = r["description"][:50] if r["description"] else ""
                    response_parts.append(f"- **{r['id']}**: {desc}\n")
                response_parts.append("\n")

            response_parts.append("Use `inspect <room-id>` for details.\n")

        elif "managed" in prompt_lower:
            delta = "\nListing managed rooms..."
            think_part.content += delta
            tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

            managed = self._list_managed_rooms()
            response_parts.append("## Managed Rooms\n\n")

            if managed:
                total = len(managed)
                response_parts.append(f"**Total**: {total} managed rooms\n\n")
                for r in managed:
                    desc = r["description"][:50] if r["description"] else ""
                    response_parts.append(f"- **{r['id']}**: {desc}\n")
                    path = r["config_path"]
                    response_parts.append(f"  - `{path}`\n")
                response_parts.append("\n")
                note = "Only managed rooms can be modified by the architect."
                response_parts.append(f"*{note}*\n")
            else:
                response_parts.append("No managed rooms found.\n\n")
                hint = "Create a room with `create <name> room` to manage it."
                response_parts.append(f"*{hint}*\n")

        elif prompt_lower.startswith("edit "):
            # Parse: edit <room-id> <field> <value>
            parts = user_prompt.split(maxsplit=3)
            if len(parts) < 4:
                response_parts.append("## Edit Room\n\n")
                usage = "Usage: `edit <room-id> <field> <value>`\n\n"
                response_parts.append(usage)
                response_parts.append("**Fields:**\n")
                response_parts.append("- `description` - Room description\n")
                response_parts.append("- `prompt` - System prompt\n")
                response_parts.append("- `model` - LLM model name\n")
                response_parts.append("- `welcome` - Welcome message\n")
            else:
                _, room_id, field, value = parts
                delta = f"\nEditing {room_id} {field}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._edit_room(room_id, field.lower(), value)
                if result["status"] == "success":
                    response_parts.append(f"## Updated {room_id}\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    restart_msg = "Restart soliplex to apply changes.\n"
                    response_parts.append(restart_msg)
                elif result["status"] == "warning":
                    warn_msg = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn_msg)
                else:
                    response_parts.append(f"## Error\n\n{result['message']}\n")

        elif "add suggestion" in prompt_lower:
            # Parse: add suggestion <room-id> <text>
            parts = user_prompt.split(maxsplit=3)
            if len(parts) < 4:
                usage = "Usage: `add suggestion <room-id> <text>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                text = parts[3]
                delta = f"\nAdding suggestion to {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._manage_suggestion(room_id, "add", text)
                if result["status"] == "success":
                    response_parts.append("## Added Suggestion\n\n")
                    added_msg = f"Added to `{room_id}`: \"{text}\"\n"
                    response_parts.append(added_msg)
                else:
                    response_parts.append(f"## Error\n\n{result['message']}\n")

        elif "remove suggestion" in prompt_lower:
            # Parse: remove suggestion <room-id> <index>
            parts = user_prompt.split()
            if len(parts) < 4:
                usage = "Usage: `remove suggestion <room-id> <index>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                index = parts[3]
                delta = f"\nRemoving suggestion {index} from {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._manage_suggestion(room_id, "remove", index)
                if result["status"] == "success":
                    response_parts.append("## Removed Suggestion\n\n")
                    removed = f"Removed suggestion {index} from `{room_id}`\n"
                    response_parts.append(removed)
                else:
                    response_parts.append(f"## Error\n\n{result['message']}\n")

        elif "inspect" in prompt_lower:
            # Extract room id from prompt
            words = user_prompt.split()
            room_id = None
            for i, w in enumerate(words):
                if w.lower() == "inspect" and i + 1 < len(words):
                    room_id = words[i + 1]
                    break

            if not room_id:
                response_parts.append("Usage: `inspect <room-id>`\n")
            else:
                delta = f"\nInspecting room: {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                info = self._inspect_room(room_id)
                if info:
                    response_parts.append(f"## Room: {info['name']}\n\n")
                    response_parts.append(f"**ID**: `{info['id']}`\n")
                    desc = info['description']
                    response_parts.append(f"**Description**: {desc}\n")
                    cfg_path = info['config_path']
                    response_parts.append(f"**Config**: `{cfg_path}`\n\n")

                    agent = info["agent"]
                    response_parts.append("### Agent\n\n")
                    response_parts.append(f"- **Kind**: {agent['kind']}\n")
                    if agent.get("factory_name"):
                        factory = agent['factory_name']
                        response_parts.append(f"- **Factory**: `{factory}`\n")
                    if agent.get("model_name"):
                        model = agent['model_name']
                        response_parts.append(f"- **Model**: {model}\n")
                    if agent.get("extra_config"):
                        response_parts.append("- **Extra Config**:\n")
                        for k, v in agent["extra_config"].items():
                            response_parts.append(f"  - `{k}`: {v}\n")
                    response_parts.append("\n")

                    if info.get("suggestions"):
                        response_parts.append("### Suggestions\n\n")
                        for s in info["suggestions"]:
                            response_parts.append(f"- {s}\n")
                        response_parts.append("\n")

                    if info.get("tools"):
                        response_parts.append("### Tools\n\n")
                        for t in info["tools"]:
                            response_parts.append(f"- `{t}`\n")
                        response_parts.append("\n")

                    if info.get("mcp_toolsets"):
                        response_parts.append("### MCP Toolsets\n\n")
                        for m in info["mcp_toolsets"]:
                            response_parts.append(f"- `{m}`\n")
                        response_parts.append("\n")
                else:
                    all_ids = [r["id"] for r in self._list_rooms()]
                    response_parts.append(f"Room `{room_id}` not found.\n\n")
                    response_parts.append("Available rooms:\n")
                    for rid in all_ids:
                        response_parts.append(f"- `{rid}`\n")

        elif "reference" in prompt_lower or "show" in prompt_lower:
            for ref in REFERENCE_IMPLEMENTATIONS:
                if ref in prompt_lower:
                    delta = f"\nFetching {ref}..."
                    think_part.content += delta
                    tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                    yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)
                    source = self._read_reference_implementation(ref)
                    title = ref.title()
                    response_parts.append(f"## {title} Reference\n\n")
                    response_parts.append(f"```python\n{source}\n```\n")
                    break
            else:
                response_parts.append("## Available References\n\n")
                for ref in REFERENCE_IMPLEMENTATIONS:
                    response_parts.append(f"- `{ref}`\n")

        elif "scaffold" in prompt_lower or "create" in prompt_lower:
            parts = user_prompt.split("room", 1)
            if len(parts) > 1:
                name_part = parts[0].replace("create", "")
                name_part = name_part.replace("scaffold", "").strip()
                name = name_part or "NewRoom"
                intent = parts[1].strip() if len(parts) > 1 else user_prompt
            else:
                name = "NewRoom"
                intent = user_prompt

            delta = f"\nScaffolding room: {name}..."
            think_part.content += delta
            delta_ev = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=delta_ev)

            files = self._scaffold_room(name, intent)

            # Write files immediately
            result = self._apply_scaffold(files)

            response_parts.append("## Room Created\n\n")
            if result["written"]:
                response_parts.append("**Files written:**\n")
                for path in result["written"]:
                    response_parts.append(f"- `{path}`\n")
                response_parts.append("\n")

            if result["errors"]:
                response_parts.append("**Errors:**\n")
                for err in result["errors"]:
                    response_parts.append(f"- {err}\n")
                response_parts.append("\n")

            # Show content for reference
            response_parts.append("**Content:**\n\n")
            for path, content in files.items():
                block = f"```yaml\n# {path}\n{content}```\n\n"
                response_parts.append(block)

            response_parts.append("Restart soliplex to load the new room.\n")

        else:
            header = f"## System Architect\n\n{SYSTEM_PROMPT}\n\n"
            response_parts.append(header)
            response_parts.append("**Introspection:**\n")
            response_parts.append("- `rooms` - List registered rooms\n")
            response_parts.append("- `managed` - List managed rooms\n")
            response_parts.append("- `inspect <id>` - Inspect a room\n\n")
            response_parts.append("**Room Management:**\n")
            response_parts.append("- `create <name> room` - Create room\n")
            response_parts.append("- `edit <id> <field> <val>` - Edit room\n")
            response_parts.append("- `add suggestion <id> <text>`\n")
            response_parts.append("- `remove suggestion <id> <idx>`\n\n")
            response_parts.append("**Codebase Analysis:**\n")
            response_parts.append("- `refresh` - Build knowledge graph\n")
            response_parts.append("- `find <name>` - Search entities\n")
            response_parts.append("- `show joker|faux|brainstorm` - Refs\n")

        if emitter:
            emitter.update_activity("analysis", {
                "status": "complete",
                "prompt": user_prompt[:60],
            }, activity_id)

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        response = "".join(response_parts)
        text_part = ai_messages.TextPart(response)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response)


def create_analysis_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> AnalysisAgent:
    """Factory function to create the analysis agent."""
    return AnalysisAgent(
        agent_config, tool_configs, mcp_client_toolset_configs
    )
