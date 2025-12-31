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
import json
import typing
import uuid
from collections import abc
from pathlib import Path

import yaml
from agentic_patterns.domain_exploration import ExplorationBoundary
from agentic_patterns.domain_exploration import KnowledgeStore
from agentic_patterns.domain_exploration import explore_domain
from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools
from pydantic_ai.models import openai as openai_models
from pydantic_ai.providers import ollama as ollama_providers
from soliplex import config

from crazy_glue.factories.analysis.room_editor import RoomConfigEditor

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = (
    ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]
)

# Default paths
DEFAULT_MAP_PATH = "db/project_map.json"
DEFAULT_ROOTS = ["src/crazy_glue", "rooms/"]
PENDING_TOOL_PATH = "db/pending_tool.json"
TOOLS_DIR = "src/crazy_glue/tools"
PROMPTS_PATH = "db/prompts.json"

# Python reserved keywords that can't be used as identifiers
PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})


def _sanitize_identifier(name: str) -> tuple[str, str | None]:
    """Convert a user-provided name into a valid Python identifier.

    Args:
        name: User-provided tool/function name

    Returns:
        Tuple of (sanitized_name, error_message).
        If error_message is not None, sanitization failed.
    """
    import keyword
    import re

    if not name or not name.strip():
        return "", "Name cannot be empty"

    # Start with lowercase, strip whitespace
    result = name.strip().lower()

    # Replace common separators with underscores
    result = re.sub(r"[-\s.]+", "_", result)

    # Remove any character that isn't alphanumeric or underscore
    result = re.sub(r"[^a-z0-9_]", "", result)

    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result)

    # Strip leading/trailing underscores
    result = result.strip("_")

    # If starts with digit, prefix with underscore
    if result and result[0].isdigit():
        result = f"_{result}"

    # Check if empty after sanitization
    if not result:
        return "", f"Name '{name}' contains no valid identifier characters"

    # Check for Python keywords
    if keyword.iskeyword(result) or result in PYTHON_KEYWORDS:
        result = f"{result}_tool"

    # Validate it's a valid identifier
    if not result.isidentifier():
        return "", f"Could not create valid identifier from '{name}'"

    return result, None


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
    _model: typing.Any = dataclasses.field(default=None, repr=False)

    output_type = None

    @property
    def model_name(self) -> str:
        """Model name from room config."""
        default = "gpt-oss:latest"
        return self.agent_config.extra_config.get("model_name", default)

    def _get_model(self):
        """Create model using soliplex configuration."""
        if self._model is None:
            installation_config = self.agent_config._installation_config
            provider_base_url = installation_config.get_environment(
                "OLLAMA_BASE_URL"
            )
            provider = ollama_providers.OllamaProvider(
                base_url=f"{provider_base_url}/v1",
            )
            self._model = openai_models.OpenAIChatModel(
                model_name=self.model_name,
                provider=provider,
            )
        return self._model

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

    # Available tool types for rooms
    AVAILABLE_TOOLS = {
        "soliplex.tools.search_documents": {
            "description": "RAG document search",
            "requires": "rag_lancedb_stem config",
        },
        "soliplex.tools.research_report": {
            "description": "RAG research report generation",
            "requires": "rag_lancedb_stem config",
        },
        "soliplex.tools.ask_with_rich_citations": {
            "description": "RAG Q&A with citations",
            "requires": "rag_lancedb_stem config",
        },
    }

    def _list_available_tools(self) -> list[dict]:
        """List available tool types that can be added to rooms."""
        tools = []
        for name, info in self.AVAILABLE_TOOLS.items():
            tools.append({
                "name": name,
                "description": info["description"],
                "requires": info["requires"],
            })
        return tools

    def _manage_tool(
        self,
        room_id: str,
        action: str,
        tool_name: str,
    ) -> dict:
        """Add or remove a tool from a room.

        Args:
            room_id: The room to modify
            action: "add" or "remove"
            tool_name: Tool name to add/remove

        Returns:
            Dict with status and any errors.
        """
        editor, error = self._get_room_editor(room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        try:
            if action == "add":
                # Validate tool name
                if tool_name not in self.AVAILABLE_TOOLS:
                    avail = ", ".join(self.AVAILABLE_TOOLS.keys())
                    msg = f"Unknown tool: {tool_name}. Available: {avail}"
                    return {"status": "error", "message": msg}

                editor.add_tool(tool_name)
                editor.save()

                # Validate
                errors = editor.validate()
                if errors:
                    return {
                        "status": "warning",
                        "message": f"Added but validation issues: {errors}",
                    }

                msg = f"Added {tool_name} to {room_id}"
                return {"status": "success", "message": msg}

            elif action == "remove":
                if editor.remove_tool(tool_name):
                    editor.save()
                    msg = f"Removed {tool_name} from {room_id}"
                    return {"status": "success", "message": msg}
                else:
                    msg = f"Tool {tool_name} not found in {room_id}"
                    return {"status": "error", "message": msg}

            else:
                msg = f"Unknown action: {action}"
                return {"status": "error", "message": msg}

        except Exception as e:
            return {"status": "error", "message": f"Failed: {e}"}

    def _manage_mcp(
        self,
        room_id: str,
        action: str,
        name: str,
        kind: str = None,
        url: str = None,
        command: str = None,
    ) -> dict:
        """Add or remove an MCP client toolset from a room.

        Args:
            room_id: The room to modify
            action: "add" or "remove"
            name: MCP toolset name
            kind: "http" or "stdio" (for add)
            url: URL for HTTP MCP (for add)
            command: Command for stdio MCP (for add)

        Returns:
            Dict with status and any errors.
        """
        editor, error = self._get_room_editor(room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        try:
            if action == "add":
                if kind == "http":
                    if not url:
                        return {"status": "error", "message": "URL required"}
                    editor.add_mcp_http(name, url)
                elif kind == "stdio":
                    if not command:
                        msg = "Command required for stdio MCP"
                        return {"status": "error", "message": msg}
                    editor.add_mcp_stdio(name, command)
                else:
                    msg = f"Unknown MCP kind: {kind}. Use http or stdio."
                    return {"status": "error", "message": msg}

                editor.save()

                # Validate
                errors = editor.validate()
                if errors:
                    return {
                        "status": "warning",
                        "message": f"Added but validation issues: {errors}",
                    }

                msg = f"Added MCP {name} ({kind}) to {room_id}"
                return {"status": "success", "message": msg}

            elif action == "remove":
                if editor.remove_mcp(name):
                    editor.save()
                    msg = f"Removed MCP {name} from {room_id}"
                    return {"status": "success", "message": msg}
                else:
                    msg = f"MCP {name} not found in {room_id}"
                    return {"status": "error", "message": msg}

            else:
                msg = f"Unknown action: {action}"
                return {"status": "error", "message": msg}

        except Exception as e:
            return {"status": "error", "message": f"Failed: {e}"}

    def _list_secrets(self) -> list[dict]:
        """List configured secrets (names only, not values)."""
        secrets = []
        for secret in self.installation.secrets:
            resolved = secret._resolved is not None
            secrets.append({
                "name": secret.secret_name,
                "resolved": resolved,
                "sources": [type(s).__name__ for s in secret.sources],
            })
        return secrets

    def _check_secret(self, name: str) -> dict:
        """Check if a secret is configured and resolved."""
        for secret in self.installation.secrets:
            if secret.secret_name == name:
                resolved = secret._resolved is not None
                return {
                    "name": name,
                    "configured": True,
                    "resolved": resolved,
                    "sources": [type(s).__name__ for s in secret.sources],
                }
        return {
            "name": name,
            "configured": False,
            "resolved": False,
            "sources": [],
        }

    def _toggle_mcp_server(self, room_id: str, enable: bool) -> dict:
        """Enable or disable MCP server mode for a room.

        Args:
            room_id: The room to modify
            enable: True to enable, False to disable

        Returns:
            Dict with status and any errors.
        """
        editor, error = self._get_room_editor(room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        try:
            editor.set_allow_mcp(enable)
            editor.save()

            # Validate
            errors = editor.validate()
            if errors:
                return {
                    "status": "warning",
                    "message": f"Saved but validation issues: {errors}",
                }

            action = "enabled" if enable else "disabled"
            msg = f"MCP server mode {action} for {room_id}"
            return {"status": "success", "message": msg}

        except Exception as e:
            return {"status": "error", "message": f"Failed: {e}"}

    # ----- Tool Generation (Milestone 7) -----

    def _get_pending_tool_path(self) -> Path:
        """Get path to pending tool file."""
        return _get_project_root() / PENDING_TOOL_PATH

    def _load_pending_tool(self) -> dict | None:
        """Load pending tool from staging file."""
        path = self._get_pending_tool_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_pending_tool(self, tool_data: dict) -> None:
        """Save pending tool to staging file."""
        path = self._get_pending_tool_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tool_data, indent=2))

    def _clear_pending_tool(self) -> None:
        """Clear pending tool staging file."""
        path = self._get_pending_tool_path()
        if path.exists():
            path.unlink()

    async def _generate_tool_code(
        self,
        name: str,
        description: str,
        refinement: str | None = None,
    ) -> tuple[str, str]:
        """Generate tool code and description from user request using LLM.

        Args:
            name: Tool function name (snake_case)
            description: What the tool should do
            refinement: Optional refinement to previous generation

        Returns:
            Tuple of (generated_code, generated_description).
            The description is a concise summary suitable for tool_description.
        """
        import pydantic
        from pydantic_ai import Agent

        # Define structured output model
        class GeneratedTool(pydantic.BaseModel):
            """LLM output containing generated tool code and description."""

            code: str = pydantic.Field(
                description="Complete Python code for the tool"
            )
            summary: str = pydantic.Field(
                description="Concise 1-sentence tool description"
            )

        # Name is already sanitized by _generate_tool
        func_name = name
        model_name = "".join(w.capitalize() for w in func_name.split("_"))
        model_name += "Result"

        # Build prompt for LLM code generation
        refinement_line = ""
        if refinement:
            refinement_line = f"**Refinement**: {refinement}\n"
        prompt = f"""Generate a Python tool for pydantic-ai.

**Tool name**: {func_name}
**Description**: {description}
{refinement_line}
**Requirements**:
1. Create a Pydantic model `{model_name}` for structured output
2. Create an async function `{func_name}` that returns `{model_name}`
3. Add a docstring to the function describing what it does (IMPORTANT!)
4. Function should NOT use RunContext - just take necessary parameters
5. Implement the ACTUAL logic, not a placeholder
6. Handle errors gracefully, returning error info in the model
7. Use appropriate imports (pathlib, pydantic, etc.)

**Example pattern** (adapt fields/logic to the actual task):

import pydantic

class SearchResult(pydantic.BaseModel):
    found: bool
    matches: list[str]
    count: int
    error: str | None = None

async def search_files(pattern: str, path: str = ".") -> SearchResult:
    \"\"\"Search for files matching a glob pattern.\"\"\"
    from pathlib import Path
    try:
        target = Path(path).resolve()
        matches = [str(p) for p in target.rglob(pattern)]
        return SearchResult(
            found=len(matches) > 0,
            matches=matches,
            count=len(matches),
        )
    except Exception as e:
        return SearchResult(found=False, matches=[], count=0, error=str(e))

Provide:
1. **code**: Complete Python code (no markdown fences)
2. **summary**: Concise 1-sentence description (e.g., "Find large files")"""

        model = self._get_model()
        agent: Agent[None, GeneratedTool] = Agent(
            model, output_type=GeneratedTool
        )
        result = await agent.run(prompt)
        tool_output = result.output

        code = tool_output.code
        summary = tool_output.summary

        # Clean up any markdown fences if present
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        # Add module docstring if missing
        if not code.startswith('"""'):
            header = f'"""\nTool: {name}\n\n{summary}\n"""\n\n'
            code = header + code

        return code, summary

    async def _generate_tool(
        self,
        room_id: str,
        name: str,
        description: str,
    ) -> dict:
        """Generate a tool and stage it for approval.

        Args:
            room_id: Target room for the tool
            name: Tool name (will be sanitized to valid Python identifier)
            description: What the tool should do

        Returns:
            Dict with status, code preview, and any errors.
        """
        # Sanitize and validate tool name first
        func_name, error = _sanitize_identifier(name)
        if error:
            return {"status": "error", "message": error}

        # Validate room exists and is managed
        editor, error = self._get_room_editor(room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        # Generate the code and description using LLM (pass sanitized name)
        code, generated_description = await self._generate_tool_code(
            func_name, description
        )

        # Stage tool (sanitized name + LLM-generated description)
        tool_data = {
            "room_id": room_id,
            "name": func_name,  # Store sanitized name
            "original_name": name,  # Keep original for display
            "description": generated_description,  # Use LLM-generated summary
            "user_description": description,  # Keep user's original request
            "code": code,
            "file_path": f"{TOOLS_DIR}/{func_name}.py",
        }
        self._save_pending_tool(tool_data)

        return {
            "status": "staged",
            "code": code,
            "description": generated_description,
            "file_path": tool_data["file_path"],
            "message": f"Tool '{name}' staged for {room_id}",
        }

    def _apply_pending_tool(self) -> dict:
        """Apply the staged tool - validate, write, wire to room, check-config.

        Process:
        1. Validate Python syntax (AST parse)
        2. Validate code compiles
        3. Write tool file to disk
        4. Wire tool into room's YAML config
        5. Run check-config to validate
        """
        pending = self._load_pending_tool()
        if not pending:
            return {"status": "error", "message": "No pending tool to apply"}

        project_root = _get_project_root()
        file_path = project_root / pending["file_path"]
        name = pending["name"]
        room_id = pending["room_id"]
        code = pending["code"]

        # Validate Python syntax
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "error",
                "message": f"Syntax error in generated code: {e}",
            }

        # Validate code compiles
        try:
            compile(code, str(file_path), "exec")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Compilation error: {e}",
            }

        # Build module path for tool registration
        # Strip "src/" prefix since it's not part of the Python module path
        file_path_str = pending["file_path"]
        module_path = file_path_str
        if module_path.startswith("src/"):
            module_path = module_path[4:]  # Remove "src/"
        tool_module = module_path.replace("/", ".").replace(".py", "")

        # The tool_name must be module.function, not just module
        # Name is already sanitized when stored in pending_tool
        func_name = name
        # e.g., "crazy_glue.tools.big_file.big_file"
        tool_dotted_path = f"{tool_module}.{func_name}"

        try:
            # Write the tool file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code)

            # Validate module can be imported AND function exists
            try:
                module = importlib.import_module(tool_module)
                tool_func = getattr(module, func_name, None)
                if tool_func is None:
                    if file_path.exists():
                        file_path.unlink()
                    msg = f"Function '{func_name}' not found in {tool_module}"
                    return {"status": "error", "message": msg}
                if not callable(tool_func):
                    if file_path.exists():
                        file_path.unlink()
                    msg = f"'{func_name}' in {tool_module} is not callable"
                    return {"status": "error", "message": msg}
            except ImportError as e:
                # Cleanup and return error
                if file_path.exists():
                    file_path.unlink()
                return {
                    "status": "error",
                    "message": f"Import failed for {tool_module}: {e}",
                }

            # Wire tool into room config (full dotted path to function)
            editor, error = self._get_room_editor(
                room_id, require_managed=True
            )
            if error:
                return {"status": "error", "message": error}

            editor.add_tool(tool_dotted_path)
            editor.save()

            # Validate saved YAML is parseable
            try:
                yaml.safe_load(editor.config_path.read_text())
            except yaml.YAMLError as e:
                # Rollback
                if file_path.exists():
                    file_path.unlink()
                return {
                    "status": "error",
                    "message": f"Invalid YAML after save: {e}",
                }

            # Run check-config to validate
            errors = editor.validate()
            if errors:
                # Rollback: remove tool from config
                editor.remove_tool(tool_dotted_path)
                editor.save()
                # Also remove the file
                if file_path.exists():
                    file_path.unlink()
                err_msg = "; ".join(errors)
                return {
                    "status": "error",
                    "message": f"Config validation failed: {err_msg}",
                }

            # Success - clear staging
            self._clear_pending_tool()

            return {
                "status": "success",
                "message": f"Tool '{name}' added to {room_id}",
                "file_path": str(file_path),
                "tool_name": tool_dotted_path,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to apply: {e}"}

    def _discard_pending_tool(self) -> dict:
        """Discard the staged tool."""
        pending = self._load_pending_tool()
        if not pending:
            return {"status": "error", "message": "No pending tool to discard"}

        name = pending["name"]
        self._clear_pending_tool()
        msg = f"Discarded pending tool '{name}'"
        return {"status": "success", "message": msg}

    async def _refine_pending_tool(self, refinement: str) -> dict:
        """Refine the staged tool with additional requirements."""
        pending = self._load_pending_tool()
        if not pending:
            return {"status": "error", "message": "No pending tool to refine"}

        # Regenerate with refinement (use user_description if available)
        base_desc = pending.get("user_description", pending["description"])
        code, new_description = await self._generate_tool_code(
            pending["name"], base_desc, refinement=refinement
        )

        # Update staging with new code and description
        pending["description"] = new_description
        pending["code"] = code
        self._save_pending_tool(pending)

        return {
            "status": "staged",
            "code": code,
            "description": new_description,
            "file_path": pending["file_path"],
            "message": f"Tool refined with: {refinement}",
        }

    # ----- Prompt Library (Milestone 8) -----

    def _get_prompts_path(self) -> Path:
        """Get path to prompts library file."""
        return _get_project_root() / PROMPTS_PATH

    def _load_prompts(self) -> dict:
        """Load prompts from library file."""
        path = self._get_prompts_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("prompts", {})
            except (json.JSONDecodeError, KeyError):
                return {}
        return {}

    def _save_prompts(self, prompts: dict) -> None:
        """Save prompts to library file."""
        path = self._get_prompts_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"prompts": prompts}
        path.write_text(json.dumps(data, indent=2))

    def _list_prompts(self) -> list[dict]:
        """List all saved prompts."""
        prompts = self._load_prompts()
        result = []
        for key, prompt in prompts.items():
            result.append({
                "id": key,
                "name": prompt.get("name", key),
                "preview": prompt.get("content", "")[:60] + "...",
                "created": prompt.get("created", ""),
            })
        return sorted(result, key=lambda p: p["name"])

    def _get_prompt(self, name: str) -> dict | None:
        """Get a specific prompt by name/id."""
        prompts = self._load_prompts()
        # Try exact match first
        if name in prompts:
            return prompts[name]
        # Try case-insensitive match
        name_lower = name.lower()
        for key, prompt in prompts.items():
            if key.lower() == name_lower:
                return prompt
        return None

    def _add_prompt(self, name: str, content: str) -> dict:
        """Add a new prompt to the library."""
        import datetime

        prompts = self._load_prompts()

        # Create slug from name
        slug = name.lower().replace(" ", "-").replace("_", "-")

        if slug in prompts:
            return {
                "status": "error",
                "message": f"Prompt '{slug}' already exists",
            }

        prompts[slug] = {
            "name": name,
            "content": content,
            "created": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        self._save_prompts(prompts)

        return {
            "status": "success",
            "message": f"Prompt '{name}' saved as '{slug}'",
            "id": slug,
        }

    def _remove_prompt(self, name: str) -> dict:
        """Remove a prompt from the library."""
        prompts = self._load_prompts()

        # Find the prompt
        key_to_remove = None
        name_lower = name.lower()
        for key in prompts:
            if key.lower() == name_lower:
                key_to_remove = key
                break

        if not key_to_remove:
            return {
                "status": "error",
                "message": f"Prompt '{name}' not found",
            }

        del prompts[key_to_remove]
        self._save_prompts(prompts)

        return {
            "status": "success",
            "message": f"Prompt '{key_to_remove}' removed",
        }

    def _use_prompt(self, room_id: str, prompt_name: str) -> dict:
        """Apply a prompt to a room's system prompt."""
        # Get the prompt
        prompt = self._get_prompt(prompt_name)
        if not prompt:
            return {
                "status": "error",
                "message": f"Prompt '{prompt_name}' not found",
            }

        # Get the room editor
        editor, error = self._get_room_editor(room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        try:
            editor.set_system_prompt(prompt["content"])
            editor.save()

            # Validate
            errors = editor.validate()
            if errors:
                return {
                    "status": "warning",
                    "message": f"Applied but validation issues: {errors}",
                }

            p_name = prompt.get("name", prompt_name)
            msg = f"Applied prompt '{p_name}' to {room_id}"
            return {"status": "success", "message": msg}

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
            # Validate YAML before writing
            if rel_path.endswith(".yaml") or rel_path.endswith(".yml"):
                try:
                    yaml.safe_load(content)
                except yaml.YAMLError as e:
                    errors.append(f"Invalid YAML in {rel_path}: {e}")
                    continue

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

        elif prompt_lower == "list tools":
            delta = "\nListing available tools..."
            think_part.content += delta
            tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

            tools = self._list_available_tools()
            response_parts.append("## Available Tools\n\n")
            for t in tools:
                response_parts.append(f"### `{t['name']}`\n")
                response_parts.append(f"- {t['description']}\n")
                response_parts.append(f"- Requires: {t['requires']}\n\n")

        elif "generate tool" in prompt_lower:
            # Parse: generate tool <room-id> <name> <description...>
            parts = user_prompt.split(maxsplit=4)
            if len(parts) < 5:
                response_parts.append("## Generate Tool\n\n")
                usage = "Usage: `generate tool <room-id> <name> <desc>`\n\n"
                response_parts.append(usage)
                response_parts.append("**Example:**\n")
                example = "generate tool my-room list_files List directory"
                response_parts.append(f"`{example}`\n")
            else:
                room_id = parts[2]
                name = parts[3]
                description = parts[4]
                delta = f"\nGenerating tool {name} for {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = await self._generate_tool(room_id, name, description)
                if result["status"] == "staged":
                    response_parts.append("## Tool Generated (Staged)\n\n")
                    response_parts.append(f"**Name**: `{name}`\n")
                    response_parts.append(f"**Room**: `{room_id}`\n")
                    gen_desc = result.get('description', '')
                    response_parts.append(f"**Description**: {gen_desc}\n")
                    file_path = result['file_path']
                    response_parts.append(f"**File**: `{file_path}`\n\n")
                    response_parts.append("**Preview:**\n")
                    code_block = f"```python\n{result['code']}```\n\n"
                    response_parts.append(code_block)
                    response_parts.append("**Commands:**\n")
                    response_parts.append("- `apply tool` - Save and add\n")
                    response_parts.append("- `discard tool` - Discard\n")
                    response_parts.append("- Or describe changes to refine\n")
                else:
                    response_parts.append(f"## Error\n\n{result['message']}\n")

        elif prompt_lower == "apply tool":
            pending = self._load_pending_tool()
            if not pending:
                response_parts.append("## No Pending Tool\n\n")
                response_parts.append("Generate a tool first with:\n")
                usage = "`generate tool <room-id> <name> <desc>`\n"
                response_parts.append(usage)
            else:
                delta = f"\nApplying tool {pending['name']}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._apply_pending_tool()
                if result["status"] == "success":
                    response_parts.append("## Tool Applied\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    file_path = result['file_path']
                    response_parts.append(f"**File**: `{file_path}`\n\n")
                    restart = "Restart soliplex to load the tool.\n"
                    response_parts.append(restart)
                elif result["status"] == "warning":
                    warn = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn)
                else:
                    response_parts.append(f"## Error\n\n{result['message']}\n")

        elif prompt_lower == "discard tool":
            pending = self._load_pending_tool()
            if not pending:
                response_parts.append("## No Pending Tool\n\n")
                response_parts.append("Nothing to discard.\n")
            else:
                result = self._discard_pending_tool()
                response_parts.append("## Tool Discarded\n\n")
                response_parts.append(f"{result['message']}\n")

        elif "add tool" in prompt_lower:
            # Parse: add tool <room-id> <tool-name>
            parts = user_prompt.split()
            if len(parts) < 4:
                usage = "Usage: `add tool <room-id> <tool-name>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                tool_name = parts[3]
                delta = f"\nAdding {tool_name} to {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._manage_tool(room_id, "add", tool_name)
                if result["status"] == "success":
                    response_parts.append("## Added Tool\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    restart = "Restart soliplex to apply changes.\n"
                    response_parts.append(restart)
                elif result["status"] == "warning":
                    warn = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn)
                else:
                    response_parts.append(f"## Error\n\n{result['message']}\n")

        elif "remove tool" in prompt_lower:
            # Parse: remove tool <room-id> <tool-name>
            parts = user_prompt.split()
            if len(parts) < 4:
                usage = "Usage: `remove tool <room-id> <tool-name>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                tool_name = parts[3]
                delta = f"\nRemoving {tool_name} from {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._manage_tool(room_id, "remove", tool_name)
                if result["status"] == "success":
                    response_parts.append("## Removed Tool\n\n")
                    response_parts.append(f"{result['message']}\n")
                else:
                    response_parts.append(f"## Error\n\n{result['message']}\n")

        elif prompt_lower == "list mcp":
            delta = "\nListing MCP toolset options..."
            think_part.content += delta
            tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

            response_parts.append("## MCP Client Toolsets\n\n")
            response_parts.append("**HTTP MCP** (`kind: http`)\n")
            response_parts.append("- Connects to remote MCP servers\n")
            response_parts.append("- `add mcp http <room> <name> <url>`\n\n")
            response_parts.append("**Stdio MCP** (`kind: stdio`)\n")
            response_parts.append("- Runs local MCP server as subprocess\n")
            response_parts.append("- `add mcp stdio <room> <name> <cmd>`\n")

        elif "add mcp http" in prompt_lower:
            # Parse: add mcp http <room-id> <name> <url>
            parts = user_prompt.split()
            if len(parts) < 6:
                usage = "Usage: `add mcp http <room-id> <name> <url>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[3]
                name = parts[4]
                url = parts[5]
                delta = f"\nAdding HTTP MCP {name} to {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._manage_mcp(
                    room_id, "add", name, kind="http", url=url
                )
                if result["status"] == "success":
                    response_parts.append("## Added MCP Toolset\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    restart = "Restart soliplex to apply changes.\n"
                    response_parts.append(restart)
                elif result["status"] == "warning":
                    warn = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn)
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)

        elif "add mcp stdio" in prompt_lower:
            # Parse: add mcp stdio <room-id> <name> <command...>
            parts = user_prompt.split(maxsplit=5)
            if len(parts) < 6:
                usage = "Usage: `add mcp stdio <room-id> <name> <command>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[3]
                name = parts[4]
                command = parts[5]
                delta = f"\nAdding stdio MCP {name} to {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._manage_mcp(
                    room_id, "add", name, kind="stdio", command=command
                )
                if result["status"] == "success":
                    response_parts.append("## Added MCP Toolset\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    restart = "Restart soliplex to apply changes.\n"
                    response_parts.append(restart)
                elif result["status"] == "warning":
                    warn = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn)
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)

        elif "remove mcp" in prompt_lower:
            # Parse: remove mcp <room-id> <name>
            parts = user_prompt.split()
            if len(parts) < 4:
                usage = "Usage: `remove mcp <room-id> <name>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                name = parts[3]
                delta = f"\nRemoving MCP {name} from {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._manage_mcp(room_id, "remove", name)
                if result["status"] == "success":
                    response_parts.append("## Removed MCP Toolset\n\n")
                    response_parts.append(f"{result['message']}\n")
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)

        elif prompt_lower == "list secrets":
            delta = "\nListing configured secrets..."
            think_part.content += delta
            tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

            secrets = self._list_secrets()
            response_parts.append("## Configured Secrets\n\n")
            if secrets:
                for s in secrets:
                    status = "resolved" if s["resolved"] else "NOT resolved"
                    response_parts.append(f"- `{s['name']}`: {status}\n")
                    sources = ", ".join(s["sources"])
                    response_parts.append(f"  - Sources: {sources}\n")
            else:
                response_parts.append("No secrets configured.\n")

        elif "check secret" in prompt_lower:
            # Parse: check secret <name>
            parts = user_prompt.split()
            if len(parts) < 3:
                response_parts.append("Usage: `check secret <name>`\n")
            else:
                name = parts[2]
                delta = f"\nChecking secret: {name}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                info = self._check_secret(name)
                response_parts.append(f"## Secret: {name}\n\n")
                if info["configured"]:
                    status = "resolved" if info["resolved"] else "NOT resolved"
                    response_parts.append(f"**Status**: {status}\n")
                    sources = ", ".join(info["sources"])
                    response_parts.append(f"**Sources**: {sources}\n")
                else:
                    response_parts.append("**Not configured**\n\n")
                    hint = "Add to installation.yaml secrets section.\n"
                    response_parts.append(hint)

        elif prompt_lower == "list prompts":
            delta = "\nListing saved prompts..."
            think_part.content += delta
            tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
            yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

            prompts = self._list_prompts()
            response_parts.append("## Prompt Library\n\n")
            if prompts:
                for p in prompts:
                    response_parts.append(f"### `{p['id']}`\n")
                    response_parts.append(f"**Name**: {p['name']}\n")
                    response_parts.append(f"**Preview**: {p['preview']}\n\n")
            else:
                response_parts.append("No prompts saved yet.\n\n")
                hint = "Use `add prompt <name>` to create one.\n"
                response_parts.append(hint)

        elif "show prompt" in prompt_lower:
            # Parse: show prompt <name>
            parts = user_prompt.split(maxsplit=2)
            if len(parts) < 3:
                response_parts.append("Usage: `show prompt <name>`\n")
            else:
                name = parts[2]
                prompt = self._get_prompt(name)
                if prompt:
                    response_parts.append(f"## Prompt: {prompt['name']}\n\n")
                    response_parts.append("**Content:**\n")
                    response_parts.append(f"```\n{prompt['content']}\n```\n")
                else:
                    err = f"## Error\n\nPrompt '{name}' not found.\n"
                    response_parts.append(err)

        elif "add prompt" in prompt_lower:
            # Parse: add prompt <name> <content...>
            # or: add prompt <name> (then content on next lines)
            parts = user_prompt.split(maxsplit=2)
            if len(parts) < 3:
                response_parts.append("## Add Prompt\n\n")
                usage = "Usage: `add prompt <name> <content>`\n\n"
                response_parts.append(usage)
                response_parts.append("**Example:**\n")
                example = "add prompt helpful You are a helpful assistant."
                response_parts.append(f"`{example}`\n")
            else:
                name = parts[2].split()[0]  # First word is name
                content = " ".join(parts[2].split()[1:])  # Rest is content

                if not content:
                    response_parts.append("## Error\n\n")
                    response_parts.append("Prompt content required.\n")
                else:
                    delta = f"\nSaving prompt '{name}'..."
                    think_part.content += delta
                    tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                    yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                    result = self._add_prompt(name, content)
                    if result["status"] == "success":
                        response_parts.append("## Prompt Saved\n\n")
                        response_parts.append(f"{result['message']}\n\n")
                        response_parts.append("**Content:**\n")
                        response_parts.append(f"```\n{content}\n```\n")
                    else:
                        err = f"## Error\n\n{result['message']}\n"
                        response_parts.append(err)

        elif "remove prompt" in prompt_lower:
            # Parse: remove prompt <name>
            parts = user_prompt.split()
            if len(parts) < 3:
                response_parts.append("Usage: `remove prompt <name>`\n")
            else:
                name = parts[2]
                delta = f"\nRemoving prompt '{name}'..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._remove_prompt(name)
                if result["status"] == "success":
                    response_parts.append("## Prompt Removed\n\n")
                    response_parts.append(f"{result['message']}\n")
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)

        elif "use prompt" in prompt_lower:
            # Parse: use prompt <room-id> <prompt-name>
            parts = user_prompt.split()
            if len(parts) < 4:
                response_parts.append("## Use Prompt\n\n")
                usage = "Usage: `use prompt <room-id> <prompt-name>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                prompt_name = parts[3]
                delta = f"\nApplying prompt '{prompt_name}' to {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._use_prompt(room_id, prompt_name)
                if result["status"] == "success":
                    response_parts.append("## Prompt Applied\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    restart = "Restart soliplex to apply changes.\n"
                    response_parts.append(restart)
                elif result["status"] == "warning":
                    warn = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn)
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)

        elif "enable mcp-server" in prompt_lower:
            # Parse: enable mcp-server <room-id>
            parts = user_prompt.split()
            if len(parts) < 3:
                usage = "Usage: `enable mcp-server <room-id>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                delta = f"\nEnabling MCP server mode for {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._toggle_mcp_server(room_id, enable=True)
                if result["status"] == "success":
                    response_parts.append("## MCP Server Enabled\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    restart = "Restart soliplex to apply changes.\n"
                    response_parts.append(restart)
                elif result["status"] == "warning":
                    warn = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn)
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)

        elif "disable mcp-server" in prompt_lower:
            # Parse: disable mcp-server <room-id>
            parts = user_prompt.split()
            if len(parts) < 3:
                usage = "Usage: `disable mcp-server <room-id>`\n"
                response_parts.append(usage)
            else:
                room_id = parts[2]
                delta = f"\nDisabling MCP server mode for {room_id}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = self._toggle_mcp_server(room_id, enable=False)
                if result["status"] == "success":
                    response_parts.append("## MCP Server Disabled\n\n")
                    response_parts.append(f"{result['message']}\n\n")
                    restart = "Restart soliplex to apply changes.\n"
                    response_parts.append(restart)
                elif result["status"] == "warning":
                    warn = f"## Warning\n\n{result['message']}\n"
                    response_parts.append(warn)
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)

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
            # Parse: create <name> room [for <intent>]
            # or: create <name> (name only, no "room" suffix)
            words = user_prompt.split()
            name = "NewRoom"
            intent = user_prompt

            # Find "create" or "scaffold" position
            start_idx = 0
            for i, w in enumerate(words):
                if w.lower() in ("create", "scaffold"):
                    start_idx = i + 1
                    break

            # Extract name - words after create/scaffold until "room" or end
            name_words = []
            intent_start = len(words)
            for i in range(start_idx, len(words)):
                if words[i].lower() == "room":
                    intent_start = i + 1
                    break
                name_words.append(words[i])

            if name_words:
                name = " ".join(name_words)
            if intent_start < len(words):
                intent = " ".join(words[intent_start:])

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
            # Check for pending tool - treat unrecognized input as refinement
            pending = self._load_pending_tool()
            if pending and user_prompt.strip():
                delta = f"\nRefining tool {pending['name']}..."
                think_part.content += delta
                tdelta = ai_messages.ThinkingPartDelta(content_delta=delta)
                yield ai_messages.PartDeltaEvent(index=0, delta=tdelta)

                result = await self._refine_pending_tool(user_prompt.strip())
                if result["status"] == "staged":
                    response_parts.append("## Tool Refined\n\n")
                    p_name = pending['name']
                    p_room = pending['room_id']
                    response_parts.append(f"**Name**: `{p_name}`\n")
                    response_parts.append(f"**Room**: `{p_room}`\n")
                    gen_desc = result.get('description', '')
                    response_parts.append(f"**Description**: {gen_desc}\n\n")
                    response_parts.append("**Updated Preview:**\n")
                    code_block = f"```python\n{result['code']}```\n\n"
                    response_parts.append(code_block)
                    response_parts.append("**Commands:**\n")
                    response_parts.append("- `apply tool` - Save and add\n")
                    response_parts.append("- `discard tool` - Discard\n")
                    response_parts.append("- Or describe more changes\n")
                else:
                    err = f"## Error\n\n{result['message']}\n"
                    response_parts.append(err)
            else:
                header = f"## System Architect\n\n{SYSTEM_PROMPT}\n\n"
                response_parts.append(header)
                response_parts.append("**Introspection:**\n")
                response_parts.append("- `rooms` - List registered rooms\n")
                response_parts.append("- `managed` - List managed rooms\n")
                response_parts.append("- `inspect <id>` - Inspect room\n\n")
                response_parts.append("**Room Management:**\n")
                response_parts.append("- `create <name> room` - Create room\n")
                response_parts.append("- `edit <id> <field> <val>`\n")
                response_parts.append("- `add suggestion <id> <text>`\n")
                response_parts.append("- `remove suggestion <id> <idx>`\n\n")
                response_parts.append("**Tool Generation:**\n")
                response_parts.append("- `generate tool <id> <name> <desc>`\n")
                response_parts.append("- `apply tool` - Apply staged tool\n")
                response_parts.append("- `discard tool` - Discard staged\n\n")
                response_parts.append("**Tool Management:**\n")
                response_parts.append("- `list tools` - Show available\n")
                response_parts.append("- `add tool <id> <tool-name>`\n")
                response_parts.append("- `remove tool <id> <tool-name>`\n\n")
                response_parts.append("**MCP Toolsets:**\n")
                response_parts.append("- `list mcp` - Show MCP options\n")
                response_parts.append("- `add mcp http <id> <name> <url>`\n")
                response_parts.append("- `add mcp stdio <id> <name> <cmd>`\n")
                response_parts.append("- `remove mcp <id> <name>`\n\n")
                response_parts.append("**Secrets:**\n")
                response_parts.append("- `list secrets` - Show secrets\n")
                response_parts.append("- `check secret <name>`\n\n")
                response_parts.append("**Prompt Library:**\n")
                response_parts.append("- `list prompts` - Show saved\n")
                response_parts.append("- `show prompt <name>`\n")
                response_parts.append("- `add prompt <name> <content>`\n")
                response_parts.append("- `remove prompt <name>`\n")
                response_parts.append("- `use prompt <id> <name>`\n\n")
                response_parts.append("**MCP Server Mode:**\n")
                response_parts.append("- `enable mcp-server <id>`\n")
                response_parts.append("- `disable mcp-server <id>`\n\n")
                response_parts.append("**Codebase Analysis:**\n")
                response_parts.append("- `refresh` - Build knowledge graph\n")
                response_parts.append("- `find <name>` - Search entities\n")
                response_parts.append("- `show joker|faux|brainstorm`\n")

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
