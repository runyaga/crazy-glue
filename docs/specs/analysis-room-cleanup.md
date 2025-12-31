# Spec: Analysis Room Cleanup

## Status: Draft

## Goal

Clean up the 2000-line `analysis_factory.py` without adding complexity.
No new agents, no router, no LLM overhead. Just good code organization.

## Approach

1. **Command handler dispatch** - Replace if/elif with dict lookup
2. **Extract tools** - Move logic to testable pydantic-ai tools
3. **Preserve validations** - Keep all hard-won lessons

## Current State

```python
# analysis_factory.py - ~2000 lines of this:
async def run_stream_events(self, user_prompt, ...):
    if user_prompt.strip().lower() == "rooms":
        # 20 lines
    elif "managed" in user_prompt.lower():
        # 15 lines
    elif "inspect" in user_prompt.lower():
        # 30 lines
    elif "create" in user_prompt.lower() and "room" in user_prompt.lower():
        # 50 lines
    # ... 40 more branches
```

## Target State

```python
# analysis_factory.py - ~300 lines
async def run_stream_events(self, user_prompt, ...):
    cmd, args = parse_command(user_prompt)
    handler = HANDLERS.get(cmd, handle_unknown)
    async for event in handler(self.ctx, args):
        yield event
```

---

## Part 1: Command Parser

Simple keyword-based parser (no LLM):

```python
# src/crazy_glue/factories/analysis/parser.py

import re
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    command: str
    args: dict[str, str]
    raw: str

def parse_command(prompt: str) -> ParsedCommand:
    """Parse user prompt into command and arguments."""
    prompt = prompt.strip()
    lower = prompt.lower()

    # Exact matches first
    if lower == "rooms":
        return ParsedCommand("list_rooms", {}, prompt)

    if lower == "managed":
        return ParsedCommand("list_managed", {}, prompt)

    if lower in ("apply tool", "apply"):
        return ParsedCommand("apply_tool", {}, prompt)

    if lower in ("discard tool", "discard"):
        return ParsedCommand("discard_tool", {}, prompt)

    if lower == "list tools":
        return ParsedCommand("list_tools", {}, prompt)

    if lower == "list prompts":
        return ParsedCommand("list_prompts", {}, prompt)

    if lower == "list mcp":
        return ParsedCommand("list_mcp", {}, prompt)

    if lower == "list secrets":
        return ParsedCommand("list_secrets", {}, prompt)

    if lower in ("refresh", "refresh graph"):
        return ParsedCommand("refresh_graph", {}, prompt)

    # Pattern matches
    if match := re.match(r"inspect\s+(\S+)", lower):
        return ParsedCommand("inspect_room", {"room_id": match.group(1)}, prompt)

    if match := re.match(r"show prompt\s+(\S+)", lower):
        return ParsedCommand("show_prompt", {"name": match.group(1)}, prompt)

    if match := re.match(r"remove prompt\s+(\S+)", lower):
        return ParsedCommand("remove_prompt", {"name": match.group(1)}, prompt)

    if match := re.match(r"check secret\s+(\S+)", lower):
        return ParsedCommand("check_secret", {"name": match.group(1)}, prompt)

    if match := re.match(r"find\s+(.+)", lower):
        return ParsedCommand("find_entity", {"query": match.group(1)}, prompt)

    if match := re.match(r"show\s+(joker|faux|brainstorm)", lower):
        return ParsedCommand("show_reference", {"name": match.group(1)}, prompt)

    # Room creation: "create X room" or "create X room for Y"
    if "create" in lower and "room" in lower:
        # Extract name and optional description
        pattern = r"create\s+(.+?)\s+room(?:\s+(?:for|to|that)\s+(.+))?"
        if match := re.match(pattern, lower):
            name = match.group(1).strip()
            desc = match.group(2).strip() if match.group(2) else name
            return ParsedCommand("create_room", {"name": name, "description": desc}, prompt)

    # Tool generation: "generate tool <room> <name> <description>"
    if lower.startswith("generate tool"):
        parts = prompt.split(maxsplit=4)
        if len(parts) >= 5:
            return ParsedCommand("generate_tool", {
                "room_id": parts[2],
                "name": parts[3],
                "description": parts[4],
            }, prompt)

    # Edit commands: "edit <room> <field> <value>"
    if lower.startswith("edit "):
        parts = prompt.split(maxsplit=3)
        if len(parts) >= 4:
            return ParsedCommand("edit_room", {
                "room_id": parts[1],
                "field": parts[2],
                "value": parts[3],
            }, prompt)

    # Add tool: "add tool <room> <tool-name>"
    if match := re.match(r"add tool\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand("add_tool", {
            "room_id": match.group(1),
            "tool_name": match.group(2),
        }, prompt)

    # Remove tool: "remove tool <room> <tool-name>"
    if match := re.match(r"remove tool\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand("remove_tool", {
            "room_id": match.group(1),
            "tool_name": match.group(2),
        }, prompt)

    # Add suggestion: "add suggestion <room> <text>"
    if match := re.match(r"add suggestion\s+(\S+)\s+(.+)", prompt, re.IGNORECASE):
        return ParsedCommand("add_suggestion", {
            "room_id": match.group(1),
            "text": match.group(2),
        }, prompt)

    # MCP commands
    if match := re.match(r"add mcp http\s+(\S+)\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand("add_mcp_http", {
            "room_id": match.group(1),
            "name": match.group(2),
            "url": match.group(3),
        }, prompt)

    if match := re.match(r"add mcp stdio\s+(\S+)\s+(\S+)\s+(.+)", lower):
        return ParsedCommand("add_mcp_stdio", {
            "room_id": match.group(1),
            "name": match.group(2),
            "command": match.group(3),
        }, prompt)

    if match := re.match(r"remove mcp\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand("remove_mcp", {
            "room_id": match.group(1),
            "name": match.group(2),
        }, prompt)

    if match := re.match(r"enable mcp-server\s+(\S+)", lower):
        return ParsedCommand("enable_mcp_server", {"room_id": match.group(1)}, prompt)

    if match := re.match(r"disable mcp-server\s+(\S+)", lower):
        return ParsedCommand("disable_mcp_server", {"room_id": match.group(1)}, prompt)

    # Prompt library
    if match := re.match(r"add prompt\s+(\S+)\s+(.+)", prompt, re.IGNORECASE):
        return ParsedCommand("add_prompt", {
            "name": match.group(1),
            "content": match.group(2),
        }, prompt)

    if match := re.match(r"use prompt\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand("use_prompt", {
            "room_id": match.group(1),
            "prompt_name": match.group(2),
        }, prompt)

    # Unknown - might be a tool refinement if pending tool exists
    return ParsedCommand("unknown", {"input": prompt}, prompt)
```

---

## Part 2: Handler Registry

```python
# src/crazy_glue/factories/analysis/handlers.py

from typing import AsyncIterator, Callable
from .context import AnalysisContext
from .parser import ParsedCommand

Handler = Callable[[AnalysisContext, ParsedCommand], AsyncIterator[NativeEvent]]

HANDLERS: dict[str, Handler] = {}

def handler(name: str):
    """Decorator to register a command handler."""
    def decorator(fn: Handler) -> Handler:
        HANDLERS[name] = fn
        return fn
    return decorator


# --- Room Handlers ---

@handler("list_rooms")
async def handle_list_rooms(ctx: AnalysisContext, cmd: ParsedCommand):
    """List all registered rooms."""
    rooms = list_rooms(ctx)
    yield text_response(format_room_list(rooms))


@handler("list_managed")
async def handle_list_managed(ctx: AnalysisContext, cmd: ParsedCommand):
    """List managed rooms."""
    rooms = list_managed_rooms(ctx)
    yield text_response(format_room_list(rooms, managed=True))


@handler("inspect_room")
async def handle_inspect_room(ctx: AnalysisContext, cmd: ParsedCommand):
    """Inspect a room's configuration."""
    room_id = cmd.args["room_id"]
    result = inspect_room(ctx, room_id)
    if "error" in result:
        yield text_response(f"Error: {result['error']}")
    else:
        yield text_response(format_room_details(result))


@handler("create_room")
async def handle_create_room(ctx: AnalysisContext, cmd: ParsedCommand):
    """Create a new room."""
    result = await create_room(ctx, cmd.args["name"], cmd.args["description"])
    if result.get("error"):
        yield text_response(f"Error: {result['error']}")
    else:
        yield text_response(f"Created room `{result['room_id']}` at `{result['path']}`")


# --- Tool Handlers ---

@handler("list_tools")
async def handle_list_tools(ctx: AnalysisContext, cmd: ParsedCommand):
    """List available tools."""
    tools = list_available_tools()
    yield text_response(format_tool_list(tools))


@handler("generate_tool")
async def handle_generate_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Generate a custom tool."""
    result = await generate_tool(
        ctx,
        cmd.args["room_id"],
        cmd.args["name"],
        cmd.args["description"],
    )
    if result.get("error"):
        yield text_response(f"Error: {result['error']}")
    else:
        yield text_response(format_staged_tool(result))


@handler("apply_tool")
async def handle_apply_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Apply staged tool."""
    result = await apply_staged_tool(ctx)
    if result.get("error"):
        yield text_response(f"Error: {result['error']}")
    else:
        yield text_response(f"Tool applied to `{result['room_id']}`")


@handler("discard_tool")
async def handle_discard_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Discard staged tool."""
    result = discard_staged_tool(ctx)
    yield text_response(result["message"])


# --- Unknown/Refinement ---

@handler("unknown")
async def handle_unknown(ctx: AnalysisContext, cmd: ParsedCommand):
    """Handle unknown commands or tool refinement."""
    # Check if there's a pending tool - might be refinement
    pending = ctx.load_pending_tool()
    if pending:
        result = await refine_pending_tool(ctx, cmd.args["input"])
        yield text_response(format_staged_tool(result))
    else:
        yield text_response(
            f"Unknown command: `{cmd.raw}`\n\n"
            "Try `rooms`, `inspect <room>`, or `create <name> room`"
        )
```

---

## Part 3: Extracted Tools

Move logic into testable functions:

```python
# src/crazy_glue/factories/analysis/tools/room_ops.py

from ..context import AnalysisContext
from ..room_editor import RoomConfigEditor

def list_rooms(ctx: AnalysisContext) -> list[dict]:
    """List all registered rooms."""
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "agent_kind": r.agent_config.kind if r.agent_config else "default",
        }
        for r in ctx.installation_config.room_configs.values()
    ]


def list_managed_rooms(ctx: AnalysisContext) -> list[dict]:
    """List rooms created by the System Architect."""
    managed_ids = RoomConfigEditor._load_managed_rooms()
    return [
        {"id": r.id, "name": r.name}
        for r in ctx.installation_config.room_configs.values()
        if r.id in managed_ids
    ]


def inspect_room(ctx: AnalysisContext, room_id: str) -> dict:
    """Get detailed room configuration."""
    room = ctx.installation_config.room_configs.get(room_id)
    if not room:
        available = list(ctx.installation_config.room_configs.keys())
        return {"error": f"Room '{room_id}' not found. Available: {available}"}

    return {
        "id": room.id,
        "name": room.name,
        "description": room.description,
        "welcome_message": room.welcome_message,
        "suggestions": room.suggestions,
        "agent": {
            "kind": room.agent_config.kind if room.agent_config else None,
            "model": room.agent_config.model_name if room.agent_config else None,
        },
        "tools": [t.tool_name for t in room.tool_configs],
        "mcp_toolsets": list(room.mcp_client_toolset_configs.keys()),
    }


async def create_room(
    ctx: AnalysisContext,
    name: str,
    description: str,
) -> dict:
    """Create a new room."""
    name_slug = name.lower().replace(" ", "-").replace("_", "-")
    room_path = ctx.project_root / "rooms" / name_slug

    if room_path.exists():
        return {"error": f"Room '{name_slug}' already exists"}

    # Get model from config
    model_name = "gpt-oss:latest"
    try:
        if ctx.installation_config.agent_configs:
            first = list(ctx.installation_config.agent_configs.values())[0]
            model_name = first.model_name or model_name
    except Exception:
        pass

    # Scaffold
    room_path.mkdir(parents=True, exist_ok=True)
    config = f'''id: "{name_slug}"
name: "{name}"
description: "{description}"
welcome_message: |
  Welcome to the {name} room!

  {description}

suggestions:
  - "What can you help me with?"
  - "Show me an example"

agent:
  kind: "default"
  model_name: "{model_name}"
  system_prompt: |
    You are a helpful assistant in the {name} room.
    Your purpose: {description}
'''
    (room_path / "room_config.yaml").write_text(config)

    # Mark as managed
    editor = RoomConfigEditor(room_path)
    editor.load()
    editor.mark_as_managed()

    # Validate
    errors = editor.validate()
    if errors:
        # Rollback
        import shutil
        shutil.rmtree(room_path)
        return {"error": f"Validation failed: {errors}"}

    return {"status": "created", "room_id": name_slug, "path": str(room_path)}
```

---

## Part 4: Context Object

```python
# src/crazy_glue/factories/analysis/context.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from soliplex.config import InstallationConfig
from pydantic_ai.models import Model

@dataclass
class AnalysisContext:
    """Shared context for all handlers."""

    installation_config: InstallationConfig
    project_root: Path
    model: Model

    # Paths
    pending_tool_path: Path = field(init=False)
    managed_rooms_path: Path = field(init=False)
    prompts_path: Path = field(init=False)

    def __post_init__(self):
        db_path = self.project_root / "db"
        self.pending_tool_path = db_path / "pending_tool.json"
        self.managed_rooms_path = db_path / "managed_rooms.json"
        self.prompts_path = db_path / "prompts.json"

    def load_pending_tool(self) -> dict | None:
        if self.pending_tool_path.exists():
            return json.loads(self.pending_tool_path.read_text())
        return None

    def save_pending_tool(self, data: dict) -> None:
        self.pending_tool_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_tool_path.write_text(json.dumps(data, indent=2))

    def clear_pending_tool(self) -> None:
        if self.pending_tool_path.exists():
            self.pending_tool_path.unlink()
```

---

## Part 5: Slim Factory

```python
# src/crazy_glue/factories/analysis_factory.py

"""System Architect agent - room and tool management."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic_ai import messages as ai_messages
from soliplex.config import AgentConfig

from .analysis.context import AnalysisContext
from .analysis.parser import parse_command
from .analysis.handlers import HANDLERS, handle_unknown

# Import handlers to register them
from .analysis import handlers  # noqa: F401


@dataclass
class AnalysisAgent:
    """System Architect for room and tool management."""

    agent_config: AgentConfig
    ctx: AnalysisContext

    async def run_stream_events(
        self,
        user_prompt: str,
        message_history: list,
        activity_id: str | None = None,
    ) -> AsyncIterator[ai_messages.AgentStreamEvent]:
        """Process user command and stream response."""

        # Parse command
        cmd = parse_command(user_prompt)

        # Get handler
        handler = HANDLERS.get(cmd.command, handle_unknown)

        # Execute and yield events
        async for event in handler(self.ctx, cmd):
            yield event


def create_analysis_agent(
    agent_config: AgentConfig,
    tool_configs: list = None,
    mcp_client_toolset_configs: dict = None,
) -> AnalysisAgent:
    """Factory function to create the analysis agent."""

    # Build context
    from .analysis.context import AnalysisContext
    from pathlib import Path

    project_root = Path.cwd()
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    ctx = AnalysisContext(
        installation_config=agent_config._installation_config,
        project_root=project_root,
        model=None,  # Created lazily when needed
    )

    return AnalysisAgent(agent_config=agent_config, ctx=ctx)
```

---

## File Structure

```
src/crazy_glue/factories/
├── analysis/
│   ├── __init__.py
│   ├── context.py           # AnalysisContext dataclass
│   ├── parser.py            # parse_command()
│   ├── handlers.py          # @handler decorated functions
│   ├── formatters.py        # format_room_list(), format_tool_list(), etc.
│   ├── room_editor.py       # (existing) RoomConfigEditor
│   ├── validators.py        # _sanitize_identifier(), validation pipeline
│   └── tools/
│       ├── __init__.py
│       ├── room_ops.py      # list_rooms, create_room, etc.
│       ├── tool_ops.py      # generate_tool, apply_tool, etc.
│       ├── mcp_ops.py       # add_mcp, remove_mcp, etc.
│       └── prompt_ops.py    # prompt library operations
└── analysis_factory.py      # ~50 lines, just wires it together
```

---

## Preserved Validations

All lessons learned stay in `validators.py`:

```python
# src/crazy_glue/factories/analysis/validators.py

# Keep everything from current implementation:
# - _sanitize_identifier()
# - YAML validation after save
# - check-config validation
# - Import validation for tools
# - Function existence check
# - Rollback on failure
```

---

## Tasks

- [ ] Create `analysis/` package structure
- [ ] Implement `parser.py` with all command patterns
- [ ] Implement `context.py`
- [ ] Extract room operations to `tools/room_ops.py`
- [ ] Extract tool operations to `tools/tool_ops.py`
- [ ] Extract MCP operations to `tools/mcp_ops.py`
- [ ] Extract prompt operations to `tools/prompt_ops.py`
- [ ] Implement handlers in `handlers.py`
- [ ] Implement formatters in `formatters.py`
- [ ] Move validators to `validators.py`
- [ ] Slim down `analysis_factory.py`
- [ ] Add unit tests for parser
- [ ] Add unit tests for each tool operation
- [ ] Add integration tests for handlers
- [ ] Verify all existing commands work
- [ ] Run `check-config` validation

## Success Criteria

- [ ] `analysis_factory.py` < 100 lines
- [ ] No if/elif chain in main file
- [ ] All 50+ commands still work
- [ ] All existing tests pass
- [ ] New unit tests for parser (100% coverage)
- [ ] New unit tests for tools
- [ ] No latency increase
- [ ] No new LLM calls for dispatch

## What We're NOT Doing

- ❌ Router agent (LLM classification)
- ❌ Sub-agents (room agent, tool agent)
- ❌ Reflection pattern for all operations
- ❌ Knowledge graph integration
- ❌ Planning pattern
- ❌ Any new LLM overhead for dispatch

## Timeline

This is a refactor, not new features. Estimate: **3-5 days**.
