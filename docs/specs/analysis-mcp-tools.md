# Analysis Room MCP Tools Specification

## Overview

This specification describes exposing Analysis Room (System Architect) operations as MCP tools, enabling external clients to programmatically manage Soliplex installations without using the conversational agent interface.

## Background

### Why Not Agent-via-MCP?

The Analysis Room agent is conversational with multi-turn planning, confirmation flows, and streaming output. MCP is a stateless request-response protocol. These are fundamentally incompatible. See `docs/specs/agent-mcp-analysis.md` for the full analysis.

### Solution: Expose Handlers as Tools

Instead of exposing the agent, we expose the underlying **handler operations** as discrete MCP tools. These are already well-factored, stateless functions that return structured data.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client                               │
│  (Claude Desktop, pydantic-ai, mcp CLI, custom client)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP + MCP Protocol
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Soliplex MCP Server                             │
│              /mcp/analysis                                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  FastMCP                                             │    │
│  │  - Tool discovery                                    │    │
│  │  - Request routing                                   │    │
│  │  - Response serialization                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  MCP Wrapper Layer                                   │    │
│  │  - ArchitectNoArgsMCPWrapper                         │    │
│  │  - ArchitectWithRoomIdMCPWrapper                     │    │
│  │  - ArchitectWithNameMCPWrapper                       │    │
│  │  (Injects tool_config for context access)            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           crazy_glue.analysis.mcp_tools                      │
│                                                              │
│  architect_list_rooms(tool_config) -> RoomListResult        │
│  architect_inspect_room(room_id, tool_config) -> dict       │
│  architect_create_room(room_id, desc, tool_config) -> dict  │
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           crazy_glue.analysis.tools                          │
│                                                              │
│  Existing operations:                                        │
│  - list_rooms(ctx)                                          │
│  - inspect_room(ctx, room_id)                               │
│  - create_room(ctx, room_id, description)                   │
│  - edit_room(ctx, room_id, field, value)                    │
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘
```

## Tool Catalog

### Read Operations

| Tool Name | Parameters | Returns | Description |
|-----------|------------|---------|-------------|
| `architect_list_rooms` | none | `RoomListResult` | List all registered rooms |
| `architect_list_managed` | none | `RoomListResult` | List rooms created by architect |
| `architect_inspect_room` | `room_id: str` | `RoomDetailsResult` | Get room configuration details |
| `architect_list_tools` | none | `ToolListResult` | List available tool templates |
| `architect_list_secrets` | none | `SecretListResult` | List configured secrets (names only) |
| `architect_check_secret` | `name: str` | `SecretCheckResult` | Check if secret is resolved |
| `architect_list_prompts` | none | `PromptListResult` | List saved prompts |
| `architect_show_prompt` | `name: str` | `PromptDetailResult` | Get prompt content |
| `architect_find_entity` | `query: str` | `SearchResult` | Search knowledge graph |
| `architect_show_reference` | `name: str` | `ReferenceResult` | Get reference implementation source |

### Write Operations

| Tool Name | Parameters | Returns | Description |
|-----------|------------|---------|-------------|
| `architect_create_room` | `room_id: str`, `description: str` | `CreateRoomResult` | Create new room |
| `architect_edit_room` | `room_id: str`, `field: str`, `value: str` | `StatusResult` | Edit room field |
| `architect_add_tool` | `room_id: str`, `tool_name: str` | `StatusResult` | Add tool to room |
| `architect_remove_tool` | `room_id: str`, `tool_name: str` | `StatusResult` | Remove tool from room |
| `architect_add_prompt` | `name: str`, `content: str` | `StatusResult` | Save prompt to library |
| `architect_remove_prompt` | `name: str` | `StatusResult` | Delete prompt |
| `architect_add_mcp_http` | `room_id: str`, `name: str`, `url: str` | `StatusResult` | Add HTTP MCP toolset |
| `architect_add_mcp_stdio` | `room_id: str`, `name: str`, `command: str` | `StatusResult` | Add stdio MCP toolset |
| `architect_remove_mcp` | `room_id: str`, `name: str` | `StatusResult` | Remove MCP toolset |
| `architect_toggle_mcp_server` | `room_id: str`, `enabled: bool` | `StatusResult` | Enable/disable MCP server |

### Excluded Operations

These operations are NOT exposed via MCP:

| Operation | Reason |
|-----------|--------|
| `generate_tool` | LLM-based, expensive, returns pending state |
| `apply_tool` | Requires pending state from generate_tool |
| `discard_tool` | Requires pending state |
| `refine_pending_tool` | LLM-based + pending state |
| `refresh_graph` | Long-running, expensive I/O operation |

## Result Models

```python
class RoomInfo(pydantic.BaseModel):
    id: str
    name: str
    description: str
    agent_kind: str
    factory: str | None = None

class RoomListResult(pydantic.BaseModel):
    rooms: list[RoomInfo]
    count: int

class RoomDetailsResult(pydantic.BaseModel):
    id: str
    name: str
    description: str | None
    welcome_message: str | None
    suggestions: list[str]
    agent: dict
    config_path: str
    tools: list[str] | None = None
    mcp_toolsets: list[str] | None = None
    error: str | None = None

class ToolInfo(pydantic.BaseModel):
    name: str
    description: str
    requires: str

class ToolListResult(pydantic.BaseModel):
    tools: list[ToolInfo]

class SecretInfo(pydantic.BaseModel):
    name: str
    resolved: bool
    sources: list[str]

class SecretListResult(pydantic.BaseModel):
    secrets: list[SecretInfo]

class SecretCheckResult(pydantic.BaseModel):
    name: str
    configured: bool
    resolved: bool
    sources: list[str]

class PromptInfo(pydantic.BaseModel):
    id: str
    name: str
    preview: str
    created: str

class PromptListResult(pydantic.BaseModel):
    prompts: list[PromptInfo]

class PromptDetailResult(pydantic.BaseModel):
    name: str
    content: str
    created: str | None = None
    error: str | None = None

class EntityInfo(pydantic.BaseModel):
    id: str
    name: str
    type: str
    summary: str
    location: str
    line: int | None = None

class SearchResult(pydantic.BaseModel):
    query: str
    results: list[EntityInfo]
    count: int

class ReferenceResult(pydantic.BaseModel):
    name: str
    source: str
    error: str | None = None

class CreateRoomResult(pydantic.BaseModel):
    room_id: str
    config_path: str
    status: str
    error: str | None = None

class StatusResult(pydantic.BaseModel):
    status: str  # "success" | "error"
    message: str
    error: str | None = None
```

## Client Integration

### Endpoint

```
https://<soliplex-host>/mcp/analysis
```

The analysis room must have `allow_mcp: true` in its configuration (already set).

### Authentication

Soliplex uses signed tokens via `itsdangerous`. Generate a token:

```bash
soliplex-cli generate-mcp-token --room analysis
```

Or via API:
```
POST /api/tokens/mcp
{
  "room_id": "analysis",
  "max_age": 3600
}
```

### Claude Desktop Configuration

Add to `~/.config/claude-desktop/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "architect": {
      "url": "https://your-soliplex.com/mcp/analysis",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

### pydantic-ai Integration

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

# Connect to architect tools
architect = MCPServerStreamableHTTP(
    url="https://your-soliplex.com/mcp/analysis",
    headers={"Authorization": "Bearer <token>"}
)

# Use in agent
agent = Agent(
    model="openai:gpt-4",
    toolsets=[architect]
)

# Agent can now call architect_list_rooms, architect_create_room, etc.
result = await agent.run("List all the rooms in the installation")
```

### Direct MCP Client

```python
from mcp import Client

async with Client("https://your-soliplex.com/mcp/analysis") as client:
    # List available tools
    tools = await client.list_tools()

    # Call a tool
    result = await client.call_tool(
        "architect_list_rooms",
        {}
    )
    print(result.rooms)
```

### curl Example

```bash
# List rooms via MCP protocol
curl -X POST https://your-soliplex.com/mcp/analysis \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "architect_list_rooms",
      "arguments": {}
    },
    "id": 1
  }'
```

## Implementation

### Files to Create (crazy-glue)

#### `src/crazy_glue/analysis/mcp_tools.py`

```python
"""MCP-compatible tools for Analysis Room operations.

These tools are exposed via Soliplex's MCP server at /mcp/analysis.
Each tool receives tool_config for installation context access.
"""

from __future__ import annotations

import pydantic
from soliplex import config

from crazy_glue.analysis.context import AnalysisContext

# --- Result Models ---
# (See Result Models section above)

# --- Context Builder ---

def _build_context(tool_config: config.ToolConfig) -> AnalysisContext:
    """Build AnalysisContext from tool_config.

    The tool_config provides access to the installation configuration
    via the _installation_config attribute set during Soliplex load.
    """
    installation_config = tool_config._installation_config

    # Get analysis room's agent config for extra settings
    analysis_room = installation_config.room_configs.get("analysis")
    agent_config = analysis_room.agent_config if analysis_room else None

    return AnalysisContext(
        installation_config=installation_config,
        agent_config=agent_config,
    )

# --- Read Operations ---

def architect_list_rooms(tool_config: config.ToolConfig) -> RoomListResult:
    """List all registered rooms in the installation."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import list_rooms
    rooms = list_rooms(ctx)
    return RoomListResult(rooms=[RoomInfo(**r) for r in rooms], count=len(rooms))

def architect_list_managed(tool_config: config.ToolConfig) -> RoomListResult:
    """List rooms created by the architect (managed rooms)."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import list_managed_rooms
    rooms = list_managed_rooms(ctx)
    return RoomListResult(rooms=[RoomInfo(**r) for r in rooms], count=len(rooms))

def architect_inspect_room(
    room_id: str,
    tool_config: config.ToolConfig
) -> RoomDetailsResult:
    """Get detailed configuration for a specific room."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import inspect_room
    result = inspect_room(ctx, room_id)
    if "error" in result:
        return RoomDetailsResult(
            id=room_id, name="", description=None, welcome_message=None,
            suggestions=[], agent={}, config_path="", error=result["error"]
        )
    return RoomDetailsResult(**result)

# ... additional tool implementations

# --- Write Operations ---

def architect_create_room(
    room_id: str,
    description: str,
    tool_config: config.ToolConfig
) -> CreateRoomResult:
    """Create a new room with the given ID and description."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import create_room
    result = create_room(ctx, room_id, description)
    return CreateRoomResult(**result)

# ... additional write operations
```

### Files to Modify (soliplex)

#### `src/soliplex/config.py`

Add wrapper classes:

```python
# --- Architect Tool Wrappers ---

@dataclasses.dataclass
class ArchitectNoArgsMCPWrapper:
    """Wrapper for architect tools with no args (just tool_config)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self):
        return self._func(tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectWithRoomIdMCPWrapper:
    """Wrapper for architect tools that take room_id."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str):
        return self._func(room_id, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectWithNameMCPWrapper:
    """Wrapper for architect tools that take name."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, name: str):
        return self._func(name, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectWithQueryMCPWrapper:
    """Wrapper for architect tools that take query."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, query: str):
        return self._func(query, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectCreateRoomMCPWrapper:
    """Wrapper for create_room (room_id + description)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str, description: str):
        return self._func(room_id, description, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectEditRoomMCPWrapper:
    """Wrapper for edit_room (room_id + field + value)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str, field: str, value: str):
        return self._func(room_id, field, value, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectRoomToolMCPWrapper:
    """Wrapper for add/remove tool (room_id + tool_name)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str, tool_name: str):
        return self._func(room_id, tool_name, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectPromptMCPWrapper:
    """Wrapper for add_prompt (name + content)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, name: str, content: str):
        return self._func(name, content, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectMcpHttpMCPWrapper:
    """Wrapper for add_mcp_http (room_id + name + url)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str, name: str, url: str):
        return self._func(room_id, name, url, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectMcpStdioMCPWrapper:
    """Wrapper for add_mcp_stdio (room_id + name + command)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str, name: str, command: str):
        return self._func(room_id, name, command, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectRemoveMcpMCPWrapper:
    """Wrapper for remove_mcp (room_id + name)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str, name: str):
        return self._func(room_id, name, tool_config=self._tool_config)

@dataclasses.dataclass
class ArchitectToggleMcpServerMCPWrapper:
    """Wrapper for toggle_mcp_server (room_id + enabled)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, room_id: str, enabled: bool):
        return self._func(room_id, enabled, tool_config=self._tool_config)


# Update registry
MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME.update({
    # Read operations
    "crazy_glue.analysis.mcp_tools.architect_list_rooms": ArchitectNoArgsMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_list_managed": ArchitectNoArgsMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_inspect_room": ArchitectWithRoomIdMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_list_tools": ArchitectNoArgsMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_list_secrets": ArchitectNoArgsMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_check_secret": ArchitectWithNameMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_list_prompts": ArchitectNoArgsMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_show_prompt": ArchitectWithNameMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_find_entity": ArchitectWithQueryMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_show_reference": ArchitectWithNameMCPWrapper,
    # Write operations
    "crazy_glue.analysis.mcp_tools.architect_create_room": ArchitectCreateRoomMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_edit_room": ArchitectEditRoomMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_add_tool": ArchitectRoomToolMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_remove_tool": ArchitectRoomToolMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_add_prompt": ArchitectPromptMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_remove_prompt": ArchitectWithNameMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_add_mcp_http": ArchitectMcpHttpMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_add_mcp_stdio": ArchitectMcpStdioMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_remove_mcp": ArchitectRemoveMcpMCPWrapper,
    "crazy_glue.analysis.mcp_tools.architect_toggle_mcp_server": ArchitectToggleMcpServerMCPWrapper,
})
```

### Files to Modify (crazy-glue)

#### `rooms/analysis/room_config.yaml`

Add tools section:

```yaml
tools:
  # Read operations
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_list_rooms"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_list_managed"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_inspect_room"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_list_tools"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_list_secrets"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_check_secret"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_list_prompts"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_show_prompt"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_find_entity"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_show_reference"
    allow_mcp: true
  # Write operations
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_create_room"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_edit_room"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_add_tool"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_remove_tool"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_add_prompt"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_remove_prompt"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_add_mcp_http"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_add_mcp_stdio"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_remove_mcp"
    allow_mcp: true
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_toggle_mcp_server"
    allow_mcp: true
```

## Testing

### Unit Tests

```python
# tests/test_mcp_tools.py

import pytest
from unittest.mock import MagicMock

from crazy_glue.analysis import mcp_tools

@pytest.fixture
def mock_tool_config():
    """Create mock tool_config with installation context."""
    config = MagicMock()
    config._installation_config.room_configs = {
        "analysis": MagicMock(),
        "research": MagicMock(),
    }
    return config

def test_architect_list_rooms(mock_tool_config):
    result = mcp_tools.architect_list_rooms(mock_tool_config)
    assert isinstance(result, mcp_tools.RoomListResult)
    assert result.count >= 0

def test_architect_inspect_room_not_found(mock_tool_config):
    result = mcp_tools.architect_inspect_room("nonexistent", mock_tool_config)
    assert result.error is not None
```

### Integration Tests

```python
# tests/integration/test_mcp_integration.py

import pytest
from mcp import Client

@pytest.mark.integration
async def test_mcp_list_rooms():
    async with Client("http://localhost:8000/mcp/analysis") as client:
        tools = await client.list_tools()
        assert "architect_list_rooms" in [t.name for t in tools]

        result = await client.call_tool("architect_list_rooms", {})
        assert "rooms" in result
```

## Security Considerations

1. **Authentication Required**: All MCP endpoints require valid tokens
2. **Room-scoped Tokens**: Tokens are scoped to specific rooms via salt
3. **No Sensitive Data**: `list_secrets` returns names only, not values
4. **Write Audit**: Consider logging write operations for audit trail
5. **Rate Limiting**: Consider rate limiting expensive operations

## Future Enhancements

1. **generate_tool via MCP**: Could add with async job pattern (return job_id, poll for result)
2. **Webhook notifications**: Notify on room/tool changes
3. **Batch operations**: Create multiple rooms/tools in single call
4. **Dry-run mode**: Preview changes before applying
