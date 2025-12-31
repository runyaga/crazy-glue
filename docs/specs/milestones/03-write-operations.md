# Milestone 3: Write Operations

## Prerequisites

- [Milestone 2](./02-read-operations.md) complete (all gates passed)

## Objective

Implement 10 write operation MCP tools that modify room configurations.

## Tools to Implement

| Tool | Parameters | Wrapper Type |
|------|------------|--------------|
| `architect_create_room` | `room_id`, `description` | `ArchitectCreateRoomMCPWrapper` |
| `architect_edit_room` | `room_id`, `field`, `value` | `ArchitectEditRoomMCPWrapper` |
| `architect_add_tool` | `room_id`, `tool_name` | `ArchitectRoomToolMCPWrapper` |
| `architect_remove_tool` | `room_id`, `tool_name` | `ArchitectRoomToolMCPWrapper` |
| `architect_add_prompt` | `name`, `content` | `ArchitectPromptMCPWrapper` |
| `architect_remove_prompt` | `name` | `ArchitectWithNameMCPWrapper` |
| `architect_add_mcp_http` | `room_id`, `name`, `url` | `ArchitectMcpHttpMCPWrapper` |
| `architect_add_mcp_stdio` | `room_id`, `name`, `command` | `ArchitectMcpStdioMCPWrapper` |
| `architect_remove_mcp` | `room_id`, `name` | `ArchitectRemoveMcpMCPWrapper` |
| `architect_toggle_mcp_server` | `room_id`, `enabled` | `ArchitectToggleMcpServerMCPWrapper` |

## Deliverables

### 1. Additional Wrapper Classes (soliplex)

**File:** `src/soliplex/config.py`

```python
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
```

### 2. Result Models (crazy-glue)

**File:** `src/crazy_glue/analysis/mcp_tools.py` - add models

```python
class CreateRoomResult(pydantic.BaseModel):
    """Result from creating a room."""
    room_id: str
    config_path: str
    status: str
    error: str | None = None


class StatusResult(pydantic.BaseModel):
    """Generic status result for write operations."""
    status: str  # "success" | "error"
    message: str
    error: str | None = None
```

### 3. Tool Functions (crazy-glue)

**File:** `src/crazy_glue/analysis/mcp_tools.py` - add functions

```python
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


def architect_edit_room(
    room_id: str,
    field: str,
    value: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Edit a room field (description, prompt, model, welcome)."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import edit_room
    result = edit_room(ctx, room_id, field, value)
    return StatusResult(**result)


def architect_add_tool(
    room_id: str,
    tool_name: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Add a predefined tool to a room."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import manage_tool
    result = manage_tool(ctx, room_id, tool_name, action="add")
    return StatusResult(**result)


def architect_remove_tool(
    room_id: str,
    tool_name: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Remove a tool from a room."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import manage_tool
    result = manage_tool(ctx, room_id, tool_name, action="remove")
    return StatusResult(**result)


def architect_add_prompt(
    name: str,
    content: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Save a new prompt to the prompt library."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import add_prompt
    result = add_prompt(ctx, name, content)
    return StatusResult(**result)


def architect_remove_prompt(
    name: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Delete a prompt from the library."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import remove_prompt
    result = remove_prompt(ctx, name)
    return StatusResult(**result)


def architect_add_mcp_http(
    room_id: str,
    name: str,
    url: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Add an HTTP MCP toolset to a room."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import manage_mcp
    result = manage_mcp(ctx, room_id, name, action="add", kind="http", url=url)
    return StatusResult(**result)


def architect_add_mcp_stdio(
    room_id: str,
    name: str,
    command: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Add a stdio MCP toolset to a room."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import manage_mcp
    result = manage_mcp(ctx, room_id, name, action="add", kind="stdio", command=command)
    return StatusResult(**result)


def architect_remove_mcp(
    room_id: str,
    name: str,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Remove an MCP toolset from a room."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import manage_mcp
    result = manage_mcp(ctx, room_id, name, action="remove")
    return StatusResult(**result)


def architect_toggle_mcp_server(
    room_id: str,
    enabled: bool,
    tool_config: config.ToolConfig
) -> StatusResult:
    """Enable or disable MCP server mode for a room."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import toggle_mcp_server
    result = toggle_mcp_server(ctx, room_id, enabled)
    return StatusResult(**result)
```

### 4. Wrapper Registry Updates (soliplex)

Add to `MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME`:

```python
# Write operations (Milestone 3)
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
```

### 5. Room Config Updates (crazy-glue)

**File:** `rooms/analysis/room_config.yaml` - add to tools section

```yaml
  # Milestone 3 - Write operations
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

## Gating Criteria

### Gate 1: All 20 Tools in MCP List
```bash
TOKEN=$(soliplex-cli generate-mcp-token --room analysis)
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}' \
  | jq '[.result.tools[].name | select(startswith("architect_"))] | length'
```
**Expected:** `20`

### Gate 2: Create Room Works
```bash
# Create test room
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "method":"tools/call",
    "params":{
      "name":"architect_create_room",
      "arguments":{"room_id":"test-mcp-room","description":"Test room created via MCP"}
    },
    "id":1
  }' | jq '.result.status'
```
**Expected:** `"success"`

### Gate 3: Verify Room Exists
```bash
# Check room appears in list
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "method":"tools/call",
    "params":{"name":"architect_list_rooms","arguments":{}},
    "id":1
  }' | jq '.result.rooms[].id' | grep test-mcp-room
```
**Expected:** `"test-mcp-room"`

### Gate 4: Edit Room Works
```bash
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "method":"tools/call",
    "params":{
      "name":"architect_edit_room",
      "arguments":{"room_id":"test-mcp-room","field":"description","value":"Updated via MCP"}
    },
    "id":1
  }' | jq '.result.status'
```
**Expected:** `"success"`

### Gate 5: Cleanup Test Room
```bash
# Manual cleanup - remove test room directory
rm -rf rooms/test-mcp-room
```

### Gate 6: Ruff Passes
```bash
ruff check src/crazy_glue/analysis/mcp_tools.py
ruff check tests/test_mcp_tools.py
```
**Expected:** No errors or warnings

### Gate 7: Unit Tests Pass
```bash
pytest tests/test_mcp_tools.py -v
```
**Expected:** All tests pass

## Success Criteria

| Gate | Description | Pass/Fail |
|------|-------------|-----------|
| 1 | 20 tools visible in MCP list | |
| 2 | Create room returns success | |
| 3 | Created room appears in list | |
| 4 | Edit room returns success | |
| 5 | Test room cleaned up | |
| 6 | Ruff check passes | |
| 7 | Unit tests pass | |
| 8 | Lessons learned updated | |

### Gate 8: Update Lessons Learned

Update `docs/specs/soliplex-lessons.md` with:
- Multi-parameter wrapper patterns
- Write operation result conventions (StatusResult)
- Filesystem side effects handling
- Error handling patterns
- Any security considerations discovered

## Security Notes

Write operations modify the filesystem. Consider:
- Rate limiting for write operations
- Audit logging for changes
- Role-based access (future enhancement)

## Next Milestone

Once all gates pass → [Milestone 4: Documentation & Tests](./04-docs-and-tests.md)
