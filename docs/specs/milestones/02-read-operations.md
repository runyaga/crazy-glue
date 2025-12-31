# Milestone 2: Read Operations

## Prerequisites

- [Milestone 1](./01-single-tool-poc.md) complete (all gates passed)

## Objective

Implement remaining 9 read-only MCP tools using the pattern validated in Milestone 1.

## Tools to Implement

| Tool | Parameters | Wrapper Type |
|------|------------|--------------|
| `architect_list_managed` | none | `ArchitectNoArgsMCPWrapper` |
| `architect_inspect_room` | `room_id: str` | `ArchitectWithRoomIdMCPWrapper` |
| `architect_list_tools` | none | `ArchitectNoArgsMCPWrapper` |
| `architect_list_secrets` | none | `ArchitectNoArgsMCPWrapper` |
| `architect_check_secret` | `name: str` | `ArchitectWithNameMCPWrapper` |
| `architect_list_prompts` | none | `ArchitectNoArgsMCPWrapper` |
| `architect_show_prompt` | `name: str` | `ArchitectWithNameMCPWrapper` |
| `architect_find_entity` | `query: str` | `ArchitectWithQueryMCPWrapper` |
| `architect_show_reference` | `name: str` | `ArchitectWithNameMCPWrapper` |

## Deliverables

### 1. Additional Wrapper Classes (soliplex)

**File:** `src/soliplex/config.py`

```python
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
```

### 2. Tool Functions (crazy-glue)

**File:** `src/crazy_glue/analysis/mcp_tools.py` - add to existing file

```python
# Additional result models
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


# Tool functions
def architect_list_managed(tool_config: config.ToolConfig) -> RoomListResult:
    """List rooms created by the architect (managed rooms)."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import list_managed_rooms
    rooms = list_managed_rooms(ctx)
    return RoomListResult(rooms=[RoomInfo(**r) for r in rooms], count=len(rooms))


def architect_inspect_room(room_id: str, tool_config: config.ToolConfig) -> RoomDetailsResult:
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


def architect_list_tools(tool_config: config.ToolConfig) -> ToolListResult:
    """List available tool templates that can be added to rooms."""
    from crazy_glue.analysis.tools import list_available_tools
    tools = list_available_tools()
    return ToolListResult(tools=[ToolInfo(**t) for t in tools])


def architect_list_secrets(tool_config: config.ToolConfig) -> SecretListResult:
    """List configured secrets (names only, not values)."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import list_secrets
    secrets = list_secrets(ctx)
    return SecretListResult(secrets=[SecretInfo(**s) for s in secrets])


def architect_check_secret(name: str, tool_config: config.ToolConfig) -> SecretCheckResult:
    """Check if a specific secret is configured and resolved."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import check_secret
    result = check_secret(ctx, name)
    return SecretCheckResult(**result)


def architect_list_prompts(tool_config: config.ToolConfig) -> PromptListResult:
    """List saved prompts in the prompt library."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import list_prompts
    prompts = list_prompts(ctx)
    return PromptListResult(prompts=[PromptInfo(**p) for p in prompts])


def architect_show_prompt(name: str, tool_config: config.ToolConfig) -> PromptDetailResult:
    """Get the content of a specific prompt."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import get_prompt
    prompt = get_prompt(ctx, name)
    if prompt is None:
        return PromptDetailResult(name=name, content="", error=f"Prompt '{name}' not found")
    return PromptDetailResult(
        name=prompt.get("name", name),
        content=prompt.get("content", ""),
        created=prompt.get("created"),
    )


def architect_find_entity(query: str, tool_config: config.ToolConfig) -> SearchResult:
    """Search the knowledge graph for entities matching the query."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import query_graph
    results = query_graph(ctx, query)
    if results and isinstance(results, list) and results and "error" in results[0]:
        return SearchResult(query=query, results=[], count=0)
    return SearchResult(
        query=query,
        results=[EntityInfo(**r) for r in (results or [])],
        count=len(results or [])
    )


def architect_show_reference(name: str, tool_config: config.ToolConfig) -> ReferenceResult:
    """Get source code for a reference implementation (joker, faux, brainstorm)."""
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import read_reference_implementation
    source = read_reference_implementation(ctx, name)
    if source.startswith("Unknown reference:") or source.startswith("Error"):
        return ReferenceResult(name=name, source="", error=source)
    return ReferenceResult(name=name, source=source)
```

### 3. Wrapper Registry Updates (soliplex)

Add to `MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME`:

```python
# Read operations (Milestone 2)
"crazy_glue.analysis.mcp_tools.architect_list_managed": ArchitectNoArgsMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_inspect_room": ArchitectWithRoomIdMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_list_tools": ArchitectNoArgsMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_list_secrets": ArchitectNoArgsMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_check_secret": ArchitectWithNameMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_list_prompts": ArchitectNoArgsMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_show_prompt": ArchitectWithNameMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_find_entity": ArchitectWithQueryMCPWrapper,
"crazy_glue.analysis.mcp_tools.architect_show_reference": ArchitectWithNameMCPWrapper,
```

### 4. Room Config Updates (crazy-glue)

**File:** `rooms/analysis/room_config.yaml`

```yaml
tools:
  # Milestone 1
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_list_rooms"
    allow_mcp: true
  # Milestone 2 - Read operations
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
```

## Gating Criteria

### Gate 1: All Tools Import
```bash
python -c "
from crazy_glue.analysis.mcp_tools import (
    architect_list_rooms, architect_list_managed, architect_inspect_room,
    architect_list_tools, architect_list_secrets, architect_check_secret,
    architect_list_prompts, architect_show_prompt, architect_find_entity,
    architect_show_reference
)
print('OK')
"
```
**Expected:** `OK`

### Gate 2: All 10 Tools in MCP List
```bash
TOKEN=$(soliplex-cli generate-mcp-token --room analysis)
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}' \
  | jq '[.result.tools[].name | select(startswith("architect_"))] | length'
```
**Expected:** `10`

### Gate 3: Each Tool Returns Valid Response

Test each tool:
```bash
# architect_inspect_room
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "method":"tools/call",
    "params":{"name":"architect_inspect_room","arguments":{"room_id":"analysis"}},
    "id":1
  }' | jq '.result.id'
```
**Expected:** `"analysis"`

### Gate 4: Ruff Passes
```bash
ruff check src/crazy_glue/analysis/mcp_tools.py
ruff check tests/test_mcp_tools.py
```
**Expected:** No errors or warnings

### Gate 5: Unit Tests Pass
```bash
pytest tests/test_mcp_tools.py -v
```
**Expected:** All read operation tests pass

## Success Criteria

| Gate | Description | Pass/Fail |
|------|-------------|-----------|
| 1 | All tool functions import | |
| 2 | 10 tools visible in MCP list | |
| 3 | Each tool returns valid data | |
| 4 | Ruff check passes | |
| 5 | Unit tests pass | |
| 6 | Lessons learned updated | |

### Gate 6: Update Lessons Learned

Update `docs/specs/soliplex-lessons.md` with:
- Patterns for single-parameter tools (room_id, name, query)
- Wrapper reuse strategies
- Docstring → MCP description behavior
- Any new gotchas encountered

## Next Milestone

Once all gates pass → [Milestone 3: Write Operations](./03-write-operations.md)
