# Milestone 1: Single Tool Proof-of-Concept

## Objective

Implement `architect_list_rooms` end-to-end to validate the full integration pattern before bulk implementation.

## Why This Tool?

- Simplest signature: no parameters (just `tool_config`)
- Read-only: no side effects to debug
- Uses existing `list_rooms()` operation
- Quick to verify: returns observable data

## Deliverables

### 1. MCP Tools Module (crazy-glue)

**File:** `src/crazy_glue/analysis/mcp_tools.py`

```python
"""MCP-compatible tools for Analysis Room operations."""
from __future__ import annotations

import pydantic
from soliplex import config

from crazy_glue.analysis.context import AnalysisContext


class RoomInfo(pydantic.BaseModel):
    """Room summary for list results."""
    id: str
    name: str
    description: str
    agent_kind: str
    factory: str | None = None


class RoomListResult(pydantic.BaseModel):
    """Result from listing rooms."""
    rooms: list[RoomInfo]
    count: int


def _build_context(tool_config: config.ToolConfig) -> AnalysisContext:
    """Build AnalysisContext from tool_config.

    The tool_config provides access to the installation configuration
    via the _installation_config attribute set during Soliplex load.
    """
    installation_config = tool_config._installation_config
    analysis_room = installation_config.room_configs.get("analysis")
    agent_config = analysis_room.agent_config if analysis_room else None

    return AnalysisContext(
        installation_config=installation_config,
        agent_config=agent_config,
    )


def architect_list_rooms(tool_config: config.ToolConfig) -> RoomListResult:
    """List all registered rooms in the installation.

    Returns a list of room summaries including ID, name, description,
    and agent type information.
    """
    ctx = _build_context(tool_config)
    from crazy_glue.analysis.tools import list_rooms
    rooms = list_rooms(ctx)
    return RoomListResult(
        rooms=[RoomInfo(**r) for r in rooms],
        count=len(rooms)
    )
```

### 2. Wrapper Class (soliplex)

**File:** `src/soliplex/config.py` - add near existing wrappers (~line 750)

```python
@dataclasses.dataclass
class ArchitectNoArgsMCPWrapper:
    """Wrapper for architect tools with no args (just tool_config)."""
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self):
        return self._func(tool_config=self._tool_config)
```

**Add to `MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME` dict:**

```python
MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME["crazy_glue.analysis.mcp_tools.architect_list_rooms"] = ArchitectNoArgsMCPWrapper
```

### 3. Room Configuration (crazy-glue)

**File:** `rooms/analysis/room_config.yaml` - add after `allow_mcp: true`

```yaml
tools:
  - tool_name: "crazy_glue.analysis.mcp_tools.architect_list_rooms"
    allow_mcp: true
```

## Gating Criteria

### Gate 1: Module Loads
```bash
# From crazy-glue directory
python -c "from crazy_glue.analysis.mcp_tools import architect_list_rooms; print('OK')"
```
**Expected:** `OK` (no import errors)

### Gate 2: Soliplex Starts
```bash
# From crazy-glue directory
soliplex serve --port 8765
```
**Expected:** Server starts without errors, logs show analysis room loaded

### Gate 3: Tool Appears in MCP List
```bash
# Generate token
TOKEN=$(soliplex-cli generate-mcp-token --room analysis)

# List tools via MCP protocol
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}' | jq '.result.tools[].name'
```
**Expected:** Output includes `architect_list_rooms`

### Gate 4: Tool Returns Data
```bash
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "method":"tools/call",
    "params":{"name":"architect_list_rooms","arguments":{}},
    "id":2
  }' | jq '.result'
```
**Expected:** JSON with `rooms` array and `count` field matching number of rooms

### Gate 5: Ruff Passes
```bash
ruff check src/crazy_glue/analysis/mcp_tools.py
ruff check tests/test_mcp_tools.py
```
**Expected:** No errors or warnings

### Gate 6: Unit Test Passes
```bash
pytest tests/test_mcp_tools.py::test_architect_list_rooms -v
```
**Expected:** Test passes

## Test File

**File:** `tests/test_mcp_tools.py`

```python
"""Tests for Analysis Room MCP tools."""
import pytest
from unittest.mock import MagicMock, patch


def test_architect_list_rooms():
    """Test architect_list_rooms returns room list."""
    from crazy_glue.analysis.mcp_tools import architect_list_rooms, RoomListResult

    # Mock tool_config with installation context
    mock_config = MagicMock()
    mock_config._installation_config.room_configs = {
        "analysis": MagicMock(
            agent_config=MagicMock(extra_config={})
        ),
        "research": MagicMock(),
    }

    # Mock list_rooms to return test data
    with patch("crazy_glue.analysis.tools.list_rooms") as mock_list:
        mock_list.return_value = [
            {"id": "analysis", "name": "Analysis", "description": "Test", "agent_kind": "factory"},
            {"id": "research", "name": "Research", "description": "Test2", "agent_kind": "default"},
        ]

        result = architect_list_rooms(mock_config)

    assert isinstance(result, RoomListResult)
    assert result.count == 2
    assert len(result.rooms) == 2
    assert result.rooms[0].id == "analysis"
```

## Success Criteria

All 7 gates must pass before proceeding to Milestone 2.

| Gate | Description | Pass/Fail |
|------|-------------|-----------|
| 1 | Module imports without error | |
| 2 | Soliplex starts successfully | |
| 3 | Tool appears in MCP tool list | |
| 4 | Tool returns expected data | |
| 5 | Ruff check passes | |
| 6 | Unit test passes | |
| 7 | Lessons learned updated | |

### Gate 7: Update Lessons Learned

Update `docs/specs/soliplex-lessons.md` with:
- Pattern used for no-args tool with tool_config
- How tool_config._installation_config provides context
- Any gotchas encountered during implementation

## Files Modified

| Repo | File | Action |
|------|------|--------|
| crazy-glue | `src/crazy_glue/analysis/mcp_tools.py` | Create |
| crazy-glue | `rooms/analysis/room_config.yaml` | Modify |
| crazy-glue | `tests/test_mcp_tools.py` | Create |
| soliplex | `src/soliplex/config.py` | Modify |

## Next Milestone

Once all gates pass → [Milestone 2: Read Operations](./02-read-operations.md)
