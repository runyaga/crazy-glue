# Soliplex Integration Lessons Learned

This document captures patterns and insights for integrating with Soliplex, particularly for exposing tools via MCP. Updated after each milestone.

---

## Core Concepts

### Tool Exposure via MCP

Soliplex exposes tools at `/mcp/{room_id}` when:
1. Room has `allow_mcp: true` in config
2. Tool has `allow_mcp: true` in its entry
3. Tool does NOT require `FASTAPI_CONTEXT`

### Key Files

| File | Purpose |
|------|---------|
| `src/soliplex/config.py` | ToolConfig, wrappers, `MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME` |
| `src/soliplex/mcp_server.py` | `mcp_tool()`, `room_mcp_tools()`, `setup_mcp_for_rooms()` |
| `src/soliplex/mcp_auth.py` | `FastMCPTokenProvider` for authentication |

---

## Milestone 1 Lessons

*To be updated after completing Milestone 1*

### Pattern: No-Args Tool with tool_config

```python
# Tool function signature
def my_tool(tool_config: config.ToolConfig) -> ResultModel:
    """Docstring becomes MCP tool description."""
    # Access installation via tool_config._installation_config
    pass

# Wrapper class
@dataclasses.dataclass
class NoArgsMCPWrapper:
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self):
        return self._func(tool_config=self._tool_config)

# Registration
MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME["module.path.my_tool"] = NoArgsMCPWrapper
```

### Key Insight: tool_config._installation_config

The `ToolConfig` object has `_installation_config` set during Soliplex load. This provides access to:
- `room_configs` - All room configurations
- `get_secret(name)` - Secret resolution
- `get_environment(name)` - Environment variables

### Gotcha: Function Signature Inspection

Soliplex inspects function signatures to determine `tool_requires`:
- Has `ctx` param → `FASTAPI_CONTEXT` (NOT MCP-compatible)
- Has `tool_config` param → `TOOL_CONFIG` (needs wrapper)
- Neither → `BARE` (can be exposed directly)

---

## Milestone 2 Lessons

*To be updated after completing Milestone 2*

### Pattern: Single Parameter Tools

```python
def my_tool(param: str, tool_config: config.ToolConfig) -> ResultModel:
    pass

@dataclasses.dataclass
class WithParamMCPWrapper:
    _func: abc.Callable[..., typing.Any]
    _tool_config: ToolConfig

    def __call__(self, param: str):
        return self._func(param, tool_config=self._tool_config)
```

### Key Insight: Docstring → Description

The tool's docstring becomes the MCP tool description. Keep it concise and actionable.

---

## Milestone 3 Lessons

*To be updated after completing Milestone 3*

### Pattern: Multi-Parameter Tools

Each unique signature needs its own wrapper class.

### Key Insight: Write Operations

Write operations should return a `StatusResult` with:
- `status`: "success" or "error"
- `message`: Human-readable description
- `error`: Error details if failed

---

## Milestone 4 Lessons

*To be updated after completing Milestone 4*

---

## Quick Reference

### Adding a New MCP Tool

1. **Create tool function** (crazy-glue)
   ```python
   def architect_my_tool(param: str, tool_config: config.ToolConfig) -> MyResult:
       """Tool description."""
       ctx = _build_context(tool_config)
       # Implementation
   ```

2. **Create/reuse wrapper** (soliplex)
   ```python
   @dataclasses.dataclass
   class MyToolWrapper:
       _func: abc.Callable[..., typing.Any]
       _tool_config: ToolConfig

       def __call__(self, param: str):
           return self._func(param, tool_config=self._tool_config)
   ```

3. **Register wrapper** (soliplex)
   ```python
   MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME["crazy_glue.analysis.mcp_tools.architect_my_tool"] = MyToolWrapper
   ```

4. **Add to room config** (crazy-glue)
   ```yaml
   tools:
     - tool_name: "crazy_glue.analysis.mcp_tools.architect_my_tool"
       allow_mcp: true
   ```

### Testing MCP Tools

```bash
# Generate token
TOKEN=$(soliplex-cli generate-mcp-token --room analysis)

# List tools
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Call tool
curl -s http://localhost:8765/mcp/analysis \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc":"2.0",
    "method":"tools/call",
    "params":{"name":"architect_my_tool","arguments":{"param":"value"}},
    "id":1
  }'
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Tool not appearing in list | Check `allow_mcp: true` on both room and tool |
| "Tool requires context" | Tool has `ctx` param - not MCP compatible |
| "Wrapper not found" | Register in `MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME` |
| Import error | Ensure tool module is installed/importable |

---

## Architecture Diagrams

### MCP Request Flow

```
Client Request
    │
    ▼
FastMCP Server (fastmcp library)
    │
    ▼
mcp_tool() in mcp_server.py
    │
    ├─ Check allow_mcp
    ├─ Check tool_requires != FASTAPI_CONTEXT
    │
    ▼
Wrapper Lookup (MCP_TOOL_CONFIG_WRAPPERS_BY_TOOL_NAME)
    │
    ▼
Wrapper.__call__() - injects tool_config
    │
    ▼
Tool Function - does the work
    │
    ▼
Pydantic Result Model
    │
    ▼
JSON Response to Client
```

### Context Building

```
tool_config (from Soliplex)
    │
    ├─ _installation_config
    │       │
    │       ├─ room_configs
    │       ├─ get_secret()
    │       └─ get_environment()
    │
    ▼
_build_context(tool_config)
    │
    ▼
AnalysisContext
    │
    ├─ installation_config
    ├─ agent_config
    ├─ project_root
    └─ storage paths (db/*)
```
