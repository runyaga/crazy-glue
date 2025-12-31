# Agent-as-MCP Analysis: Why Option C (Vendor Soliplex) Is Not Viable

## Executive Summary

This document analyzes the feasibility of exposing the Analysis Room agent via MCP (Model Context Protocol). The conclusion is that vendoring Soliplex to add agent-as-MCP support is **not viable** due to fundamental protocol mismatches and high engineering cost with low business value.

**Recommendation:** Implement Option A - expose individual handlers as discrete MCP tools.

---

## Background

### What We Want
Expose the Analysis Room agent (System Architect) via MCP so external systems (e.g., Claude Desktop) can invoke its capabilities.

### What MCP Actually Is
MCP is a **stateless request-response protocol** for tools:
```
Client → Server: "call tool X with args Y"
Server → Client: "here's result Z"
```

No memory. No streaming. No conversation state.

---

## Current Architecture

### Soliplex Has Two Separate MCP Stacks

#### 1. MCP Client (pydantic-ai) - Agent CALLS external tools
```python
# agents.py:83-86
toolsets = [
    make_mcp_client_toolset(mctc)      # pydantic_ai.mcp.MCPServerStdio
    for mctc in mcp_client_configs     # pydantic_ai.mcp.MCPServerStreamableHTTP
]
```

#### 2. MCP Server (fastmcp) - Room EXPOSES tools to external callers
```python
# mcp_server.py:78-88
mcp = fmcp_server.FastMCP(
    key,
    tools=room_mcp_tools(room_config),  # Tool.from_function() wrappers
    auth=...,
)
```

**These share nothing.** Different libraries. Different purposes. No bridge.

### What pydantic-ai Provides for MCP

| Class | Purpose | Direction |
|-------|---------|-----------|
| `MCPServerStdio` | Connect to local MCP server via subprocess | Agent → External |
| `MCPServerStreamableHTTP` | Connect to remote MCP server via HTTP | Agent → External |

**What pydantic-ai does NOT provide:**
- Any way to expose an `AbstractAgent` as an MCP server
- `ExposeAgentAsMCP` or similar - **does not exist**

---

## Analysis Agent Architecture

### How It Works
The Analysis Agent (`analysis_factory.py`) is a conversational agent with:

1. **Message history extraction** - `_extract_prompt(message_history)`
2. **Confirmation flows** - checks for "y/yes/proceed" responses
3. **Pending state** - `pending_tool.json`, `pending_plan.json`
4. **Streaming output** - `async for event in self._yield_text(...)`
5. **Multi-step planning** - `is_complex_request()` triggers LLM planner
6. **Clarifying questions** - `if plan.clarifications: ...`

### The Protocol Mismatch

| Agent Feature | MCP Support |
|---------------|-------------|
| Conversation history | None - each call is isolated |
| Plan confirmation (`y/yes/proceed`) | None - no memory between calls |
| Streaming output | None - single response only |
| Clarifying questions | None - can't do back-and-forth |
| Pending state | None - client can't see server state |
| Context accumulation | None - forgets everything |

---

## Option C: Vendor Soliplex - Detailed Breakdown

### Layer 1: Understand pydantic-ai's Agent Protocol
```python
# The agent implements AbstractAgent with streaming:
async def run_stream_events(...) -> AsyncIterator[AgentStreamEvent]:
    yield PartStartEvent(...)   # streaming chunk
    yield PartEndEvent(...)     # chunk done
    yield AgentRunResultEvent(...)  # final result
```

This is a **streaming async iterator** protocol. Not a function call.

### Layer 2: Understand fastmcp's Tool Protocol
```python
# fastmcp expects simple callables:
fmcp_tools.Tool.from_function(
    tool_config.tool,  # Must be a callable
    name=tool_config.tool_id,
)
```

This expects a **synchronous or simple async function**. Not a streaming iterator.

### Layer 3: Build an Adapter (Doesn't Exist)

Would require creating something like:
```python
class AgentToMCPAdapter:
    def __init__(self, agent: AbstractAgent):
        self.agent = agent
        self.sessions = {}  # State management nightmare

    async def __call__(self, message: str, session_id: str = None) -> str:
        # Problem 1: Where does message_history come from?
        # Problem 2: How do we maintain session state across calls?
        # Problem 3: How do we handle streaming → single response?
        # Problem 4: How do we handle confirmation flows?

        result_parts = []
        async for event in self.agent.run_stream_events(
            message_history=???  # MCP doesn't provide this
        ):
            if isinstance(event, PartStartEvent):
                result_parts.append(event.part.content)

        return "".join(result_parts)  # Lost all streaming
```

### Layer 4: Handle State (Unsolvable)

The Analysis Agent confirmation flow:
```python
if user_prompt.lower().strip() in ("y", "yes", "proceed"):
    plan_data = ctx.load_pending_plan()  # Loads from filesystem
    if plan_data:
        async for event in execute_plan(ctx, plan):
            yield event
```

For this to work via MCP:
1. Session ID in MCP calls (MCP has no standard for this)
2. Server-side session storage (adds complexity)
3. Client to send session ID with each call (clients don't do this)

### Layer 5: Vendor and Maintain Fork

```
soliplex (upstream)
    ↓ fork
soliplex-vendor (your fork)
    ↓ add
    - agent_mcp_adapter.py
    - Modified mcp_server.py
    - Modified installation.py
    - Session management
    - New auth model
```

Every upstream update = manual merge + conflict resolution.

---

## Demo Reality Check

### What Would Actually Happen

**Scenario: Customer connects Claude Desktop to Agent-via-MCP**

```
Customer: "Create a calculator room and add a compute tool"

What the native agent does:
1. "Planning your request..."
2. Shows plan with 2 steps
3. Asks "Proceed? [y/n]"
4. Customer says "y"
5. Executes step 1, streams progress
6. Executes step 2, streams progress
7. "Plan complete"

What MCP would do:
1. Single response: "Planning your request... [plan]... Proceed? [y/n]"
2. ...nothing. MCP call is done.
3. Customer tries again: "y"
4. Agent has NO MEMORY of the plan. Returns "Unknown command: y"
```

**This is embarrassing in a demo.**

---

## Business Assessment

| Factor | Assessment |
|--------|------------|
| Demo impressiveness | Worse than native UI - looks clunky, unresponsive |
| Feature completeness | Missing 60%+ of agent capabilities |
| Customer problem solved | Unclear if "agent via MCP" is a real ask |
| Differentiation | Every competitor could do the same (badly) |
| Engineering cost | High - vendor fork + adapter + state management |
| Maintenance burden | Ongoing merge conflicts with upstream |

### What Actually Sells
- Live streaming output showing agent "thinking"
- Interactive confirmation of risky operations
- Multi-step plans with progress indicators
- Conversational refinement

**All of these require conversation state. MCP doesn't have it.**

---

## Verdict: Option C Is Not Viable

| Dimension | Rating |
|-----------|--------|
| pydantic-ai support | None - no agent-as-server capability |
| fastmcp support | Tool-oriented only - no agent concept |
| Protocol fit | Streaming iterator ≠ request/response |
| State management | Requires inventing session protocol |
| Engineering effort | Build adapter + state + vendor fork |
| Maintenance burden | Ongoing merge conflicts |
| End result | Crippled agent that looks broken in demos |

---

## Recommended Alternative: Option A

Expose individual Analysis Room handlers as discrete MCP tools.

### Why This Works
1. Handlers are already well-structured discrete operations
2. Each handler has clear inputs/outputs
3. Stateless by design - no session needed
4. Uses existing Soliplex `allow_mcp` infrastructure
5. No vendoring required

### Example Tools
- `list_rooms()` - List all configured rooms
- `create_room(room_id, description)` - Create a new room
- `add_tool(room_id, tool_name)` - Add tool to room
- `inspect_room(room_id)` - Get room details

### What You Lose (Acceptable)
- Multi-step planning (user sequences calls manually)
- Confirmation flows (immediate execution)
- Streaming progress (single response)

### What You Gain
- Working MCP integration
- Clean, professional demos
- No maintenance burden
- Honest feature representation

---

## Conclusion

Option C (vendor Soliplex for agent-as-MCP) is an engineering black hole with negative business value. The fundamental mismatch between MCP's stateless tool protocol and the Analysis Agent's conversational nature cannot be bridged without inventing new protocols.

Option A (expose handlers as tools) is the pragmatic choice that delivers real value with minimal effort.
