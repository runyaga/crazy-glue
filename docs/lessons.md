# Lessons Learned: Building Soliplex Factory Agents

## Factory Agent Pattern

### Basic Structure

Factory agents in soliplex use `kind: "factory"` in `room_config.yaml`:

```yaml
agent:
  kind: "factory"
  factory_name: "my_package.factories.my_factory.create_my_agent"
  with_agent_config: true
  extra_config:
    custom_setting: value
```

The factory function receives:
- `agent_config: FactoryAgentConfig` - contains `extra_config`, `_installation_config`
- `tool_configs: ToolConfigMap` - configured tools
- `mcp_client_toolset_configs: MCP_ClientToolsetConfigMap` - MCP toolsets

### Creating Models (Soliplex Way)

Don't use external model helpers. Use soliplex's configuration:

```python
from pydantic_ai.models import openai as openai_models
from pydantic_ai.providers import ollama as ollama_providers

def _get_model(self):
    installation_config = self.agent_config._installation_config
    provider_base_url = installation_config.get_environment("OLLAMA_BASE_URL")
    provider = ollama_providers.OllamaProvider(
        base_url=f"{provider_base_url}/v1",
    )
    return openai_models.OpenAIChatModel(
        model_name="gpt-oss:20b",
        provider=provider,
    )
```

### Streaming Events

Factory agents implement `run_stream_events()` returning `AsyncIterator[NativeEvent]`:

```python
async def run_stream_events(
    self,
    output_type: typing.Any = None,
    message_history: MessageHistory | None = None,
    deferred_tool_results: typing.Any = None,
    deps: ai_tools.AgentDepsT = None,
    **kwargs: typing.Any,
) -> abc.AsyncIterator[NativeEvent]:
    # Extract user prompt
    prompt = _extract_prompt(message_history)

    # Get emitter from deps
    emitter = getattr(deps, "agui_emitter", None) if deps else None

    # Yield pydantic-ai events
    yield ai_messages.PartStartEvent(index=0, part=think_part)
    yield ai_messages.PartDeltaEvent(index=0, delta=...)
    yield ai_messages.PartEndEvent(index=0, part=think_part)

    yield ai_run.AgentRunResultEvent(result=response)
```

---

## AG-UI Integration

### State Updates: Known Issues

**Problem**: The `compute_state_delta` function uses JSON Patch `replace` operations for all changes. Per RFC 6902, `replace` requires the path to already exist. If a field is new, it should use `add` instead.

**Symptom**:
```
NotRunning: ('Parser not in RUNNING state: RunStatus.ERROR: ', StateDeltaEvent(...))
```

**Workaround**: Use activities instead of state updates:

```python
# DON'T do this (causes delta issues):
emitter.update_state(my_state)

# DO this instead:
emitter.update_activity("my_activity", {
    "status": "running",
    "progress": 50,
    "data": "whatever you need",
}, activity_id)
```

### Activities vs State

**Activities** (`update_activity`):
- Free-form dict content
- Identified by `activity_id` (UUID)
- Good for showing workflow progress
- No delta computation issues

**State** (`update_state`):
- Pydantic model
- Supports snapshots and deltas
- Delta computation can fail if schema changes between updates
- Use only when you need structured state with proper delta tracking

### State Snapshot Timing Issue (Race Condition)

**Problem**: State snapshots emitted at end of `run_stream_events()` may be rejected by the parser.

**Root Cause**: The AG-UI adapter yields `RUN_FINISHED` in `after_stream()` immediately after the agent generator completes. The emitter events (STATE_SNAPSHOT) are in a separate async stream that's multiplexed with the agent stream. Due to concurrent execution, `RUN_FINISHED` can arrive at the parser BEFORE `STATE_SNAPSHOT`. Since the parser requires `run_status == RUNNING` to accept STATE_SNAPSHOT (parser.py:472-475), the event is rejected with `NotRunning`.

**Symptom**: State snapshot at end of workflow is silently dropped or causes error.

**Architecture**:
```
agent_stream:   RUN_STARTED → [events] → RUN_FINISHED
emitter_stream: STATE_SNAPSHOT, ACTIVITY_SNAPSHOT, ...
                    ↓
            multiplex_streams (no ordering guarantee)
                    ↓
                 parser (requires RUNNING state for state events)
```

**Workaround**: Don't emit state snapshots at the very end. Use activities for final quantitative data instead:

```python
# At the end - use activity with final data (NOT state)
emitter.update_activity("results", {
    "phase": "complete",
    "total_items": 100,
    "success_rate": 0.95,
    "winner": "PRO",
    "final_scores": {"pro": 24.5, "con": 22.0},
}, activity_id)
```

### Recommended Pattern: Activities Only

Due to both delta computation issues AND the timing race condition, use **activities exclusively** for factory agents:

```python
activity_id = str(uuid.uuid4())

# During workflow - use activities
emitter.update_activity("workflow", {"status": "step1", "progress": 25}, activity_id)
emitter.update_activity("workflow", {"status": "step2", "progress": 50}, activity_id)
emitter.update_activity("workflow", {"status": "step3", "progress": 75}, activity_id)

# At the end - activity with final quantitative data
emitter.update_activity("workflow", {
    "status": "complete",
    "total_items": 100,
    "success_rate": 0.95,
}, activity_id)
```

This avoids both delta computation issues AND the RUN_FINISHED race condition.

### Activity Pattern

```python
activity_id = str(uuid.uuid4())

# Update same activity throughout workflow
emitter.update_activity("workflow_name", {"status": "starting"}, activity_id)
# ... do work ...
emitter.update_activity("workflow_name", {"status": "processing", "step": 1}, activity_id)
# ... do work ...
emitter.update_activity("workflow_name", {"status": "complete", "result": "done"}, activity_id)
```

---

## Pydantic-AI Gotchas

### Agent Output Type

Use `output_type`, not `result_type`:

```python
# WRONG
agent = Agent(model, result_type=MyModel, ...)

# CORRECT
agent = Agent(model, output_type=MyModel, ...)
```

### Accessing Agent Output

```python
result = await agent.run(prompt)
output = result.output  # The actual output (string or structured type)
```

---

## Project Structure

```
crazy-glue/
├── installation.yaml      # Soliplex config
├── .env                   # OLLAMA_BASE_URL, etc.
├── rooms/
│   └── my_room/
│       └── room_config.yaml
├── src/crazy_glue/
│   ├── __init__.py
│   └── factories/
│       ├── __init__.py    # Export factory functions
│       └── my_factory.py
└── docs/
    └── lessons.md
```

### Room Config Template

```yaml
id: "my-room"
name: "My Room Name"
description: "What this room does"
welcome_message: |
  Welcome message shown to users.

suggestions:
  - "Example prompt 1"
  - "Example prompt 2"

agent:
  kind: "factory"
  factory_name: "crazy_glue.factories.my_factory.create_my_agent"
  with_agent_config: true
  extra_config:
    num_rounds: 3
```

---

## Running the Server

```bash
cd /path/to/crazy-glue
source .venv/bin/activate
soliplex-cli serve . --no-auth-mode --port 8001
```

### Debugging Tips

1. **Cached bytecode**: Restart server after code changes
2. **Check imports**: `python -c "from crazy_glue.factories import create_my_agent"`
3. **Server logs**: Check terminal for tracebacks
4. **TUI vs Web**: Some issues only appear in specific clients

---

## AG-UI Tool Call Visibility

### Problem: Tool Calls Don't Appear in AG-UI

When using pydantic-ai's `agent.run()`, tool calls execute internally without yielding AG-UI events. The AG-UI client never sees `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, or `TOOL_CALL_RESULT` events.

**Root Cause**: pydantic-ai's `agent.run()` is a blocking call that handles tool execution internally. Unlike `agent.iter()` which streams events, `run()` returns only the final result.

**Investigation Path**:
1. Checked `soliplex/src/soliplex/examples.py` - `FauxAgent` manually yields `ToolCallPart` events
2. Checked `soliplex/src/soliplex/agui/parser.py` - AG-UI expects specific event types (lines 392-449)

### Solution: Extract and Emit Tool Calls Post-Execution

After `agent.run()` completes, extract tool calls from the result's message history and emit `ToolCallPart` events:

```python
async def run_stream_events(self, ...) -> abc.AsyncIterator[NativeEvent]:
    # Run the agent (tools execute internally)
    result = await agent.run(prompt, deps=ctx)

    # Extract tool calls from result messages
    tool_calls_made = []
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if isinstance(part, ai_messages.ToolCallPart):
                    tool_calls_made.append(part.tool_name)

    # Yield tool call events for AG-UI visibility
    for tool_name in tool_calls_made:
        part_index += 1
        tc_part = ai_messages.ToolCallPart(tool_name=tool_name)
        yield ai_messages.PartStartEvent(index=part_index, part=tc_part)
        yield ai_messages.PartEndEvent(index=part_index, part=tc_part)
```

**Trade-off**: This shows tools after completion rather than real-time. For real-time streaming, you would need to use `agent.iter()` instead of `agent.run()`.

### AG-UI Event Types Reference

From `parser.py`, the AG-UI parser handles:
- `TOOL_CALL_START` (line 392) - Starts a tool call
- `TOOL_CALL_ARGS` (line 424) - Tool arguments as JSON delta
- `TOOL_CALL_END` (line 434) - Ends tool call
- `TOOL_CALL_RESULT` (line 449) - Tool execution result

The `FauxAgent` example shows the pattern:

```python
# From soliplex/src/soliplex/examples.py
for tool_name, tool_config in self.tool_configs.items():
    tc_part = ai_messages.ToolCallPart(tool_name)
    yield ai_messages.PartStartEvent(index=part_index, part=tc_part)
    await tool_config.tool(ctx)  # Actually run the tool
    yield ai_messages.PartEndEvent(index=part_index, part=tc_part)
    part_index += 1
```

---

## Accessing Live Soliplex Configuration

### The `_installation_config` Pattern

Factory agents receive `agent_config: FactoryAgentConfig` which has a private `_installation_config` attribute containing the **live** `InstallationConfig` object:

```python
@dataclasses.dataclass
class IntrospectiveContext:
    agent_config: config.FactoryAgentConfig

    @property
    def installation_config(self) -> config.InstallationConfig:
        """Get the live installation config."""
        return self.agent_config._installation_config
```

### What You Can Access

The `InstallationConfig` provides:

| Attribute | Type | Description |
|-----------|------|-------------|
| `room_configs` | `dict[str, RoomConfig]` | All loaded room configurations |
| `get_environment(key)` | Method | Get environment variable value |
| `environment` | `dict` | Raw environment dict |
| `secrets` | `list` | Configured secrets |
| `id` | `str` | Installation ID |
| `_config_path` | `Path` | Path to installation.yaml |
| `room_paths` | `list[Path]` | Room search paths |

### Example: Room Introspection Tool

```python
@agent.tool
def list_all_rooms(
    ctx: ai_tools.RunContext[IntrospectiveContext],
) -> list[dict[str, typing.Any]]:
    """List ALL rooms in the current soliplex installation."""
    installation = ctx.deps.installation_config
    rooms = []

    for room_id, room_config in installation.room_configs.items():
        agent_config = room_config.agent_config
        rooms.append({
            "id": room_id,
            "name": room_config.name,
            "description": room_config.description,
            "agent_kind": agent_config.kind,
            "factory_name": getattr(agent_config, "factory_name", None),
        })

    return sorted(rooms, key=lambda r: r["id"])
```

### Accessing Room Configuration Details

Each `RoomConfig` object provides:

```python
room = installation.room_configs[room_id]

# Basic info
room.id              # Room identifier
room.name            # Display name
room.description     # Description text
room.welcome_message # Welcome shown to users
room.suggestions     # List of example prompts

# Agent info
room.agent_config.kind        # "factory" or "prompt"
room.agent_config.factory_name  # For factory agents
room.agent_config.extra_config  # Custom config dict

# Features
room.tool_configs       # Configured tools
room.enable_attachments # File upload enabled
room._config_path       # Path to room_config.yaml
```

---

## Building Introspective Agents

### Design Principles

1. **Use real data**: Access live configuration, don't hardcode
2. **Be comprehensive**: Many small, focused tools > few large tools
3. **Self-awareness**: Include tools like `who_am_i` and `list_my_tools`
4. **Pattern knowledge**: Embed domain knowledge (patterns, diagrams)

### Tool Categories for Introspection

| Category | Tools | Data Source |
|----------|-------|-------------|
| Room Inspection | `list_all_rooms`, `inspect_room`, `get_room_suggestions` | `installation.room_configs` |
| Configuration | `get_installation_info`, `get_environment_variables`, `get_secrets_info` | `installation.*` |
| Factory Agents | `list_factory_agents`, `inspect_factory` | Room configs + `importlib` |
| Patterns | `explain_pattern`, `list_patterns`, `find_rooms_by_pattern` | Knowledge base dict |
| Diagrams | `generate_installation_diagram`, `generate_room_diagram` | Dynamic mermaid generation |
| System | `run_health_check`, `who_am_i`, `list_my_tools` | Mixed sources |

### Security Considerations

- Mask sensitive values (tokens, passwords) in environment tools
- Show secret names but never values
- Be careful with file path exposure

```python
@agent.tool
def get_environment_variables(ctx):
    for key in ["OLLAMA_BASE_URL", "LOGFIRE_TOKEN"]:
        value = installation.get_environment(key)
        if value and ("TOKEN" in key or "SECRET" in key):
            env[key] = value[:8] + "..."  # Mask sensitive values
        else:
            env[key] = value or "(not set)"
```

### Dynamic Diagram Generation

Generate mermaid diagrams from live data:

```python
@agent.tool
def generate_installation_diagram(ctx):
    installation = ctx.deps.installation_config
    rooms = list(installation.room_configs.keys())

    room_nodes = "\n        ".join([f'R{i}["{r}"]' for i, r in enumerate(rooms)])

    return f"""```mermaid
flowchart TB
    subgraph "Installation"
        subgraph Rooms
        {room_nodes}
        end
    end
```"""

---

## Composite & Human-in-the-Loop Patterns

### Chaining Patterns

Real-world agents often need to combine multiple patterns. A powerful combination is **Planning + Human-in-the-Loop + Parallelization + Reflection**:

1.  **Planning**: Decompose a vague goal into concrete steps.
2.  **Human-in-the-Loop**: Pause to get user approval or refinement of the plan.
3.  **Parallelization**: Execute the approved steps concurrently for speed.
4.  **Reflection**: Synthesize and polish the results.

### Managing State Across Turns

Soliplex agents are stateless between runs, but context is preserved in `message_history`. To implement multi-step workflows (like "propose plan" -> "get feedback" -> "execute"), you must reconstruct state from the conversation history.

**Strategy**:
1.  **Analyze History**: Check the last few messages to determine the current "Phase".
2.  **Phase: Planning**: If no plan exists, generate one.
3.  **Phase: Approval**: If a plan was just proposed, check the user's latest response.
    *   If "Approved" -> Transition to **Execution**.
    *   If "Change X" -> Modify plan and remain in **Approval**.
4.  **Phase: Execution**: If approved, run the heavy patterns (Parallel/Reflection).

```python
async def run(self, prompt, message_history, ...):
    # 1. Reconstruct state
    phase = determine_phase(message_history)
    
    if phase == "PLANNING":
        plan = await create_plan(prompt)
        return format_plan_for_approval(plan)
        
    elif phase == "APPROVAL":
        if is_approved(prompt):
            # Transition to execution
            return await execute_heavy_work(previous_plan)
        else:
            # Refine plan
            new_plan = await refine_plan(previous_plan, feedback=prompt)
            return format_plan_for_approval(new_plan)
```
```

---

## Future Improvements

1. Fix `compute_state_delta` to use `add` for new keys instead of `replace`
2. Make TUI parser more tolerant of delta operations on missing paths
3. Add state snapshot-only mode option for factory agents
4. Fix multiplex timing: ensure emitter events arrive before RUN_FINISHED
5. Allow parser to accept STATE_SNAPSHOT in FINISHED state (for final state)
6. Add real-time tool call streaming via `agent.iter()` support
7. Provide public accessor for `_installation_config` (currently private API)
