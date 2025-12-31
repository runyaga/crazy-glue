# Lessons Learned: Building Soliplex Factory Agents

> [!CAUTION]
> **THIS DOCUMENTATION IS FROM A VIBE-CODING EXPERIMENT**
>
> The code in this repository is messy, unmaintainable, and should not be used as a reference.
> These lessons are observations from the chaos, not best practices.

---

## The Vibe Coding Experiment

### What We Tried

This repository was an experiment to answer: **"How much of a mess will be made if just vibe coding with agentic patterns?"**

The approach:
- Take patterns from [agentic-patterns-book](https://github.com/runyaga/agentic-patterns-book)
- Combine them into "rooms" in Soliplex
- Use rapid AI-assisted coding without proper planning
- See what happens

### What We Learned

#### The Bad

1. **Code becomes unmaintainable fast**
   - The introspective factory grew to 2000+ lines through incremental additions
   - Duplicate implementations accumulated (same tool implemented 3 different ways)
   - Architecture decisions made on-the-fly don't compose well

2. **Documentation drifts immediately**
   - What the code does != what docs say it does
   - Comments become lies within hours
   - Examples stop working as code evolves

3. **Patterns get corrupted**
   - Clean pattern implementations get "enhanced" with edge cases
   - Workarounds for bugs become permanent features
   - Original design intent is lost

4. **Technical debt compounds exponentially**
   - Each quick fix creates new edge cases
   - Fixing one thing breaks three others
   - Eventually you can't change anything safely

#### The Good

1. **Great for prototyping**
   - Rapidly explore if an idea is even feasible
   - Discover unknown unknowns before investing in clean code
   - Generate throwaway code to understand problem space

2. **Fast learning loop**
   - Quick iterations reveal what works and what doesn't
   - Failure is cheap and informative
   - Real patterns emerge from experimentation

3. **Captures tacit knowledge**
   - Lessons.md accumulated useful gotchas
   - Edge cases documented as they were hit
   - Workarounds preserved for future reference

### Conclusion

**Vibe coding is a decent prototyping tool, but produces unmaintainable garbage for production.**

The right workflow:
1. Vibe code to explore and prototype
2. Throw it all away
3. Implement properly with lessons learned

---

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

## MCP Toolset Integration in Factory Agents

### Configuring MCP Toolsets

Factory agents can connect to external MCP servers (like haiku-rag) via `mcp_client_toolsets` in `room_config.yaml`:

```yaml
agent:
  kind: "factory"
  factory_name: "my_package.factories.my_factory.create_my_agent"
  with_agent_config: true

mcp_client_toolsets:
  haiku-rag:
    kind: http
    url: "http://127.0.0.1:8001/mcp"
```

### Using MCP Toolsets in Factory Code

The factory receives `mcp_client_toolset_configs` and can create toolsets:

```python
@dataclasses.dataclass
class MyAgent:
    agent_config: config.FactoryAgentConfig
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None
    _rag_toolset = None

    def _get_rag_toolset(self):
        """Get MCP toolset from config."""
        if self._rag_toolset is None and self.mcp_client_toolset_configs:
            from soliplex import mcp_client

            rag_config = self.mcp_client_toolset_configs.get("haiku-rag")
            if rag_config:
                toolset_klass = mcp_client.TOOLSET_CLASS_BY_KIND[rag_config.kind]
                self._rag_toolset = toolset_klass(**rag_config.tool_kwargs)

        return self._rag_toolset
```

### Calling MCP Tools

Use `direct_call_tool()` within an async context manager:

```python
@agent.tool
async def rag_search(
    ctx: ai_tools.RunContext[MyContext],
    query: str,
) -> dict[str, typing.Any]:
    """Search the RAG knowledge base."""
    if not ctx.deps.rag_toolset:
        return {"error": "RAG toolset not configured"}

    try:
        async with ctx.deps.rag_toolset:
            result = await ctx.deps.rag_toolset.direct_call_tool(
                "search",
                {"query": query, "limit": 5},
            )
        return {"success": True, "results": result}
    except Exception as e:
        return {"error": str(e)}
```

### Passing Toolset to Context

The toolset must be passed to the agent context in `run_stream_events()`:

```python
async def run_stream_events(self, ...):
    agent = self._create_agent()
    rag_toolset = self._get_rag_toolset()
    ctx = MyContext(
        agent_config=self.agent_config,
        rag_toolset=rag_toolset,  # Pass toolset here
    )

    result = await agent.run(prompt, deps=ctx)
```

### Running the MCP Server

For haiku-rag, start the MCP server before using the room:

```bash
# Initialize database (first time only)
haiku-rag --config haiku-rag.yaml init --db db/rag.lancedb

# Start MCP server
haiku-rag --config haiku-rag.yaml serve --db db/rag.lancedb --mcp --mcp-port 8001
```

### haiku-rag MCP Tool Names

| Tool Name | Description | Arguments |
|-----------|-------------|-----------|
| `add_document_from_text` | Add text content | `{"content": "...", "title": "..."}` |
| `add_document_from_url` | Add from URL | `{"url": "https://...", "title": "..."}` |
| `add_document_from_file` | Add local file | `{"file_path": "/path/to/file", "title": "..."}` |
| `search_documents` | Hybrid search | `{"query": "...", "limit": 5}` |
| `list_documents` | List documents | `{"limit": 20}` |
| `delete_document` | Delete document | `{"document_id": "..."}` |
| `get_document` | Get by ID | `{"document_id": "..."}` |
| `ask_question` | RAG Q&A | `{"question": "..."}` |
| `research_question` | Deep research | `{"question": "..."}` |

### Error Handling

Always handle the case where the MCP server isn't running:

```python
if not ctx.deps.rag_toolset:
    return {"error": "RAG toolset not configured. Start haiku-rag MCP server first."}
```

---

## AG-UI Event Lifecycle Rule

### The Golden Rule

**If you emit a START event, you MUST emit a corresponding END event.**

This applies to all AG-UI paired events:
- `PartStartEvent` → `PartEndEvent`
- `TEXT_MESSAGE_START` → `TEXT_MESSAGE_END`
- `TOOL_CALL_START` → `TOOL_CALL_END`
- `THINKING_START` → `THINKING_END`
- `STEP_STARTED` → `STEP_FINISHED`
- `RUN_STARTED` → `RUN_FINISHED` (or `RUN_ERROR`)

### Common Pitfalls

1. **Index mismatch**: PartDeltaEvent must target the same index as the PartStartEvent
2. **Conditional returns**: Ensure all code paths end started parts
3. **Loop iterations**: Don't reuse part indices across loop iterations for different parts
4. **Unhandled exceptions**: Async operations (agent.run, asyncio.gather) can throw, bypassing PartEndEvent

### Known Issues in Factory Agents

The following factories have unhandled exceptions that can bypass `PartEndEvent`:

| Factory | Location | Risk |
|---------|----------|------|
| shark_tank_factory.py | Lines 214, 285, 399 | planner.run() and asyncio.gather() can throw |
| research_factory.py | Line 371 | asyncio.gather() can throw |
| introspective_factory.py | Line 1876 | synthesizer.run() can throw |

These need try-except blocks to ensure AG-UI event lifecycle completes on error.

### Correct Pattern

```python
# Start thinking part at known index
think_part = ai_messages.ThinkingPart("...")
yield ai_messages.PartStartEvent(index=0, part=think_part)

# Deltas ALWAYS target index 0 (the think_part)
for item in items:
    yield ai_messages.PartDeltaEvent(
        index=0,  # Always 0 for think_part
        delta=ai_messages.ThinkingPartDelta(content_delta=f"...")
    )

    # Tool calls get incrementing indices
    part_index += 1
    tc_part = ai_messages.ToolCallPart(tool_name="...")
    yield ai_messages.PartStartEvent(index=part_index, part=tc_part)
    yield ai_messages.PartEndEvent(index=part_index, part=tc_part)

# End thinking part at index 0
yield ai_messages.PartEndEvent(index=0, part=think_part)
```

---

## Dependency Injection for External Pattern Libraries

### The Problem

When integrating external pattern libraries (like `agentic-patterns`) with Soliplex, you
encounter a configuration mismatch:

1. **External libraries** have their own model defaults (e.g., `get_fast_model()`,
   `get_strong_model()`) that read from environment variables
2. **Soliplex** manages model configuration through `InstallationConfig` and room YAML files
3. **Result**: External patterns use wrong models because they can't see Soliplex's config

### The Solution: Factory Function Injection

Refactor external libraries to accept models via factory functions:

```python
# In agentic-patterns library
def create_generator_agent(model: Model | None = None) -> Agent:
    """Create generator with optional model override."""
    return Agent(
        model or get_fast_model(),  # Fallback to library default
        system_prompt=GENERATOR_PROMPT,
        ...
    )

async def run_best_of_n(
    problem: ProblemStatement,
    n: int = 5,
    *,
    generator: Agent | None = None,  # Accept pre-configured agents
    evaluator: Agent | None = None,
) -> BestOfNResult:
    ...
```

### Using in Soliplex Factories

The Soliplex factory creates models using installation config, then injects them:

```python
@dataclasses.dataclass
class ThoughtCandidatesAgent:
    agent_config: config.FactoryAgentConfig
    _fast_model: typing.Any = dataclasses.field(default=None, repr=False)
    _strong_model: typing.Any = dataclasses.field(default=None, repr=False)

    @property
    def fast_model_name(self) -> str:
        """Model name from room config - no hardcoded defaults."""
        return self.agent_config.extra_config["fast_model_name"]

    @property
    def strong_model_name(self) -> str:
        return self.agent_config.extra_config["strong_model_name"]

    def _get_fast_model(self):
        if self._fast_model is None:
            installation = self.agent_config._installation_config
            base_url = installation.get_environment("OLLAMA_BASE_URL")
            provider = ollama_providers.OllamaProvider(
                base_url=f"{base_url}/v1",
            )
            self._fast_model = openai_models.OpenAIChatModel(
                model_name=self.fast_model_name,
                provider=provider,
            )
        return self._fast_model

    async def run_stream_events(self, ...):
        result = await run_best_of_n(
            problem,
            n=self.num_candidates,
            generator=create_generator_agent(self._get_fast_model()),
            evaluator=create_evaluator_agent(self._get_strong_model()),
        )
```

### Room Configuration

Models are configured in `room_config.yaml`, not hardcoded in Python:

```yaml
agent:
  kind: "factory"
  factory_name: "crazy_glue.factories.thought_candidates_factory.create_thought_candidates_agent"
  with_agent_config: true
  extra_config:
    # Model configuration - required, no defaults
    fast_model_name: "qwen3:4b"      # Fast model for generation
    strong_model_name: "qwen3:8b"    # Strong model for evaluation

    # Pattern-specific config
    num_candidates: 5
    max_words: 100
```

### Key Principles

1. **No hardcoded model defaults in factory code** - All model names come from YAML config
2. **External libraries use factory functions** - Accept `Model | None` with internal defaults
3. **Soliplex factories create models** - Using `_installation_config.get_environment()`
4. **Inject at call site** - Pass configured agents/models to library functions
5. **Support multiple model tiers** - Fast models for generation, strong for evaluation

### Benefits

- **Separation of concerns**: Library doesn't know about Soliplex, Soliplex doesn't
  hardcode model names
- **Flexibility**: Each room can use different models
- **Testability**: Factory functions accept mocks easily
- **Backward compatibility**: Libraries still work standalone with their defaults

---

## Validating Room Configurations

### Always Run `check-config` Before Finalizing

**Critical Step**: After creating or modifying rooms, ALWAYS validate with:

```bash
soliplex-cli check-config ./installation.yaml
```

This catches configuration errors that Python syntax checks miss:
- Invalid YAML structure
- Missing required fields
- Factory import failures
- Incomplete room configs

### Lesson: Test Artifacts Can Break Production

**Problem Encountered**: A test (`test_apply_scaffold_allows_safe_paths`) wrote an
incomplete room config file (`rooms/test/room_config.yaml`) with only `id` and `name`
fields. This passed Python tests but broke the entire Soliplex installation:

```
FromYamlException: Error in YAML configuration:
/Users/runyaga/dev/crazy-glue/rooms/test/room_config.yaml; Kind: room; Config:
{'id': 'test', 'name': 'Test', ...};
```

**Root Cause**: The test actually wrote to the project directory instead of a temp
directory, leaving behind an incomplete config that `soliplex-cli check-config`
validates.

**Fix**: Tests that generate files should either:
1. Use `tempfile.TemporaryDirectory()` and write there
2. Clean up after themselves
3. Only test path validation logic without actual file writes

```python
# WRONG: Writes to project directory
def test_apply_scaffold(self, analysis_agent):
    safe_files = {"rooms/test/room_config.yaml": "id: test\nname: Test"}
    result = analysis_agent._apply_scaffold(safe_files)  # Writes to rooms/

# RIGHT: Test path filtering logic only
def test_apply_scaffold_allows_safe_paths(self, analysis_agent):
    safe_files = {"rooms/test-temp/room_config.yaml": "..."}
    allowed = ["rooms/", "src/crazy_glue/factories/"]
    for path in safe_files:
        assert any(path.startswith(p) for p in allowed)  # No file write
```

### Room Config Minimum Required Fields

A valid `room_config.yaml` requires at minimum:

```yaml
id: "my-room"
name: "My Room Name"
description: "What this room does"
welcome_message: |
  Welcome text

agent:
  kind: "default"  # or "factory"
```

Missing `description`, `welcome_message`, or `agent` block will fail validation.

---

## System Architect Room: Runtime Room Management

### Overview

The System Architect room allows introspecting and managing rooms at runtime,
including creating rooms, editing configs, managing tools, and storing prompts.

### Key Learning: RoomConfig Doesn't Accept Arbitrary Fields

**Problem**: Tried to track managed rooms with a `_managed_by` field in YAML:

```yaml
id: "calculator"
_managed_by: "analysis"  # BREAKS!
```

**Error**: `TypeError: RoomConfig.__init__() got unexpected keyword argument`

**Solution**: External tracking file (`db/managed_rooms.json`):

```json
{"managed_rooms": ["calculator", "my-room"]}
```

### Key Learning: Tool Config Format

**Problem**: Tried to wire custom tools as:

```yaml
tools:
  custom:src.crazy_glue.tools.my_tool: {}  # WRONG!
```

**Error**: `'str' object has no attribute 'get'`

**Root Cause**: Soliplex `tools:` expects a **list** of dicts with `tool_name`:

```yaml
tools:
  - tool_name: "soliplex.tools.search_documents"
    rag_lancedb_stem: "my_db"
```

**Key Insight**: Tools must be registered in `TOOL_CONFIG_CLASSES_BY_TOOL_NAME`.
Custom module imports are NOT supported without modifying soliplex core.

**Solution**: Tool generation writes files but does NOT auto-wire to room configs.

### Key Learning: Room Name Parsing

**Problem**: "create room9" extracted "NewRoom" instead of "room9".

**Root Cause**: Split on "room" word: `"create room9".split("room")` -> `["create ", "9"]`

**Solution**: Parse word-by-word, looking for standalone "room":

```python
for i in range(start_idx, len(words)):
    if words[i].lower() == "room":  # Standalone word
        break
    name_words.append(words[i])
```

### File-Based State Storage Pattern

Used JSON files in `db/` for simple persistence:

| File | Purpose |
|------|---------|
| `db/managed_rooms.json` | Track architect-created rooms |
| `db/pending_tool.json` | Stage tools for preview/approval |
| `db/prompts.json` | Prompt library storage |

**Benefits**: No database, easy to inspect/debug, survives restarts.

### Two-Step Tool Generation Workflow

```
1. generate tool <room> <name> <desc>  -> LLM generates code
2. [optional] Send refinement text     -> Regenerate with feedback
3. apply tool | discard tool           -> Finalize
```

**Benefits**: Prevents accidental file creation, allows iteration.

### Key Learning: Sanitize User Input for Python Identifiers

**Problem**: User-provided tool names like `big-filename` can't be used directly
as Python function names (hyphens, spaces, special chars, keywords are invalid).

**Solution**: Create a comprehensive `_sanitize_identifier` function:

```python
def _sanitize_identifier(name: str) -> tuple[str, str | None]:
    """Convert user-provided name to valid Python identifier."""
    import keyword, re

    if not name or not name.strip():
        return "", "Name cannot be empty"

    result = name.strip().lower()
    result = re.sub(r"[-\s.]+", "_", result)  # Separators → underscore
    result = re.sub(r"[^a-z0-9_]", "", result)  # Remove invalid chars
    result = re.sub(r"_+", "_", result)  # Collapse underscores
    result = result.strip("_")

    if result and result[0].isdigit():
        result = f"_{result}"  # Prefix leading digit

    if keyword.iskeyword(result):
        result = f"{result}_tool"  # Suffix keywords

    if not result.isidentifier():
        return "", f"Could not create valid identifier from '{name}'"

    return result, None
```

**Key points**:
- Validate early (in `_generate_tool`), not at each usage point
- Store sanitized name so downstream functions don't re-sanitize
- Return error tuple for clear error handling
- Test comprehensively (hyphens, spaces, digits, keywords, special chars)

### Key Learning: Use LLM for Tool Code Generation

**Problem**: Static f-string templates can't handle:
- Arbitrary input parameters (not just `path`)
- Domain-specific output models (not generic success/data/error)
- Actual implementation logic (not TODO placeholders)

**Solution**: Use the LLM to generate tool code dynamically:

```python
async def _generate_tool_code(self, name: str, description: str) -> str:
    prompt = f"""Generate a Python tool for pydantic-ai.

**Tool name**: {func_name}
**Description**: {description}

**Requirements**:
1. Create a Pydantic model for structured output
2. Create an async function that returns the model
3. Implement the ACTUAL logic, not a placeholder
4. Handle errors gracefully
"""
    agent = Agent(model, output_type=str)
    result = await agent.run(prompt)
    return result.output
```

**Key Points**:
- LLM generates domain-specific Pydantic models with meaningful fields
- LLM implements actual logic based on the description
- Refinement feedback can be passed for iterative improvement
- Still validate with AST, compile, import, and check-config after generation

### Tool Application Process (Critical)

When applying a generated tool, the full process must be:

1. **Validate Python syntax** - `ast.parse(code)`
2. **Validate compilation** - `compile(code, path, "exec")`
3. **Write tool file** to `src/crazy_glue/tools/<name>.py`
4. **Validate import** - `importlib.import_module(module)` to catch runtime errors
5. **Wire into room YAML** - Add to `tools:` list with `tool_name`
6. **Run check-config** - Validate the entire configuration
7. **Rollback on failure** - Remove from YAML and delete file if check fails

```python
def _apply_pending_tool(self) -> dict:
    # 1. AST validate
    ast.parse(code)

    # 2. Compile validate
    compile(code, str(file_path), "exec")

    # 3. Write file
    file_path.write_text(code)

    # 4. Import validate (catches runtime import errors)
    try:
        importlib.import_module(tool_module)
    except ImportError as e:
        file_path.unlink()  # Cleanup
        return {"status": "error", "message": f"Import failed: {e}"}

    # 5. Wire to room
    editor.add_tool(tool_module)
    editor.save()

    # 6. Check-config
    errors = editor.validate()
    if errors:
        # 7. Rollback
        editor.remove_tool(tool_module)
        editor.save()
        file_path.unlink()
        return {"status": "error", "message": errors}
```

**Key Insight**: The `tools:` section in room YAML must be a **list** format:

```yaml
tools:
  - tool_name: "crazy_glue.tools.my_tool"
  - tool_name: "soliplex.tools.search_documents"
    rag_lancedb_stem: "my_db"
```

NOT a dict format (which was our initial mistake):

```yaml
# WRONG!
tools:
  my_tool: {}
```

### Critical: Module Path Must NOT Include "src/"

**Problem**: Tool file at `src/crazy_glue/tools/my_tool.py` was registered as
`src.crazy_glue.tools.my_tool` which fails with:

```
ModuleNotFoundError: No module named 'src'
```

**Root Cause**: The `src/` directory is a Python packaging convention (src layout),
NOT part of the Python module path. Python imports from the package root.

**Solution**: Strip `src/` prefix when building module path:

```python
file_path_str = "src/crazy_glue/tools/my_tool.py"
module_path = file_path_str
if module_path.startswith("src/"):
    module_path = module_path[4:]  # Remove "src/"
tool_module = module_path.replace("/", ".").replace(".py", "")
# Result: "crazy_glue.tools.my_tool"
```

```yaml
# CORRECT
tools:
  - tool_name: "crazy_glue.tools.my_tool"

# WRONG - "src" is not a module!
tools:
  - tool_name: "src.crazy_glue.tools.my_tool"
```

### Key Learning: Tool Names Must Be Full Dotted Path to Function

**Problem**: Registered tool as `crazy_glue.tools.big_file` (module path) but
soliplex expects `crazy_glue.tools.big_file.big_file` (function path).

**Symptom**: Tool doesn't work or error about missing callable.

**Root Cause**: Soliplex tool registration expects a dotted path that resolves
to a **callable function**, not just a module.

**Solution**: Build full dotted path and validate function exists:

```python
# Build module path
tool_module = "crazy_glue.tools.big_file"  # Module
func_name = "big_file"  # Function name (same as tool name)
tool_dotted_path = f"{tool_module}.{func_name}"  # Full path

# Validate function exists and is callable
module = importlib.import_module(tool_module)
tool_func = getattr(module, func_name, None)
if tool_func is None:
    raise ValueError(f"Function '{func_name}' not found in {tool_module}")
if not callable(tool_func):
    raise ValueError(f"'{func_name}' in {tool_module} is not callable")

# Register with full path
editor.add_tool(tool_dotted_path)  # "crazy_glue.tools.big_file.big_file"
```

```yaml
# CORRECT - full path to function
tools:
  - tool_name: crazy_glue.tools.big_file.big_file

# WRONG - just module path
tools:
  - tool_name: crazy_glue.tools.big_file
```

### Key Learning: Validate YAML Before Saving

**Problem**: Generated scaffolds or modified configs can have broken YAML
(e.g., improper use of `|` for multi-line strings, missing indentation).

**Solution**: Always validate YAML with `yaml.safe_load()` before writing:

```python
# Before writing any YAML file
if rel_path.endswith(".yaml") or rel_path.endswith(".yml"):
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML in {rel_path}: {e}")
        continue  # Skip writing invalid file

# After saving (if modifying via RoomConfigEditor)
try:
    yaml.safe_load(editor.config_path.read_text())
except yaml.YAMLError as e:
    # Rollback changes
    return {"status": "error", "message": f"Invalid YAML after save: {e}"}
```

**Common YAML Issues**:
- Multi-line strings without `|` or `>` indicators
- Incorrect indentation for nested blocks
- Unquoted special characters
- Missing colons or values

### Key Learning: Tool Descriptions Come From Function Docstrings

**Problem**: Tool validation fails with:
```
1 validation error for Tool
tool_description
  Input should be a valid string [type=string_type, input_value=None]
```

**Root Cause**: Soliplex has two different classes:
- `soliplex.config.ToolConfig` - runtime config (no `tool_description` field)
- `soliplex.models.Tool` - Pydantic validation model (requires `tool_description`)

The validation model extracts `tool_description` from the **function's docstring**.
If the function has no docstring, validation fails.

**Broken Code** (no docstring on function):
```python
async def big_filename(path: str = ".") -> BigFilenameResult:
    try:  # No docstring! Validation will fail.
        ...
```

**Working Code** (docstring on function):
```python
async def big_filename(path: str = ".") -> BigFilenameResult:
    """Find the file or folder with the longest filename."""
    try:
        ...
```

**Key Point**: The docstring must be on the **function itself**, not just at
module level. pydantic-ai extracts the function's docstring for the tool
description.

**Best Practice**: When generating tools with LLM, ensure the prompt explicitly
requires a docstring on the async function - this becomes the tool's description.

### Key Learning: YAML Dump Doesn't Preserve Block Style

**Problem**: When using `yaml.dump()` to save YAML files, multi-line strings
lose their block style (`|`) and become quoted strings with embedded newlines:

```yaml
# Before (scaffold template)
welcome_message: |
  Welcome to the room!

# After yaml.dump()
welcome_message: 'Welcome to the room!

  '
```

**Root Cause**: PyYAML's default dumper uses the most compact representation,
which for multi-line strings is quoted style with literal `\n` characters.

**Solution**: Create a custom dumper with a string representer that detects
newlines and uses block style:

```python
def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
    """Use block style (|) for multi-line strings."""
    tag = "tag:yaml.org,2002:str"
    if "\n" in data:
        return dumper.represent_scalar(tag, data, style="|")
    return dumper.represent_scalar(tag, data)

class _BlockStyleDumper(yaml.SafeDumper):
    pass

_BlockStyleDumper.add_representer(str, _str_representer)

# Usage
content = yaml.dump(data, Dumper=_BlockStyleDumper, ...)
```

**Result**: Multi-line strings now preserve readable block style:

```yaml
welcome_message: |
  Welcome to the room!
```

**Note**: Tree-sitter is NOT needed for this - PyYAML's custom representers
handle it cleanly.

### Key Learning: Use Structured Output for LLM Multi-Value Generation

**Problem**: Need LLM to generate both code AND a summary/description, but
simple string output only gives one value.

**Initial approach** (limited):
```python
agent: Agent[None, str] = Agent(model, output_type=str)
result = await agent.run(prompt)
code = result.output  # Only get code, no description
```

**Solution**: Use Pydantic structured output to get multiple values:

```python
import pydantic
from pydantic_ai import Agent

class GeneratedTool(pydantic.BaseModel):
    """LLM output containing generated tool code and description."""
    code: str = pydantic.Field(
        description="Complete Python code for the tool"
    )
    summary: str = pydantic.Field(
        description="Concise 1-sentence tool description"
    )

# Use structured output
agent: Agent[None, GeneratedTool] = Agent(
    model, output_type=GeneratedTool
)
result = await agent.run(prompt)
tool_output = result.output

code = tool_output.code      # Generated Python code
summary = tool_output.summary  # Concise description
```

**Benefits**:
- Single LLM call produces multiple values
- Pydantic validates the output structure
- Field descriptions guide the LLM
- Easy to extend with more fields (e.g., `input_type`, `output_type`)

**Prompt guidance**: Tell the LLM what each field should contain:

```
Provide:
1. **code**: Complete Python code (no markdown fences)
2. **summary**: Concise 1-sentence description (e.g., "Find large files")
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
8. Consider adding model tier configs at installation level for common defaults
9. Add `soliplex-cli check-config` to CI pipeline to catch invalid rooms early
