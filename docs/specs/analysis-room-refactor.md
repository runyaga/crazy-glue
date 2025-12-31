# Spec: Refactoring Analysis Room to Use Agentic Patterns

## Status: Draft

## Problem Statement

The System Architect room claims to be "powered by the Cartographer pattern" but is
actually an imperative command handler with ~2000 lines of if/elif parsing. It doesn't
leverage any real agentic patterns from the `agentic_patterns` library.

### Current Architecture

```
User Input
    ↓
run_stream_events()
    ↓
if/elif chain (~50 branches)
    ↓
Direct function calls (_scaffold_room, _manage_tool, etc.)
    ↓
File I/O (YAML, JSON)
    ↓
Response string
```

### Problems

1. **No agency** - The agent doesn't decide what to do; the if/elif chain does
2. **Brittle parsing** - Relies on exact string matching ("create", "rooms", etc.)
3. **No learning** - Failed operations don't improve future attempts
4. **No reflection** - Generated code isn't reviewed before presenting
5. **No planning** - Complex multi-step tasks executed linearly
6. **Monolithic** - Single 2000-line file handles everything

---

## Proposed Architecture

### Pattern Selection

| Pattern | Use Case | Benefit |
|---------|----------|---------|
| **Tool Use** | All operations | Agent decides which tool to call based on intent |
| **Reflection** | Code generation | Critic reviews generated code before presenting |
| **Planning** | Complex tasks | Decompose "create room with RAG" into steps |
| **Routing** | Request classification | Route to specialized sub-agents |

### New Architecture

```
User Input
    ↓
Router Agent (classifies intent)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Room Agent     │  Tool Agent     │  Query Agent    │
│  (CRUD rooms)   │  (generate/wire)│  (search/find)  │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                   ↓                   ↓
pydantic-ai tools   Reflection loop    Knowledge graph
    ↓                   ↓                   ↓
RoomConfigEditor    Code + Critic     explore_domain
```

---

## Detailed Design

### 1. Router Agent (Routing Pattern)

Instead of if/elif parsing, use an LLM to classify user intent:

```python
class UserIntent(pydantic.BaseModel):
    """Classified user intent."""
    category: Literal["room_management", "tool_generation", "query", "config"]
    action: str  # e.g., "create", "edit", "list", "generate"
    entities: dict[str, str]  # extracted entities like room_id, name, etc.
    confidence: float

router_agent = Agent(
    model,
    output_type=UserIntent,
    system_prompt="""Classify user requests into categories:
    - room_management: create, edit, delete, list rooms
    - tool_generation: generate, apply, discard tools
    - query: find, search, inspect, show
    - config: secrets, mcp, prompts

    Extract relevant entities (room_id, tool_name, etc.)"""
)
```

**Benefits:**
- Natural language understanding vs exact string matching
- Handles variations ("make a room", "create room", "new room")
- Extracts entities in one pass

### 2. Room Agent (Tool Use Pattern)

Convert room operations to pydantic-ai tools:

```python
@dataclass
class RoomContext:
    editor_cache: dict[str, RoomConfigEditor]
    installation_config: InstallationConfig

room_agent = Agent(
    model,
    deps_type=RoomContext,
    system_prompt="You manage soliplex rooms. Use tools to create, edit, list rooms."
)

@room_agent.tool
def list_rooms(ctx: RunContext[RoomContext]) -> list[dict]:
    """List all registered rooms."""
    return [
        {"id": r.id, "name": r.name, "description": r.description}
        for r in ctx.deps.installation_config.room_configs.values()
    ]

@room_agent.tool
def create_room(
    ctx: RunContext[RoomContext],
    name: str,
    description: str,
    model_name: str = "gpt-oss:latest",
) -> dict:
    """Create a new room with the given name and description."""
    # Implementation using RoomConfigEditor
    ...

@room_agent.tool
def edit_room(
    ctx: RunContext[RoomContext],
    room_id: str,
    field: str,
    value: str,
) -> dict:
    """Edit a room's configuration field."""
    ...
```

**Benefits:**
- Agent decides which tool based on context
- Tools are self-documenting via docstrings
- Composable - agent can call multiple tools per request

### 3. Tool Generation Agent (Reflection Pattern)

Use critic/producer loop for code generation:

```python
class GeneratedCode(pydantic.BaseModel):
    code: str
    function_name: str
    description: str
    test_cases: list[str]

class CodeReview(pydantic.BaseModel):
    approved: bool
    issues: list[str]
    suggestions: list[str]
    severity: Literal["blocker", "major", "minor", "none"]

producer_agent = Agent(
    model,
    output_type=GeneratedCode,
    system_prompt="Generate pydantic-ai tool code..."
)

critic_agent = Agent(
    model,
    output_type=CodeReview,
    system_prompt="""Review generated Python code for:
    - Syntax errors
    - Missing docstring (required for tool description)
    - Missing error handling
    - Security issues (path traversal, injection)
    - Pydantic model correctness
    """
)

async def generate_with_reflection(
    name: str,
    description: str,
    max_iterations: int = 3,
) -> GeneratedCode:
    """Generate code with critic review loop."""

    for i in range(max_iterations):
        # Generate
        code = await producer_agent.run(
            f"Generate tool '{name}': {description}"
        )

        # Review
        review = await critic_agent.run(
            f"Review this code:\n```python\n{code.output.code}\n```"
        )

        if review.output.approved:
            return code.output

        # Incorporate feedback for next iteration
        description += f"\n\nFix these issues: {review.output.issues}"

    raise ReflectionFailed("Max iterations reached")
```

**Benefits:**
- Generated code is reviewed before presenting to user
- Issues caught early (missing docstrings, syntax errors)
- Iterative improvement within single request

### 4. Query Agent (Knowledge Graph + Tool Use)

Use the knowledge graph properly:

```python
@dataclass
class QueryContext:
    knowledge_store: KnowledgeStore
    installation_config: InstallationConfig

query_agent = Agent(
    model,
    deps_type=QueryContext,
    system_prompt="""You answer questions about the codebase.
    Use tools to search the knowledge graph and inspect entities."""
)

@query_agent.tool
def search_entities(
    ctx: RunContext[QueryContext],
    query: str,
    entity_type: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search the knowledge graph for entities matching the query."""
    store = ctx.deps.knowledge_store
    return store.search(query, entity_type=entity_type, limit=limit)

@query_agent.tool
def get_entity_details(
    ctx: RunContext[QueryContext],
    entity_id: str,
) -> dict:
    """Get detailed information about a specific entity."""
    store = ctx.deps.knowledge_store
    entity = store.get(entity_id)
    return {
        "id": entity.id,
        "type": entity.type,
        "name": entity.name,
        "source": entity.source_file,
        "links": [l.to_dict() for l in store.get_links(entity_id)],
    }

@query_agent.tool
def explain_relationship(
    ctx: RunContext[QueryContext],
    entity_a: str,
    entity_b: str,
) -> str:
    """Explain the relationship between two entities."""
    # Use knowledge graph to find path between entities
    ...
```

**Benefits:**
- Agent can explore knowledge graph conversationally
- Combines multiple queries to answer complex questions
- Uses LLM to synthesize findings

### 5. Planning for Complex Tasks

For multi-step requests like "create a room with RAG tools configured for document search":

```python
class TaskStep(pydantic.BaseModel):
    action: str
    target: str
    parameters: dict[str, Any]
    depends_on: list[int] = []

class TaskPlan(pydantic.BaseModel):
    goal: str
    steps: list[TaskStep]
    estimated_tools: list[str]

planner_agent = Agent(
    model,
    output_type=TaskPlan,
    system_prompt="""Decompose complex requests into executable steps.
    Each step should map to a single tool call.
    Identify dependencies between steps."""
)

async def execute_with_planning(request: str) -> dict:
    # Plan
    plan = await planner_agent.run(request)

    # Present plan for approval (human-in-the-loop)
    yield PlanProposedEvent(plan=plan.output)

    # Wait for approval
    approval = await wait_for_approval()
    if not approval.approved:
        return {"status": "cancelled", "reason": approval.reason}

    # Execute steps
    results = {}
    for i, step in enumerate(plan.output.steps):
        # Check dependencies
        for dep in step.depends_on:
            if results.get(dep, {}).get("status") != "success":
                return {"status": "failed", "step": i, "reason": "dependency failed"}

        # Execute
        result = await execute_step(step)
        results[i] = result

        yield StepCompletedEvent(step=i, result=result)

    return {"status": "complete", "results": results}
```

**Benefits:**
- Complex tasks decomposed into reviewable steps
- User can approve/modify plan before execution
- Dependencies handled correctly

---

## Migration Strategy

### Phase 1: Extract Tools (Low Risk)

Convert existing functions to pydantic-ai tools without changing control flow:

```python
# Before: direct call in if/elif
if "rooms" in prompt:
    result = self._list_rooms()

# After: tool that can be called by agent OR directly
@agent.tool
def list_rooms(ctx: RunContext[AnalysisContext]) -> list[dict]:
    """List all registered rooms."""
    return ctx.deps.list_rooms()  # delegate to existing method
```

### Phase 2: Add Router (Medium Risk)

Add intent classification but keep fallback to if/elif:

```python
async def run_stream_events(self, ...):
    # Try router first
    try:
        intent = await self.router.run(prompt)
        if intent.output.confidence > 0.8:
            return await self.dispatch(intent.output)
    except:
        pass

    # Fallback to existing if/elif
    return await self._legacy_dispatch(prompt)
```

### Phase 3: Add Reflection (Medium Risk)

Add critic loop to tool generation:

```python
# Before
code = await self._generate_tool_code(name, description)

# After
code = await self._generate_with_reflection(name, description)
```

### Phase 4: Full Agent Architecture (High Risk)

Replace if/elif with agent dispatch:

```python
async def run_stream_events(self, ...):
    intent = await self.router.run(prompt)

    match intent.output.category:
        case "room_management":
            return await self.room_agent.run(prompt, deps=self.room_ctx)
        case "tool_generation":
            return await self.tool_agent.run(prompt, deps=self.tool_ctx)
        case "query":
            return await self.query_agent.run(prompt, deps=self.query_ctx)
```

---

## File Structure

```
src/crazy_glue/factories/
├── analysis/
│   ├── __init__.py
│   ├── router.py          # Intent classification
│   ├── room_agent.py      # Room CRUD tools
│   ├── tool_agent.py      # Tool generation with reflection
│   ├── query_agent.py     # Knowledge graph queries
│   ├── planner.py         # Task decomposition
│   └── room_editor.py     # (existing) YAML operations
└── analysis_factory.py    # Orchestrator (slimmed down)
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Lines in analysis_factory.py | ~2000 | <500 |
| If/elif branches | ~50 | 0 |
| Patterns used | 1 (explore_domain) | 4+ |
| Natural language flexibility | Low (exact match) | High (LLM classified) |
| Code generation quality | No review | Critic-reviewed |

---

## Validation Layer (Preserve & Enhance)

The current implementation has hard-won validation routines that MUST be preserved.
These represent lessons learned from production failures.

### Current Validations (Keep)

| Validation | Location | Purpose |
|------------|----------|---------|
| `_sanitize_identifier()` | analysis_factory.py | Convert user input to valid Python identifiers |
| YAML parse check | `yaml.safe_load()` after every save | Catch broken YAML before it breaks soliplex |
| `check-config` | `editor.validate()` | Full soliplex config validation |
| Import validation | `importlib.import_module()` | Verify generated tool is importable |
| Function existence | `getattr(module, func_name)` | Verify function exists and is callable |
| Docstring check | (implicit via check-config) | Tool description comes from docstring |

### Validation Flow (Current)

```
User Input
    ↓
_sanitize_identifier() → reject invalid names early
    ↓
LLM generates code
    ↓
ast.parse() → syntax validation
    ↓
compile() → compilation validation
    ↓
write file
    ↓
importlib.import_module() → import validation
    ↓
getattr(module, func) → function exists?
    ↓
callable(func) → is it callable?
    ↓
editor.add_tool() + save()
    ↓
yaml.safe_load() → YAML valid?
    ↓
editor.validate() → soliplex check-config
    ↓
rollback on any failure
```

### Enhancements for Refactor

#### 1. Move Validation to Dedicated Module

```python
# src/crazy_glue/factories/analysis/validators.py

class ValidationError(Exception):
    """Base validation error with context."""
    def __init__(self, message: str, stage: str, recoverable: bool = False):
        self.message = message
        self.stage = stage
        self.recoverable = recoverable

def validate_identifier(name: str) -> str:
    """Sanitize and validate Python identifier. Raises ValidationError."""
    sanitized, error = _sanitize_identifier(name)
    if error:
        raise ValidationError(error, stage="identifier", recoverable=True)
    return sanitized

def validate_python_code(code: str, file_path: Path) -> None:
    """Validate Python code syntax and compilation."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}", stage="syntax")

    try:
        compile(code, str(file_path), "exec")
    except Exception as e:
        raise ValidationError(f"Compilation error: {e}", stage="compile")

def validate_tool_import(module_path: str, func_name: str) -> Callable:
    """Validate tool can be imported and function exists."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValidationError(f"Import failed: {e}", stage="import")

    func = getattr(module, func_name, None)
    if func is None:
        raise ValidationError(
            f"Function '{func_name}' not found in {module_path}",
            stage="function"
        )
    if not callable(func):
        raise ValidationError(
            f"'{func_name}' is not callable",
            stage="callable"
        )
    return func

def validate_yaml_config(editor: RoomConfigEditor) -> None:
    """Validate YAML is parseable and passes check-config."""
    # Parse check
    try:
        yaml.safe_load(editor.config_path.read_text())
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML: {e}", stage="yaml_parse")

    # Soliplex check-config
    errors = editor.validate()
    if errors:
        raise ValidationError(
            f"Config validation failed: {errors}",
            stage="check_config"
        )
```

#### 2. Critic Agent Validates Generated Code

The Reflection pattern's critic should check our validation rules:

```python
critic_system_prompt = """Review generated Python code for:

**Hard Requirements (blockers):**
- Function MUST have a docstring (soliplex extracts tool_description from it)
- Function MUST be async
- Function MUST return a Pydantic model
- Function name MUST be valid Python identifier (no hyphens, spaces, keywords)
- All imports MUST be at module level or inside function

**Soft Requirements (warnings):**
- Should handle exceptions gracefully
- Should return error info in the model, not raise
- Should use pathlib for file operations
- Should validate inputs before processing

**Security (blockers):**
- No shell injection (subprocess with user input)
- No path traversal (validate paths stay within bounds)
- No eval/exec of user input
"""
```

#### 3. Validation as Pre-commit Hook

Add validation that runs before any tool is wired:

```python
class ToolValidationPipeline:
    """Pipeline of validators that must all pass."""

    def __init__(self):
        self.validators = [
            self._validate_identifier,
            self._validate_syntax,
            self._validate_imports,
            self._validate_docstring,
            self._validate_return_type,
        ]

    async def validate(self, name: str, code: str) -> list[ValidationError]:
        """Run all validators, collect all errors."""
        errors = []
        for validator in self.validators:
            try:
                await validator(name, code)
            except ValidationError as e:
                errors.append(e)
        return errors

    def _validate_docstring(self, name: str, code: str) -> None:
        """Ensure function has docstring (required for tool description)."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
                docstring = ast.get_docstring(node)
                if not docstring:
                    raise ValidationError(
                        f"Function '{name}' missing docstring (required for tool description)",
                        stage="docstring",
                        recoverable=True  # LLM can fix this
                    )
                return
        raise ValidationError(f"Function '{name}' not found", stage="function")
```

#### 4. Rollback Registry

Track what needs rollback on failure:

```python
class RollbackRegistry:
    """Track operations that need rollback on failure."""

    def __init__(self):
        self._operations: list[Callable[[], None]] = []

    def register(self, rollback_fn: Callable[[], None]) -> None:
        """Register a rollback operation."""
        self._operations.append(rollback_fn)

    def rollback_all(self) -> list[str]:
        """Execute all rollbacks in reverse order."""
        errors = []
        for op in reversed(self._operations):
            try:
                op()
            except Exception as e:
                errors.append(str(e))
        self._operations.clear()
        return errors

    def clear(self) -> None:
        """Clear registry (called on success)."""
        self._operations.clear()

# Usage in tool application
async def apply_tool(pending: dict) -> dict:
    rollback = RollbackRegistry()

    try:
        # Write file
        file_path.write_text(code)
        rollback.register(lambda: file_path.unlink() if file_path.exists() else None)

        # Validate import
        validate_tool_import(module_path, func_name)

        # Wire to room
        editor.add_tool(tool_path)
        editor.save()
        rollback.register(lambda: (editor.remove_tool(tool_path), editor.save()))

        # Validate config
        validate_yaml_config(editor)

        # Success - clear rollback registry
        rollback.clear()
        return {"status": "success"}

    except ValidationError as e:
        # Rollback all operations
        rollback_errors = rollback.rollback_all()
        return {
            "status": "error",
            "message": e.message,
            "stage": e.stage,
            "rollback_errors": rollback_errors,
        }
```

### Lessons Learned (Codified)

These MUST be preserved in the refactor:

| Lesson | Implementation |
|--------|----------------|
| Tool descriptions come from docstrings | `_validate_docstring()` checks function has docstring |
| Sanitize user input for identifiers | `_sanitize_identifier()` with comprehensive rules |
| Module path excludes "src/" | Strip prefix when building import path |
| Tool name is `module.function` not just `module` | Build full dotted path to callable |
| YAML dump loses block style | Use `_BlockStyleDumper` with custom representer |
| Validate after every save | `yaml.safe_load()` + `editor.validate()` |
| Rollback on failure | `RollbackRegistry` tracks cleanup operations |
| ToolConfig vs models.Tool | Config has no description, model requires it (from docstring) |
| RoomConfig rejects arbitrary fields | Use external JSON (`db/managed_rooms.json`) for tracking |
| Tools list format not dict | `tools: [{tool_name: ...}]` not `tools: {name: {}}` |
| Room name parsing | Word-by-word, look for standalone "room" keyword |
| Use structured output for multi-value | Pydantic model for LLM to return code + description |

### AG-UI Event Lessons (Preserve)

The refactor MUST maintain proper AG-UI event handling:

| Lesson | Implementation |
|--------|----------------|
| PartStart requires PartEnd | Every `PartStartEvent` must have matching `PartEndEvent` |
| Tool calls invisible with agent.run() | Extract from `result.all_messages()` and emit `ToolCallPart` |
| State snapshots race with RUN_FINISHED | Use activities instead of state for final data |
| Index consistency | PartDeltaEvent must target same index as PartStartEvent |

```python
# Pattern for tool call visibility in refactored agents
async def run_with_tool_visibility(agent, prompt, deps):
    result = await agent.run(prompt, deps=deps)

    # Extract and emit tool calls for AG-UI visibility
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    yield PartStartEvent(index=idx, part=part)
                    yield PartEndEvent(index=idx, part=part)
                    idx += 1

    yield AgentRunResultEvent(result=result)
```

### Streaming Considerations for Multi-Agent

When router dispatches to sub-agents, streaming needs coordination:

```python
async def run_stream_events(self, ...):
    # Start thinking
    think_part = ThinkingPart("Analyzing request...")
    yield PartStartEvent(index=0, part=think_part)

    # Router classifies (yields delta updates)
    yield PartDeltaEvent(index=0, delta=ThinkingPartDelta("Classifying intent..."))
    intent = await self.router.run(prompt)

    yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(f"Intent: {intent.category}"))

    # Dispatch to sub-agent (forward its events)
    sub_agent = self.agents[intent.category]
    async for event in sub_agent.run_stream(prompt, deps=deps):
        # Re-index events to avoid collision with our think_part
        if isinstance(event, PartStartEvent):
            event = PartStartEvent(index=event.index + 1, part=event.part)
        yield event

    # End our thinking part
    yield PartEndEvent(index=0, part=think_part)
```

---

## Open Questions

1. **Latency**: Multiple LLM calls (router → agent → tools) adds latency. Acceptable?
2. **Cost**: More LLM calls = higher cost. Worth it for better UX?
3. **Streaming**: How to stream progress from nested agent calls?
4. **Error recovery**: If one agent fails, how to gracefully degrade?
5. **Testing**: How to test agent behavior deterministically?

---

## References

- [agentic-patterns-book](https://github.com/runyaga/agentic-patterns-book)
- Chapter 21: Introspective Pattern
- Chapter 15: Reflection Pattern
- Chapter 12: Planning Pattern
- Chapter 8: Routing Pattern
