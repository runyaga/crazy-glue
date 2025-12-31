# Conversational Planning for System Architect

## Vision

Enable natural language multi-step requests that get parsed into command plans, reviewed by user, then executed deterministically.

```
User: create room333, add a most3s tool that finds filenames with most 3s, update the suggestion

Planner (qwen3:4b):
  Planning your request...

  ## Execution Plan
  1. `create room333 room` - Create new room
  2. `generate tool room333 most3s traverse directory finding filename with most 3s`
  3. `edit room333 suggestion Find the file with most 3s in its name`

  Proceed? [y/n]

User: y

Agent: [executes each command, streaming results]
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Natural Language                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Complexity Detector                        │
│         (keyword check: multiple verbs? conjunctions?)       │
└─────────────────────────────────────────────────────────────┘
                    │                    │
            simple  │                    │ complex
                    ▼                    ▼
┌───────────────────────┐    ┌───────────────────────────────┐
│   Keyword Parser      │    │      LLM Planner              │
│   (existing parser.py)│    │      (qwen3:4b fast)          │
└───────────────────────┘    └───────────────────────────────┘
                    │                    │
                    │                    ▼
                    │        ┌───────────────────────────────┐
                    │        │      Plan Display             │
                    │        │      (await user confirm)     │
                    │        └───────────────────────────────┘
                    │                    │
                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   Command Executor                           │
│           (iterate HANDLERS, yield events)                   │
└─────────────────────────────────────────────────────────────┘
```

## Model Tiers

| Tier | Model | Use Case | Latency |
|------|-------|----------|---------|
| Fast | qwen3:4b | Complexity detection, simple planning | ~200ms |
| Medium | qwen3:8b | Multi-step planning, disambiguation | ~400ms |
| Deep | gpt-oss:20b | Tool code generation (existing) | ~2-3s |

## Components

### 1. Complexity Detector

Simple heuristic check before invoking LLM:

```python
# src/crazy_glue/analysis/planner.py

def is_complex_request(prompt: str) -> bool:
    """Detect if request needs LLM planning vs keyword parsing."""
    prompt_lower = prompt.lower()

    # Multiple action verbs
    verbs = ["create", "generate", "add", "remove", "edit", "change", "update"]
    verb_count = sum(1 for v in verbs if v in prompt_lower)
    if verb_count > 1:
        return True

    # Conjunctions suggesting multiple tasks
    conjunctions = [" and ", " then ", ", ", " also ", " plus "]
    if any(c in prompt_lower for c in conjunctions):
        return True

    # Pronouns referencing previous context
    if any(p in prompt_lower for p in [" it ", " that ", " the room"]):
        return True

    return False
```

### 2. Command Plan Model

```python
# src/crazy_glue/analysis/planner.py

import pydantic

class PlannedCommand(pydantic.BaseModel):
    """A single planned command."""
    command: str  # The command string to execute
    description: str  # Human-readable description
    requires_confirm: bool = False  # Destructive operations

class ExecutionPlan(pydantic.BaseModel):
    """Plan output from LLM planner."""
    commands: list[PlannedCommand]
    clarifications: list[str] = []  # Questions if ambiguous
    warnings: list[str] = []  # Potential issues
```

### 3. LLM Planner

```python
# src/crazy_glue/analysis/planner.py

PLANNER_PROMPT = """You are a command planner for a room management system.

Convert the user's natural language request into a sequence of commands.

Available commands:
- `create <name> room` - Create a new room
- `generate tool <room> <name> <description>` - Generate a tool
- `edit <room> description|prompt|model|welcome <value>` - Edit room field
- `add suggestion <room> <text>` - Add a suggestion
- `remove suggestion <room> <index>` - Remove suggestion
- `add tool <room> <tool-name>` - Add predefined tool
- `remove tool <room> <tool-name>` - Remove tool
- `inspect <room>` - Show room details
- `rooms` - List all rooms

Rules:
1. Output each command on its own line
2. Use exact command syntax
3. If ambiguous, add clarification questions
4. Order commands logically (create before edit)
5. Reference rooms by their ID (slugified name)

User request: {request}

Output a JSON ExecutionPlan with commands, clarifications, and warnings."""


async def plan_request(
    prompt: str,
    model_name: str = "qwen3:4b",
) -> ExecutionPlan:
    """Plan a complex request into executable commands."""
    from pydantic_ai import Agent

    agent: Agent[None, ExecutionPlan] = Agent(
        model=_get_fast_model(model_name),
        output_type=ExecutionPlan,
        system_prompt=PLANNER_PROMPT,
    )

    result = await agent.run(prompt)
    return result.output


def _get_fast_model(model_name: str):
    """Get a fast local model for planning."""
    from pydantic_ai.models import openai as openai_models
    from pydantic_ai.providers import ollama as ollama_providers

    provider = ollama_providers.OllamaProvider(
        base_url="http://localhost:11434/v1",
    )
    return openai_models.OpenAIChatModel(
        model_name=model_name,
        provider=provider,
    )
```

### 4. Plan Executor

```python
# src/crazy_glue/analysis/planner.py

from crazy_glue.analysis import HANDLERS, parse_command

async def execute_plan(
    ctx: AnalysisContext,
    plan: ExecutionPlan,
) -> AsyncIterator[NativeEvent]:
    """Execute a confirmed plan."""

    for i, planned in enumerate(plan.commands):
        # Parse the command string
        cmd = parse_command(planned.command)

        # Get handler
        handler = HANDLERS.get(cmd.command)
        if not handler:
            yield _error_event(f"Unknown command: {cmd.command}")
            continue

        # Yield step indicator
        yield _text_event(f"\n**Step {i+1}**: {planned.description}\n")

        # Execute and yield events
        async for event in handler(ctx, cmd):
            yield event
```

### 5. Integration in Factory

```python
# src/crazy_glue/factories/analysis_factory.py

async def run_stream_events(self, ...):
    user_prompt = _extract_prompt(message_history)
    ctx = self._build_context()

    # Check complexity
    if is_complex_request(user_prompt):
        # Plan with fast model
        yield _thinking_event("Planning your request...")

        plan = await plan_request(user_prompt, model_name="qwen3:4b")

        # Show plan and wait for confirmation
        if plan.clarifications:
            yield _text_event(format_clarifications(plan))
            return  # Wait for user response

        yield _text_event(format_plan(plan))

        # Store plan for confirmation
        ctx.save_pending_plan(plan)
        yield _text_event("\nProceed? Reply 'y' to execute or describe changes.")
        return

    # Check for plan confirmation
    if user_prompt.lower() in ("y", "yes", "proceed"):
        plan = ctx.load_pending_plan()
        if plan:
            async for event in execute_plan(ctx, plan):
                yield event
            ctx.clear_pending_plan()
            return

    # Simple command - use keyword parser
    cmd = parse_command(user_prompt)
    handler = HANDLERS.get(cmd.command, handle_unknown)
    async for event in handler(ctx, cmd):
        yield event
```

## Plan Display Format

```markdown
## Execution Plan

I'll execute these steps:

1. **Create room** - `create room333 room`
2. **Generate tool** - `generate tool room333 most3s traverse directory finding filename with most 3s`
3. **Update suggestion** - `edit room333 suggestion Find the file with most 3s`

⚠️ **Note**: Step 2 will use the code generator (may take a few seconds)

Proceed? Reply **y** to execute, or describe any changes.
```

## Clarification Flow

When the planner detects ambiguity:

```markdown
## Clarification Needed

I understand you want to create a room and add a tool, but I have questions:

1. **Room name**: Should it be "room333" or "room-333"?
2. **Tool behavior**: Should "most 3s" count:
   - Only the filename? (e.g., `file333.txt`)
   - The full path? (e.g., `/path/to/333/file.txt`)

Please clarify and I'll create the plan.
```

## File Structure

```
src/crazy_glue/analysis/
├── __init__.py          # Add planner exports
├── planner.py           # NEW: Planning logic
├── parser.py            # Existing keyword parser
├── handlers.py          # Existing handlers
├── context.py           # Add pending_plan storage
└── formatters.py        # Add plan formatting
```

## Tasks

- [ ] Create `planner.py` with:
  - [ ] `is_complex_request()` heuristic
  - [ ] `PlannedCommand` and `ExecutionPlan` models
  - [ ] `plan_request()` using qwen3:4b
  - [ ] `execute_plan()` iterator
- [ ] Update `context.py`:
  - [ ] Add `pending_plan_path`
  - [ ] Add `load_pending_plan()`, `save_pending_plan()`, `clear_pending_plan()`
- [ ] Update `formatters.py`:
  - [ ] Add `format_plan()`
  - [ ] Add `format_clarifications()`
- [ ] Update `analysis_factory.py`:
  - [ ] Integrate complexity check
  - [ ] Handle plan confirmation flow
- [ ] Add tests:
  - [ ] Test `is_complex_request()` heuristics
  - [ ] Test plan parsing with mock LLM
  - [ ] Test execution flow
- [ ] Configuration:
  - [ ] Add `planner_model` to agent config
  - [ ] Allow model override per-tier

## Success Criteria

- [ ] Simple commands ("rooms") execute instantly (<50ms)
- [ ] Complex requests trigger planning (<500ms with qwen3:4b)
- [ ] User sees plan before execution
- [ ] Plans execute correctly using existing handlers
- [ ] Clarifications asked for ambiguous requests
- [ ] Can abort/modify plan before execution

## Example Flows

### Simple (no LLM)
```
User: rooms
→ Keyword parser → list_rooms handler → response
Latency: ~10ms
```

### Complex (with planning)
```
User: create room333 and add a tool that finds files with most 3s
→ is_complex_request() = True
→ plan_request(qwen3:4b) → ExecutionPlan
→ Display plan, await "y"
→ execute_plan() → handlers
Latency: ~500ms planning + execution
```

### Ambiguous (clarification)
```
User: add that tool to the room
→ is_complex_request() = True
→ plan_request() → clarifications: ["Which room?", "Which tool?"]
→ Display questions
→ User clarifies
→ Re-plan and execute
```

## Dependencies

- Existing refactored architecture (handlers, parser, context)
- Local Ollama with qwen3:4b model
- pydantic-ai for structured output
