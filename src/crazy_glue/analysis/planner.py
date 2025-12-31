"""Conversational planner for multi-step requests."""

from __future__ import annotations

import typing
from collections.abc import AsyncIterator

import pydantic
from pydantic_ai import Agent
from pydantic_ai import messages as ai_messages

from crazy_glue.analysis.handlers import HANDLERS
from crazy_glue.analysis.parser import parse_command

if typing.TYPE_CHECKING:
    from crazy_glue.analysis.context import AnalysisContext

# Verbs that indicate actions
ACTION_VERBS = frozenset([
    "create", "generate", "add", "remove", "edit", "change",
    "update", "set", "delete", "make", "build", "configure",
])

# Conjunctions that suggest multiple tasks
CONJUNCTIONS = [" and ", " then ", ", ", " also ", " plus ", " after that "]


class PlannedCommand(pydantic.BaseModel):
    """A single planned command."""

    command: str = pydantic.Field(
        description="The exact command string to execute"
    )
    description: str = pydantic.Field(
        description="Brief human-readable description of what this does"
    )
    requires_confirm: bool = pydantic.Field(
        default=False,
        description="True if this is a destructive or code-generating operation",
    )


class ExecutionPlan(pydantic.BaseModel):
    """Plan output from LLM planner."""

    commands: list[PlannedCommand] = pydantic.Field(
        description="Ordered list of commands to execute"
    )
    clarifications: list[str] = pydantic.Field(
        default_factory=list,
        description="Questions to ask if request is ambiguous",
    )
    warnings: list[str] = pydantic.Field(
        default_factory=list,
        description="Potential issues or notes for the user",
    )


PLANNER_PROMPT = """You are a command planner for a room management system.

Convert the user's natural language request into a sequence of commands.

## Available Commands

**Room Management:**
- `rooms` - List all rooms
- `managed` - List managed rooms (created by architect)
- `inspect <room-id>` - Show room details
- `create <name> room` or `create <name> room for <description>` - Create new room

**Room Editing (managed rooms only):**
- `edit <room-id> description <text>` - Update description
- `edit <room-id> prompt <text>` - Update system prompt
- `edit <room-id> model <name>` - Change LLM model
- `edit <room-id> welcome <text>` - Update welcome message
- `add suggestion <room-id> <text>` - Add a suggestion chip
- `remove suggestion <room-id> <index>` - Remove suggestion by index

**Tool Management:**
- `list tools` - Show available predefined tools
- `generate tool <room-id> <name> <description>` - Generate custom tool with LLM
- `apply tool` - Apply staged tool (after generate)
- `discard tool` - Discard staged tool
- `add tool <room-id> <tool-name>` - Add predefined tool
- `remove tool <room-id> <tool-name>` - Remove tool

**MCP Toolsets:**
- `list mcp` - Show MCP options
- `add mcp http <room-id> <name> <url>` - Add HTTP MCP
- `add mcp stdio <room-id> <name> <command>` - Add stdio MCP
- `remove mcp <room-id> <name>` - Remove MCP toolset
- `enable mcp-server <room-id>` - Enable MCP server mode
- `disable mcp-server <room-id>` - Disable MCP server mode

**Other:**
- `list secrets` - Show configured secrets
- `check secret <name>` - Check if secret is configured
- `list prompts` - Show saved prompts
- `show prompt <name>` - Display prompt content
- `add prompt <name> <content>` - Save a prompt
- `remove prompt <name>` - Delete a prompt
- `use prompt <room-id> <prompt-name>` - Apply prompt to room
- `refresh` - Rebuild knowledge graph
- `find <query>` - Search codebase entities

## Rules

1. Output commands in logical order (create before edit)
2. Use exact command syntax from above
3. Room IDs are slugified: "My Room" → "my-room"
4. Mark `requires_confirm: true` for: generate tool, remove operations
5. If ambiguous (which room? what exactly?), add clarification questions
6. Add warnings for potentially slow operations (tool generation)

## Examples

User: "create a calculator room and add a compute tool"
→ commands:
  1. `create calculator room` (description: "Create calculator room")
  2. `generate tool calculator compute evaluate math` (requires_confirm: true)
→ warnings: ["Tool generation may take a few seconds"]

User: "add suggestions to my-room"
→ clarifications: ["What suggestions would you like to add?"]

User: "change the description"
→ clarifications: ["Which room should I update?", "What should the new description be?"]
"""


def is_complex_request(prompt: str) -> bool:
    """Detect if request needs LLM planning vs keyword parsing."""
    prompt_lower = prompt.lower()

    # Count action verbs
    verb_count = sum(1 for v in ACTION_VERBS if v in prompt_lower)
    if verb_count > 1:
        return True

    # Check for conjunctions suggesting multiple tasks
    if any(c in prompt_lower for c in CONJUNCTIONS):
        return True

    # Pronouns referencing context (ambiguous without planning)
    context_refs = [" it ", " that ", " the room", " this ", " them "]
    if any(ref in prompt_lower for ref in context_refs):
        return True

    # Long requests are likely complex
    return len(prompt.split()) > 15


def _get_planner_model(ctx: AnalysisContext):
    """Get fast local model for planning."""
    from pydantic_ai.models import openai as openai_models
    from pydantic_ai.providers import ollama as ollama_providers

    base_url = ctx.installation_config.get_environment("OLLAMA_BASE_URL")
    if not base_url:
        base_url = "http://localhost:11434"

    provider = ollama_providers.OllamaProvider(
        base_url=f"{base_url}/v1",
    )

    # Use fast model for planning
    extra = getattr(ctx.agent_config, "extra_config", {}) or {}
    model_name = extra.get("planner_model", "qwen3:4b")

    return openai_models.OpenAIChatModel(
        model_name=model_name,
        provider=provider,
    )


async def plan_request(ctx: AnalysisContext, prompt: str) -> ExecutionPlan:
    """Plan a complex request into executable commands."""
    model = _get_planner_model(ctx)

    agent: Agent[None, ExecutionPlan] = Agent(
        model=model,
        output_type=ExecutionPlan,
        system_prompt=PLANNER_PROMPT,
    )

    result = await agent.run(prompt)
    return result.output


async def execute_plan(
    ctx: AnalysisContext,
    plan: ExecutionPlan,
) -> AsyncIterator[ai_messages.AgentStreamEvent]:
    """Execute a confirmed plan step by step."""
    from crazy_glue.analysis.handlers import handle_unknown

    total = len(plan.commands)

    for i, planned in enumerate(plan.commands, 1):
        # Step header
        header = f"\n**Step {i}/{total}**: {planned.description}\n"
        async for event in _yield_text(header):
            yield event

        # Show command being executed
        async for event in _yield_text(f"`{planned.command}`\n\n"):
            yield event

        # Parse and execute
        cmd = parse_command(planned.command)
        handler = HANDLERS.get(cmd.command, handle_unknown)

        # Debug: show parsed command
        if cmd.command == "unknown":
            warn = f"⚠️ Command not recognized: `{planned.command}`\n"
            async for event in _yield_text(warn):
                yield event
            continue

        async for event in handler(ctx, cmd):
            yield event

        # For tool generation, check if we need to apply
        if cmd.command == "generate_tool":
            pending = ctx.load_pending_tool()
            if pending:
                # Auto-apply for planned execution
                from crazy_glue.analysis.tools import apply_pending_tool

                apply_result = apply_pending_tool(ctx)
                status = apply_result.get("status", "")
                msg = apply_result.get("message", "")
                if status == "success":
                    async for event in _yield_text(f"✓ {msg}\n"):
                        yield event
                else:
                    async for event in _yield_text(f"⚠️ {msg}\n"):
                        yield event

    # Final summary
    async for event in _yield_text(f"\n✓ **Plan complete** ({total} steps)\n"):
        yield event
    async for event in _yield_text("Restart soliplex to apply changes.\n"):
        yield event


async def _yield_text(text: str):
    """Helper to yield text events."""
    text_part = ai_messages.TextPart(text)
    yield ai_messages.PartStartEvent(index=0, part=text_part)
    yield ai_messages.PartEndEvent(index=0, part=text_part)
