"""Command handlers for the System Architect room."""

from __future__ import annotations

import typing
from collections.abc import AsyncIterator
from collections.abc import Callable

from pydantic_ai import messages as ai_messages

from crazy_glue.analysis import formatters
from crazy_glue.analysis.parser import ParsedCommand
from crazy_glue.analysis.tools import add_prompt
from crazy_glue.analysis.tools import apply_pending_tool
from crazy_glue.analysis.tools import check_secret
from crazy_glue.analysis.tools import create_room
from crazy_glue.analysis.tools import discard_pending_tool
from crazy_glue.analysis.tools import edit_room
from crazy_glue.analysis.tools import generate_tool
from crazy_glue.analysis.tools import get_prompt
from crazy_glue.analysis.tools import inspect_room
from crazy_glue.analysis.tools import list_available_tools
from crazy_glue.analysis.tools import list_managed_rooms
from crazy_glue.analysis.tools import list_prompts
from crazy_glue.analysis.tools import list_rooms
from crazy_glue.analysis.tools import list_secrets
from crazy_glue.analysis.tools import manage_mcp
from crazy_glue.analysis.tools import manage_suggestion
from crazy_glue.analysis.tools import manage_tool
from crazy_glue.analysis.tools import query_graph
from crazy_glue.analysis.tools import read_reference_implementation
from crazy_glue.analysis.tools import refine_pending_tool
from crazy_glue.analysis.tools import refresh_knowledge_graph
from crazy_glue.analysis.tools import remove_prompt
from crazy_glue.analysis.tools import toggle_mcp_server
from crazy_glue.analysis.tools import use_prompt

if typing.TYPE_CHECKING:
    from crazy_glue.analysis.context import AnalysisContext

NativeEvent = ai_messages.AgentStreamEvent

Handler = Callable[
    ["AnalysisContext", ParsedCommand],
    AsyncIterator[NativeEvent],
]

HANDLERS: dict[str, Handler] = {}


def handler(name: str):
    """Decorator to register a command handler."""

    def decorator(fn: Handler) -> Handler:
        HANDLERS[name] = fn
        return fn

    return decorator


async def _yield_text(text: str):
    """Helper to yield a text response."""
    text_part = ai_messages.TextPart(text)
    yield ai_messages.PartStartEvent(index=0, part=text_part)
    yield ai_messages.PartEndEvent(index=0, part=text_part)


async def _yield_result(result: dict, fmt_success, fmt_error, fmt_warning=None):
    """Helper to format and yield based on result status."""
    status = result.get("status", "")
    msg = result.get("message", "")

    if status == "success":
        text = fmt_success(msg) if callable(fmt_success) else fmt_success
    elif status == "warning" and fmt_warning:
        text = formatters.format_warning(msg)
    elif "error" in result:
        text = formatters.format_error(result["error"])
    else:
        text = formatters.format_error(msg)

    async for event in _yield_text(text):
        yield event


# --- Room Handlers ---


@handler("list_rooms")
async def handle_list_rooms(ctx: AnalysisContext, cmd: ParsedCommand):
    """List all registered rooms."""
    rooms = list_rooms(ctx)
    async for event in _yield_text(formatters.format_room_list(rooms)):
        yield event


@handler("list_managed")
async def handle_list_managed(ctx: AnalysisContext, cmd: ParsedCommand):
    """List managed rooms."""
    rooms = list_managed_rooms(ctx)
    text = formatters.format_room_list(rooms, managed=True)
    async for event in _yield_text(text):
        yield event


@handler("inspect_room")
async def handle_inspect_room(ctx: AnalysisContext, cmd: ParsedCommand):
    """Inspect a room's configuration."""
    result = inspect_room(ctx, cmd.args["room_id"])
    if "error" in result:
        async for event in _yield_text(formatters.format_error(result["error"])):
            yield event
    else:
        async for event in _yield_text(formatters.format_room_details(result)):
            yield event


@handler("create_room")
async def handle_create_room(ctx: AnalysisContext, cmd: ParsedCommand):
    """Create a new room."""
    result = await create_room(
        ctx, cmd.args["name"], cmd.args["description"]
    )
    if result.get("error"):
        async for event in _yield_text(formatters.format_error(result["error"])):
            yield event
    else:
        msg = f"Created room `{result['room_id']}` at `{result['path']}`"
        async for event in _yield_text(formatters.format_success(msg)):
            yield event


@handler("edit_room")
async def handle_edit_room(ctx: AnalysisContext, cmd: ParsedCommand):
    """Edit a room field."""
    result = edit_room(
        ctx, cmd.args["room_id"], cmd.args["field"], cmd.args["value"]
    )
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("edit_room_usage")
async def handle_edit_room_usage(ctx: AnalysisContext, cmd: ParsedCommand):
    """Show edit room usage."""
    text = (
        "## Edit Room\n\n"
        "Usage: `edit <room-id> <field> <value>`\n\n"
        "**Fields:**\n"
        "- `description` - Room description\n"
        "- `prompt` - System prompt\n"
        "- `model` - LLM model name\n"
        "- `welcome` - Welcome message\n"
    )
    async for event in _yield_text(text):
        yield event


@handler("add_suggestion")
async def handle_add_suggestion(ctx: AnalysisContext, cmd: ParsedCommand):
    """Add a suggestion to a room."""
    result = manage_suggestion(
        ctx, cmd.args["room_id"], "add", cmd.args["text"]
    )
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("remove_suggestion")
async def handle_remove_suggestion(ctx: AnalysisContext, cmd: ParsedCommand):
    """Remove a suggestion from a room."""
    result = manage_suggestion(
        ctx, cmd.args["room_id"], "remove", cmd.args["index"]
    )
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


# --- Tool Handlers ---


@handler("list_tools")
async def handle_list_tools(ctx: AnalysisContext, cmd: ParsedCommand):
    """List available tools."""
    tools = list_available_tools()
    async for event in _yield_text(formatters.format_tool_list(tools)):
        yield event


@handler("generate_tool")
async def handle_generate_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Generate a custom tool."""
    result = await generate_tool(
        ctx,
        cmd.args["room_id"],
        cmd.args["name"],
        cmd.args["description"],
    )
    if result.get("status") == "error":
        async for event in _yield_text(formatters.format_error(result["message"])):
            yield event
    else:
        async for event in _yield_text(formatters.format_staged_tool(result)):
            yield event


@handler("generate_tool_usage")
async def handle_generate_tool_usage(ctx: AnalysisContext, cmd: ParsedCommand):
    """Show generate tool usage."""
    text = (
        "## Generate Tool\n\n"
        "Usage: `generate tool <room-id> <name> <desc>`\n\n"
        "**Example:**\n"
        "`generate tool my-room list_files List directory contents`\n"
    )
    async for event in _yield_text(text):
        yield event


@handler("apply_tool")
async def handle_apply_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Apply staged tool."""
    pending = ctx.load_pending_tool()
    if not pending:
        text = (
            "## No Pending Tool\n\n"
            "Generate a tool first with:\n"
            "`generate tool <room-id> <name> <desc>`\n"
        )
        async for event in _yield_text(text):
            yield event
        return

    result = apply_pending_tool(ctx)
    msg = result.get("message", "")
    if result["status"] == "success":
        text = (
            f"## Tool Applied\n\n"
            f"{msg}\n\n"
            f"**File**: `{result['file_path']}`\n\n"
            "Restart soliplex to load the tool.\n"
        )
        async for event in _yield_text(text):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("discard_tool")
async def handle_discard_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Discard staged tool."""
    pending = ctx.load_pending_tool()
    if not pending:
        async for event in _yield_text("## No Pending Tool\n\nNothing to discard.\n"):
            yield event
        return

    result = discard_pending_tool(ctx)
    text = f"## Tool Discarded\n\n{result['message']}\n"
    async for event in _yield_text(text):
        yield event


@handler("add_tool")
async def handle_add_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Add a tool to a room."""
    result = manage_tool(
        ctx, cmd.args["room_id"], "add", cmd.args["tool_name"]
    )
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("remove_tool")
async def handle_remove_tool(ctx: AnalysisContext, cmd: ParsedCommand):
    """Remove a tool from a room."""
    result = manage_tool(
        ctx, cmd.args["room_id"], "remove", cmd.args["tool_name"]
    )
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(msg):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


# --- MCP Handlers ---


@handler("list_mcp")
async def handle_list_mcp(ctx: AnalysisContext, cmd: ParsedCommand):
    """List MCP toolset options."""
    async for event in _yield_text(formatters.format_mcp_help()):
        yield event


@handler("add_mcp_http")
async def handle_add_mcp_http(ctx: AnalysisContext, cmd: ParsedCommand):
    """Add HTTP MCP toolset."""
    result = manage_mcp(
        ctx,
        cmd.args["room_id"],
        "add",
        cmd.args["name"],
        kind="http",
        url=cmd.args["url"],
    )
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("add_mcp_stdio")
async def handle_add_mcp_stdio(ctx: AnalysisContext, cmd: ParsedCommand):
    """Add stdio MCP toolset."""
    result = manage_mcp(
        ctx,
        cmd.args["room_id"],
        "add",
        cmd.args["name"],
        kind="stdio",
        command=cmd.args["command"],
    )
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("remove_mcp")
async def handle_remove_mcp(ctx: AnalysisContext, cmd: ParsedCommand):
    """Remove MCP toolset."""
    result = manage_mcp(ctx, cmd.args["room_id"], "remove", cmd.args["name"])
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(msg):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("enable_mcp_server")
async def handle_enable_mcp_server(ctx: AnalysisContext, cmd: ParsedCommand):
    """Enable MCP server mode."""
    result = toggle_mcp_server(ctx, cmd.args["room_id"], enable=True)
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


@handler("disable_mcp_server")
async def handle_disable_mcp_server(ctx: AnalysisContext, cmd: ParsedCommand):
    """Disable MCP server mode."""
    result = toggle_mcp_server(ctx, cmd.args["room_id"], enable=False)
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


# --- Secret Handlers ---


@handler("list_secrets")
async def handle_list_secrets(ctx: AnalysisContext, cmd: ParsedCommand):
    """List configured secrets."""
    secrets = list_secrets(ctx)
    async for event in _yield_text(formatters.format_secrets_list(secrets)):
        yield event


@handler("check_secret")
async def handle_check_secret(ctx: AnalysisContext, cmd: ParsedCommand):
    """Check a secret's status."""
    info = check_secret(ctx, cmd.args["name"])
    async for event in _yield_text(formatters.format_secret_check(info)):
        yield event


# --- Prompt Handlers ---


@handler("list_prompts")
async def handle_list_prompts(ctx: AnalysisContext, cmd: ParsedCommand):
    """List saved prompts."""
    prompts = list_prompts(ctx)
    async for event in _yield_text(formatters.format_prompts_list(prompts)):
        yield event


@handler("show_prompt")
async def handle_show_prompt(ctx: AnalysisContext, cmd: ParsedCommand):
    """Show a prompt's content."""
    prompt = get_prompt(ctx, cmd.args["name"])
    if prompt:
        async for event in _yield_text(formatters.format_prompt_detail(prompt)):
            yield event
    else:
        text = formatters.format_error(f"Prompt '{cmd.args['name']}' not found.")
        async for event in _yield_text(text):
            yield event


@handler("add_prompt")
async def handle_add_prompt(ctx: AnalysisContext, cmd: ParsedCommand):
    """Add a prompt to the library."""
    result = add_prompt(ctx, cmd.args["name"], cmd.args["content"])
    if result["status"] == "success":
        text = f"## Prompt Saved\n\n{result['message']}\n"
        async for event in _yield_text(text):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(result["message"])):
            yield event


@handler("remove_prompt")
async def handle_remove_prompt(ctx: AnalysisContext, cmd: ParsedCommand):
    """Remove a prompt from the library."""
    result = remove_prompt(ctx, cmd.args["name"])
    if result["status"] == "success":
        text = f"## Prompt Removed\n\n{result['message']}\n"
        async for event in _yield_text(text):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(result["message"])):
            yield event


@handler("use_prompt")
async def handle_use_prompt(ctx: AnalysisContext, cmd: ParsedCommand):
    """Apply a prompt to a room."""
    result = use_prompt(ctx, cmd.args["room_id"], cmd.args["prompt_name"])
    msg = result.get("message", "")
    if result["status"] == "success":
        async for event in _yield_text(formatters.format_success(msg)):
            yield event
    elif result["status"] == "warning":
        async for event in _yield_text(formatters.format_warning(msg)):
            yield event
    else:
        async for event in _yield_text(formatters.format_error(msg)):
            yield event


# --- Graph Handlers ---


@handler("refresh_graph")
async def handle_refresh_graph(ctx: AnalysisContext, cmd: ParsedCommand):
    """Refresh the knowledge graph."""
    result = await refresh_knowledge_graph(ctx)
    async for event in _yield_text(formatters.format_graph_refresh(result)):
        yield event


@handler("find_entity")
async def handle_find_entity(ctx: AnalysisContext, cmd: ParsedCommand):
    """Find entities in the knowledge graph."""
    query = cmd.args["query"]
    results = query_graph(ctx, query)
    text = formatters.format_search_results(results, query)
    async for event in _yield_text(text):
        yield event


@handler("show_reference")
async def handle_show_reference(ctx: AnalysisContext, cmd: ParsedCommand):
    """Show a reference implementation."""
    name = cmd.args["name"]
    source = read_reference_implementation(ctx, name)
    text = f"## {name.title()} Reference\n\n```python\n{source}\n```\n"
    async for event in _yield_text(text):
        yield event


# --- Unknown/Fallback ---


@handler("unknown")
async def handle_unknown(ctx: AnalysisContext, cmd: ParsedCommand):
    """Handle unknown commands or tool refinement."""
    pending = ctx.load_pending_tool()
    if pending and cmd.args.get("input"):
        result = await refine_pending_tool(ctx, cmd.args["input"])
        if result["status"] == "staged":
            text = formatters.format_refined_tool(result, pending)
            async for event in _yield_text(text):
                yield event
        else:
            async for event in _yield_text(formatters.format_error(result["message"])):
                yield event
    else:
        async for event in _yield_text(formatters.format_help()):
            yield event
