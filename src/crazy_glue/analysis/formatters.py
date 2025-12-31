"""Output formatting for the System Architect room."""

from __future__ import annotations


def format_room_list(rooms: list[dict], managed: bool = False) -> str:
    """Format list of rooms for display."""
    lines = []

    if managed:
        lines.append("## Managed Rooms\n")
        if rooms:
            lines.append(f"**Total**: {len(rooms)} managed rooms\n\n")
            for r in rooms:
                desc = r["description"][:50] if r["description"] else ""
                lines.append(f"- **{r['id']}**: {desc}\n")
                if r.get("config_path"):
                    lines.append(f"  - `{r['config_path']}`\n")
            note = "Only managed rooms can be modified by the architect."
            lines.append(f"\n*{note}*\n")
        else:
            lines.append("No managed rooms found.\n\n")
            hint = "Create a room with `create <name> room` to manage it."
            lines.append(f"*{hint}*\n")
    else:
        lines.append("## Registered Rooms\n\n")
        lines.append(f"**Total**: {len(rooms)} rooms\n\n")

        factories = [r for r in rooms if r["agent_kind"] == "factory"]
        defaults = [r for r in rooms if r["agent_kind"] != "factory"]

        if factories:
            lines.append("### Factory Agents\n\n")
            for r in factories:
                desc = r["description"][:50] if r["description"] else ""
                lines.append(f"- **{r['id']}**: {desc}\n")
            lines.append("\n")

        if defaults:
            lines.append("### Default Agents\n\n")
            for r in defaults:
                desc = r["description"][:50] if r["description"] else ""
                lines.append(f"- **{r['id']}**: {desc}\n")
            lines.append("\n")

        lines.append("Use `inspect <room-id>` for details.\n")

    return "".join(lines)


def format_room_details(info: dict) -> str:
    """Format detailed room information."""
    lines = []
    lines.append(f"## Room: {info['name']}\n\n")
    lines.append(f"**ID**: `{info['id']}`\n")
    lines.append(f"**Description**: {info.get('description', '')}\n")
    lines.append(f"**Config**: `{info.get('config_path', '')}`\n\n")

    agent = info.get("agent", {})
    lines.append("### Agent\n\n")
    lines.append(f"- **Kind**: {agent.get('kind', 'unknown')}\n")
    if agent.get("factory_name"):
        lines.append(f"- **Factory**: `{agent['factory_name']}`\n")
    if agent.get("model_name"):
        lines.append(f"- **Model**: {agent['model_name']}\n")
    if agent.get("extra_config"):
        lines.append("- **Extra Config**:\n")
        for k, v in agent["extra_config"].items():
            lines.append(f"  - `{k}`: {v}\n")
    lines.append("\n")

    if info.get("suggestions"):
        lines.append("### Suggestions\n\n")
        for i, s in enumerate(info["suggestions"]):
            lines.append(f"- [{i}] {s}\n")
        lines.append("\n")

    if info.get("tools"):
        lines.append("### Tools\n\n")
        for t in info["tools"]:
            lines.append(f"- `{t}`\n")
        lines.append("\n")

    if info.get("mcp_toolsets"):
        lines.append("### MCP Toolsets\n\n")
        for m in info["mcp_toolsets"]:
            lines.append(f"- `{m}`\n")
        lines.append("\n")

    return "".join(lines)


def format_tool_list(tools: list[dict]) -> str:
    """Format list of available tools."""
    lines = ["## Available Tools\n\n"]
    for t in tools:
        lines.append(f"### `{t['name']}`\n")
        lines.append(f"- {t['description']}\n")
        lines.append(f"- Requires: {t['requires']}\n\n")
    return "".join(lines)


def format_staged_tool(result: dict) -> str:
    """Format staged tool preview."""
    lines = ["## Tool Generated (Staged)\n\n"]
    if result.get("message"):
        lines.append(f"{result['message']}\n\n")
    lines.append(f"**Description**: {result.get('description', '')}\n")
    lines.append(f"**File**: `{result.get('file_path', '')}`\n\n")
    lines.append("**Preview:**\n")
    lines.append(f"```python\n{result.get('code', '')}```\n\n")
    lines.append("**Commands:**\n")
    lines.append("- `apply tool` - Save and add to room\n")
    lines.append("- `discard tool` - Discard\n")
    lines.append("- Or describe changes to refine\n")
    return "".join(lines)


def format_refined_tool(result: dict, pending: dict) -> str:
    """Format refined tool preview."""
    lines = ["## Tool Refined\n\n"]
    lines.append(f"**Name**: `{pending.get('name', '')}`\n")
    lines.append(f"**Room**: `{pending.get('room_id', '')}`\n")
    lines.append(f"**Description**: {result.get('description', '')}\n\n")
    lines.append("**Updated Preview:**\n")
    lines.append(f"```python\n{result.get('code', '')}```\n\n")
    lines.append("**Commands:**\n")
    lines.append("- `apply tool` - Save and add to room\n")
    lines.append("- `discard tool` - Discard\n")
    lines.append("- Or describe more changes\n")
    return "".join(lines)


def format_graph_refresh(result: dict) -> str:
    """Format knowledge graph refresh result."""
    lines = ["## Knowledge Graph Refreshed\n\n"]
    lines.append(f"- **Map Path**: {result['map_path']}\n")
    lines.append(f"- **Total Entities**: {result['total_entities']}\n")
    lines.append(f"- **Total Links**: {result['total_links']}\n\n")
    for root_info in result.get("roots_scanned", []):
        lines.append(f"### {root_info['root']}\n")
        lines.append(f"- Entities: {root_info['entities']}\n")
        lines.append(f"- Files: {root_info['files_processed']}\n\n")
    return "".join(lines)


def format_search_results(results: list[dict], query: str) -> str:
    """Format graph search results."""
    lines = [f"## Search Results for '{query}'\n\n"]

    if results and "error" not in results[0]:
        for r in results:
            loc = f"{r['location']}:{r.get('line', '?')}"
            lines.append(f"### {r['name']} ({r['type']})\n")
            lines.append(f"- **Location**: `{loc}`\n")
            lines.append(f"- **Summary**: {r['summary']}\n")
            lines.append(f"- **ID**: `{r['id']}`\n\n")
    else:
        lines.append("No results found.\n")

    return "".join(lines)


def format_secrets_list(secrets: list[dict]) -> str:
    """Format list of secrets."""
    lines = ["## Configured Secrets\n\n"]
    if secrets:
        for s in secrets:
            status = "resolved" if s["resolved"] else "NOT resolved"
            lines.append(f"- `{s['name']}`: {status}\n")
            sources = ", ".join(s["sources"])
            lines.append(f"  - Sources: {sources}\n")
    else:
        lines.append("No secrets configured.\n")
    return "".join(lines)


def format_secret_check(info: dict) -> str:
    """Format secret check result."""
    lines = [f"## Secret: {info['name']}\n\n"]
    if info["configured"]:
        status = "resolved" if info["resolved"] else "NOT resolved"
        lines.append(f"**Status**: {status}\n")
        sources = ", ".join(info["sources"])
        lines.append(f"**Sources**: {sources}\n")
    else:
        lines.append("**Not configured**\n\n")
        lines.append("Add to installation.yaml secrets section.\n")
    return "".join(lines)


def format_prompts_list(prompts: list[dict]) -> str:
    """Format list of prompts."""
    lines = ["## Prompt Library\n\n"]
    if prompts:
        for p in prompts:
            lines.append(f"### `{p['id']}`\n")
            lines.append(f"**Name**: {p['name']}\n")
            lines.append(f"**Preview**: {p['preview']}\n\n")
    else:
        lines.append("No prompts saved yet.\n\n")
        lines.append("Use `add prompt <name>` to create one.\n")
    return "".join(lines)


def format_prompt_detail(prompt: dict) -> str:
    """Format a single prompt's details."""
    lines = [f"## Prompt: {prompt['name']}\n\n"]
    lines.append("**Content:**\n")
    lines.append(f"```\n{prompt['content']}\n```\n")
    return "".join(lines)


def format_mcp_help() -> str:
    """Format MCP toolset help."""
    lines = ["## MCP Client Toolsets\n\n"]
    lines.append("**HTTP MCP** (`kind: http`)\n")
    lines.append("- Connects to remote MCP servers\n")
    lines.append("- `add mcp http <room> <name> <url>`\n\n")
    lines.append("**Stdio MCP** (`kind: stdio`)\n")
    lines.append("- Runs local MCP server as subprocess\n")
    lines.append("- `add mcp stdio <room> <name> <cmd>`\n")
    return "".join(lines)


def format_help() -> str:
    """Format help message with all available commands."""
    lines = ["## System Architect\n\n"]
    lines.append("**Introspection:**\n")
    lines.append("- `rooms` - List registered rooms\n")
    lines.append("- `managed` - List managed rooms\n")
    lines.append("- `inspect <id>` - Inspect room\n\n")
    lines.append("**Room Management:**\n")
    lines.append("- `create <name> room` - Create room\n")
    lines.append("- `edit <id> <field> <val>`\n")
    lines.append("- `add suggestion <id> <text>`\n")
    lines.append("- `remove suggestion <id> <idx>`\n\n")
    lines.append("**Tool Generation:**\n")
    lines.append("- `generate tool <id> <name> <desc>`\n")
    lines.append("- `apply tool` - Apply staged tool\n")
    lines.append("- `discard tool` - Discard staged\n\n")
    lines.append("**Tool Management:**\n")
    lines.append("- `list tools` - Show available\n")
    lines.append("- `add tool <id> <tool-name>`\n")
    lines.append("- `remove tool <id> <tool-name>`\n\n")
    lines.append("**MCP Toolsets:**\n")
    lines.append("- `list mcp` - Show MCP options\n")
    lines.append("- `add mcp http <id> <name> <url>`\n")
    lines.append("- `add mcp stdio <id> <name> <cmd>`\n")
    lines.append("- `remove mcp <id> <name>`\n\n")
    lines.append("**Secrets:**\n")
    lines.append("- `list secrets` - Show secrets\n")
    lines.append("- `check secret <name>`\n\n")
    lines.append("**Prompt Library:**\n")
    lines.append("- `list prompts` - Show saved\n")
    lines.append("- `show prompt <name>`\n")
    lines.append("- `add prompt <name> <content>`\n")
    lines.append("- `remove prompt <name>`\n")
    lines.append("- `use prompt <id> <name>`\n\n")
    lines.append("**MCP Server Mode:**\n")
    lines.append("- `enable mcp-server <id>`\n")
    lines.append("- `disable mcp-server <id>`\n\n")
    lines.append("**Codebase Analysis:**\n")
    lines.append("- `refresh` - Build knowledge graph\n")
    lines.append("- `find <name>` - Search entities\n")
    lines.append("- `show joker|faux|brainstorm`\n")
    return "".join(lines)


def format_error(message: str) -> str:
    """Format error message."""
    return f"## Error\n\n{message}\n"


def format_success(message: str) -> str:
    """Format success message."""
    return f"{message}\n\nRestart soliplex to apply changes.\n"


def format_warning(message: str) -> str:
    """Format warning message."""
    return f"## Warning\n\n{message}\n"


def format_plan(plan) -> str:
    """Format execution plan for display."""
    from crazy_glue.analysis.planner import ExecutionPlan

    if isinstance(plan, dict):
        # Convert dict to ExecutionPlan if needed
        plan = ExecutionPlan(**plan)

    lines = ["## Execution Plan\n\n"]
    lines.append("I'll execute these steps:\n\n")

    for i, cmd in enumerate(plan.commands, 1):
        confirm_marker = " ⚠️" if cmd.requires_confirm else ""
        lines.append(f"{i}. **{cmd.description}**{confirm_marker}\n")
        lines.append(f"   `{cmd.command}`\n\n")

    if plan.warnings:
        lines.append("**Notes:**\n")
        for w in plan.warnings:
            lines.append(f"- {w}\n")
        lines.append("\n")

    lines.append("Proceed? Reply **y** to execute, or describe changes.\n")

    return "".join(lines)


def format_clarifications(plan) -> str:
    """Format clarification questions from planner."""
    from crazy_glue.analysis.planner import ExecutionPlan

    if isinstance(plan, dict):
        plan = ExecutionPlan(**plan)

    lines = ["## Clarification Needed\n\n"]
    lines.append("I need more information:\n\n")

    for i, q in enumerate(plan.clarifications, 1):
        lines.append(f"{i}. {q}\n")

    lines.append("\nPlease clarify and I'll create the plan.\n")

    return "".join(lines)
