"""Command parser for the System Architect room."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedCommand:
    """Parsed command with extracted arguments."""

    command: str
    args: dict[str, str]
    raw: str


def parse_command(prompt: str) -> ParsedCommand:
    """Parse user prompt into command and arguments.

    Simple keyword-based parser - no LLM overhead.
    """
    prompt = prompt.strip()
    lower = prompt.lower()

    # Exact matches first
    if lower == "rooms":
        return ParsedCommand("list_rooms", {}, prompt)

    if lower == "managed":
        return ParsedCommand("list_managed", {}, prompt)

    if lower in ("apply tool", "apply"):
        return ParsedCommand("apply_tool", {}, prompt)

    if lower in ("discard tool", "discard"):
        return ParsedCommand("discard_tool", {}, prompt)

    if lower == "list tools":
        return ParsedCommand("list_tools", {}, prompt)

    if lower == "list prompts":
        return ParsedCommand("list_prompts", {}, prompt)

    if lower == "list mcp":
        return ParsedCommand("list_mcp", {}, prompt)

    if lower == "list secrets":
        return ParsedCommand("list_secrets", {}, prompt)

    if lower in ("refresh", "refresh graph", "scan"):
        return ParsedCommand("refresh_graph", {}, prompt)

    # Pattern matches - order matters (more specific first)

    # Inspect room
    if match := re.match(r"inspect\s+(\S+)", lower):
        return ParsedCommand("inspect_room", {"room_id": match.group(1)}, prompt)

    # Show prompt
    if match := re.match(r"show prompt\s+(\S+)", lower):
        return ParsedCommand("show_prompt", {"name": match.group(1)}, prompt)

    # Remove prompt
    if match := re.match(r"remove prompt\s+(\S+)", lower):
        return ParsedCommand("remove_prompt", {"name": match.group(1)}, prompt)

    # Check secret
    if match := re.match(r"check secret\s+(\S+)", lower):
        return ParsedCommand("check_secret", {"name": match.group(1)}, prompt)

    # Find/search/query entity
    if match := re.match(r"(?:find|search|query)\s+(?:for\s+)?(.+)", lower):
        return ParsedCommand("find_entity", {"query": match.group(1).strip()}, prompt)

    # Show reference implementation
    if match := re.match(r"show\s+(joker|faux|brainstorm)", lower):
        return ParsedCommand("show_reference", {"name": match.group(1)}, prompt)

    # Room creation: "create X room" or "create X room for Y"
    if "create" in lower and "room" in lower:
        pattern = r"create\s+(.+?)\s+room(?:\s+(?:for|to|that)\s+(.+))?"
        if match := re.match(pattern, lower):
            name = match.group(1).strip()
            desc = match.group(2).strip() if match.group(2) else name
            return ParsedCommand(
                "create_room", {"name": name, "description": desc}, prompt
            )

    # Tool generation: "generate tool <room> <name> <description>"
    if lower.startswith("generate tool"):
        parts = prompt.split(maxsplit=4)
        if len(parts) >= 5:
            return ParsedCommand(
                "generate_tool",
                {
                    "room_id": parts[2],
                    "name": parts[3],
                    "description": parts[4],
                },
                prompt,
            )
        return ParsedCommand(
            "generate_tool_usage", {}, prompt
        )

    # Edit commands: "edit <room> <field> <value>"
    if lower.startswith("edit "):
        parts = prompt.split(maxsplit=3)
        if len(parts) >= 4:
            return ParsedCommand(
                "edit_room",
                {
                    "room_id": parts[1],
                    "field": parts[2].lower(),
                    "value": parts[3],
                },
                prompt,
            )
        return ParsedCommand("edit_room_usage", {}, prompt)

    # Add suggestion: "add suggestion <room> <text>"
    if match := re.match(r"add suggestion\s+(\S+)\s+(.+)", prompt, re.IGNORECASE):
        return ParsedCommand(
            "add_suggestion",
            {"room_id": match.group(1), "text": match.group(2)},
            prompt,
        )

    # Remove suggestion: "remove suggestion <room> <index>"
    if match := re.match(r"remove suggestion\s+(\S+)\s+(\d+)", lower):
        return ParsedCommand(
            "remove_suggestion",
            {"room_id": match.group(1), "index": match.group(2)},
            prompt,
        )

    # Add tool: "add tool <room> <tool-name>"
    if match := re.match(r"add tool\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand(
            "add_tool",
            {"room_id": match.group(1), "tool_name": match.group(2)},
            prompt,
        )

    # Remove tool: "remove tool <room> <tool-name>"
    if match := re.match(r"remove tool\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand(
            "remove_tool",
            {"room_id": match.group(1), "tool_name": match.group(2)},
            prompt,
        )

    # MCP commands
    if match := re.match(r"add mcp http\s+(\S+)\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand(
            "add_mcp_http",
            {
                "room_id": match.group(1),
                "name": match.group(2),
                "url": match.group(3),
            },
            prompt,
        )

    if match := re.match(r"add mcp stdio\s+(\S+)\s+(\S+)\s+(.+)", lower):
        return ParsedCommand(
            "add_mcp_stdio",
            {
                "room_id": match.group(1),
                "name": match.group(2),
                "command": match.group(3),
            },
            prompt,
        )

    if match := re.match(r"remove mcp\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand(
            "remove_mcp",
            {"room_id": match.group(1), "name": match.group(2)},
            prompt,
        )

    # MCP server mode
    if match := re.match(r"enable mcp-server\s+(\S+)", lower):
        return ParsedCommand(
            "enable_mcp_server", {"room_id": match.group(1)}, prompt
        )

    if match := re.match(r"disable mcp-server\s+(\S+)", lower):
        return ParsedCommand(
            "disable_mcp_server", {"room_id": match.group(1)}, prompt
        )

    # Prompt library - add prompt
    if match := re.match(r"add prompt\s+(\S+)\s+(.+)", prompt, re.IGNORECASE):
        return ParsedCommand(
            "add_prompt",
            {"name": match.group(1), "content": match.group(2)},
            prompt,
        )

    # Use prompt on room
    if match := re.match(r"use prompt\s+(\S+)\s+(\S+)", lower):
        return ParsedCommand(
            "use_prompt",
            {"room_id": match.group(1), "prompt_name": match.group(2)},
            prompt,
        )

    # Unknown - might be a tool refinement if pending tool exists
    return ParsedCommand("unknown", {"input": prompt}, prompt)
