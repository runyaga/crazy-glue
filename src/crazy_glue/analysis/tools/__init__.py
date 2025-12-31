"""Tool operations for the System Architect room."""

from crazy_glue.analysis.tools.graph_ops import query_graph
from crazy_glue.analysis.tools.graph_ops import read_entity_source
from crazy_glue.analysis.tools.graph_ops import read_reference_implementation
from crazy_glue.analysis.tools.graph_ops import refresh_knowledge_graph
from crazy_glue.analysis.tools.mcp_ops import check_secret
from crazy_glue.analysis.tools.mcp_ops import list_secrets
from crazy_glue.analysis.tools.mcp_ops import manage_mcp
from crazy_glue.analysis.tools.mcp_ops import toggle_mcp_server
from crazy_glue.analysis.tools.prompt_ops import add_prompt
from crazy_glue.analysis.tools.prompt_ops import get_prompt
from crazy_glue.analysis.tools.prompt_ops import list_prompts
from crazy_glue.analysis.tools.prompt_ops import remove_prompt
from crazy_glue.analysis.tools.prompt_ops import use_prompt
from crazy_glue.analysis.tools.room_ops import create_room
from crazy_glue.analysis.tools.room_ops import edit_room
from crazy_glue.analysis.tools.room_ops import inspect_room
from crazy_glue.analysis.tools.room_ops import list_managed_rooms
from crazy_glue.analysis.tools.room_ops import list_rooms
from crazy_glue.analysis.tools.room_ops import manage_suggestion
from crazy_glue.analysis.tools.tool_ops import apply_pending_tool
from crazy_glue.analysis.tools.tool_ops import discard_pending_tool
from crazy_glue.analysis.tools.tool_ops import generate_tool
from crazy_glue.analysis.tools.tool_ops import list_available_tools
from crazy_glue.analysis.tools.tool_ops import manage_tool
from crazy_glue.analysis.tools.tool_ops import refine_pending_tool

__all__ = [
    # Room ops
    "list_rooms",
    "list_managed_rooms",
    "inspect_room",
    "create_room",
    "edit_room",
    "manage_suggestion",
    # Tool ops
    "list_available_tools",
    "generate_tool",
    "apply_pending_tool",
    "discard_pending_tool",
    "refine_pending_tool",
    "manage_tool",
    # MCP ops
    "list_secrets",
    "check_secret",
    "manage_mcp",
    "toggle_mcp_server",
    # Prompt ops
    "list_prompts",
    "get_prompt",
    "add_prompt",
    "remove_prompt",
    "use_prompt",
    # Graph ops
    "refresh_knowledge_graph",
    "query_graph",
    "read_entity_source",
    "read_reference_implementation",
]
