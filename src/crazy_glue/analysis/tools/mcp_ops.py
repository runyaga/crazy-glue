"""MCP and secret operations for the System Architect."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from crazy_glue.analysis.context import AnalysisContext


def list_secrets(ctx: AnalysisContext) -> list[dict]:
    """List configured secrets (names only, not values)."""
    secrets = []
    for secret in ctx.installation_config.secrets:
        resolved = secret._resolved is not None
        secrets.append({
            "name": secret.secret_name,
            "resolved": resolved,
            "sources": [type(s).__name__ for s in secret.sources],
        })
    return secrets


def check_secret(ctx: AnalysisContext, name: str) -> dict:
    """Check if a secret is configured and resolved."""
    for secret in ctx.installation_config.secrets:
        if secret.secret_name == name:
            resolved = secret._resolved is not None
            return {
                "name": name,
                "configured": True,
                "resolved": resolved,
                "sources": [type(s).__name__ for s in secret.sources],
            }
    return {
        "name": name,
        "configured": False,
        "resolved": False,
        "sources": [],
    }


def manage_mcp(
    ctx: AnalysisContext,
    room_id: str,
    action: str,
    name: str,
    kind: str | None = None,
    url: str | None = None,
    command: str | None = None,
) -> dict:
    """Add or remove an MCP client toolset from a room."""
    from crazy_glue.analysis.tools.room_ops import get_room_editor

    editor, error = get_room_editor(ctx, room_id, require_managed=True)
    if error:
        return {"status": "error", "message": error}

    try:
        if action == "add":
            if kind == "http":
                if not url:
                    return {"status": "error", "message": "URL required"}
                editor.add_mcp_http(name, url)
            elif kind == "stdio":
                if not command:
                    msg = "Command required for stdio MCP"
                    return {"status": "error", "message": msg}
                editor.add_mcp_stdio(name, command)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown MCP kind: {kind}. Use http or stdio.",
                }

            editor.save()

            errors = editor.validate()
            if errors:
                return {
                    "status": "warning",
                    "message": f"Added but validation issues: {errors}",
                }

            return {
                "status": "success",
                "message": f"Added MCP {name} ({kind}) to {room_id}",
            }

        elif action == "remove":
            if editor.remove_mcp(name):
                editor.save()
                return {
                    "status": "success",
                    "message": f"Removed MCP {name} from {room_id}",
                }
            else:
                return {
                    "status": "error",
                    "message": f"MCP {name} not found in {room_id}",
                }

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    except Exception as e:
        return {"status": "error", "message": f"Failed: {e}"}


def toggle_mcp_server(ctx: AnalysisContext, room_id: str, enable: bool) -> dict:
    """Enable or disable MCP server mode for a room."""
    from crazy_glue.analysis.tools.room_ops import get_room_editor

    editor, error = get_room_editor(ctx, room_id, require_managed=True)
    if error:
        return {"status": "error", "message": error}

    try:
        editor.set_allow_mcp(enable)
        editor.save()

        errors = editor.validate()
        if errors:
            return {
                "status": "warning",
                "message": f"Saved but validation issues: {errors}",
            }

        action = "enabled" if enable else "disabled"
        return {
            "status": "success",
            "message": f"MCP server mode {action} for {room_id}",
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed: {e}"}
