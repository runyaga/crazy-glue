"""Room operations for the System Architect."""

from __future__ import annotations

import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from crazy_glue.analysis.context import AnalysisContext
    from crazy_glue.analysis.room_editor import RoomConfigEditor


def list_rooms(ctx: AnalysisContext) -> list[dict]:
    """List all registered rooms in the installation."""
    rooms = []
    for room_id, room_cfg in ctx.installation_config.room_configs.items():
        agent_cfg = room_cfg.agent_config
        rooms.append({
            "id": room_id,
            "name": room_cfg.name,
            "description": room_cfg.description or "",
            "agent_kind": agent_cfg.kind,
            "factory": getattr(agent_cfg, "factory_name", None),
        })
    return sorted(rooms, key=lambda r: r["id"])


def list_managed_rooms(ctx: AnalysisContext) -> list[dict]:
    """List rooms created by the architect (managed rooms)."""
    from crazy_glue.analysis.room_editor import RoomConfigEditor

    managed = []
    for room_id, room_cfg in ctx.installation_config.room_configs.items():
        config_path = getattr(room_cfg, "_config_path", None)
        if not config_path:
            continue

        room_dir = Path(config_path).parent
        try:
            editor = RoomConfigEditor(room_dir)
            if editor.is_managed():
                agent_cfg = room_cfg.agent_config
                managed.append({
                    "id": room_id,
                    "name": room_cfg.name,
                    "description": room_cfg.description or "",
                    "agent_kind": agent_cfg.kind,
                    "config_path": str(config_path),
                })
        except Exception:
            continue

    return sorted(managed, key=lambda r: r["id"])


def inspect_room(ctx: AnalysisContext, room_id: str) -> dict:
    """Get detailed info about a specific room."""
    room_cfg = ctx.installation_config.room_configs.get(room_id)
    if not room_cfg:
        available = list(ctx.installation_config.room_configs.keys())
        return {"error": f"Room '{room_id}' not found. Available: {available}"}

    agent_cfg = room_cfg.agent_config
    result = {
        "id": room_id,
        "name": room_cfg.name,
        "description": room_cfg.description,
        "welcome_message": room_cfg.welcome_message,
        "suggestions": room_cfg.suggestions or [],
        "agent": {
            "kind": agent_cfg.kind,
        },
        "config_path": str(room_cfg._config_path),
    }

    # Add factory-specific info
    if agent_cfg.kind == "factory":
        result["agent"]["factory_name"] = agent_cfg.factory_name
        result["agent"]["extra_config"] = agent_cfg.extra_config or {}
    elif agent_cfg.kind == "default":
        model = getattr(agent_cfg, "model_name", None)
        result["agent"]["model_name"] = model
        prompt = getattr(agent_cfg, "_system_prompt_text", None)
        if prompt:
            result["agent"]["system_prompt_preview"] = prompt[:200]

    # Add tool info
    if room_cfg.tool_configs:
        result["tools"] = list(room_cfg.tool_configs.keys())

    # Add MCP toolsets
    if room_cfg.mcp_client_toolset_configs:
        result["mcp_toolsets"] = list(room_cfg.mcp_client_toolset_configs.keys())

    return result


def get_room_editor(
    ctx: AnalysisContext,
    room_id: str,
    require_managed: bool = True,
) -> tuple[RoomConfigEditor | None, str | None]:
    """Get a RoomConfigEditor for a room.

    Args:
        ctx: Analysis context
        room_id: The room ID to get editor for
        require_managed: If True, only allow editing managed rooms

    Returns:
        Tuple of (editor, error_message). Editor is None if error.
    """
    from crazy_glue.analysis.room_editor import RoomConfigEditor

    room_dir = None

    # First check in-memory config (loaded rooms)
    room_cfg = ctx.installation_config.room_configs.get(room_id)
    if room_cfg:
        config_path = getattr(room_cfg, "_config_path", None)
        if config_path:
            room_dir = Path(config_path).parent

    # Fall back to filesystem for newly created rooms not yet loaded
    if room_dir is None:
        candidate = ctx.project_root / "rooms" / room_id
        if candidate.exists() and (candidate / "room_config.yaml").exists():
            room_dir = candidate

    if room_dir is None:
        return None, f"Room `{room_id}` not found."

    try:
        editor = RoomConfigEditor(room_dir)
    except Exception as e:
        return None, f"Failed to load room config: {e}"

    if require_managed and not editor.is_managed():
        return None, (
            f"Room `{room_id}` is not managed by the architect.\n"
            "Only rooms created with `create <name> room` can be edited."
        )

    return editor, None


async def create_room(
    ctx: AnalysisContext,
    name: str,
    description: str,
) -> dict:
    """Create a new room."""
    import shutil

    import yaml

    from crazy_glue.analysis.room_editor import RoomConfigEditor

    name_slug = name.lower().replace(" ", "-").replace("_", "-")
    room_path = ctx.project_root / "rooms" / name_slug

    if room_path.exists():
        return {"error": f"Room '{name_slug}' already exists"}

    # Get model name from installation config or use default
    model_name = "gpt-oss:latest"
    try:
        if ctx.installation_config.agent_configs:
            first = list(ctx.installation_config.agent_configs.values())[0]
            model_name = first.model_name or model_name
    except Exception:
        pass

    # Scaffold the room
    room_path.mkdir(parents=True, exist_ok=True)
    config = f'''id: "{name_slug}"
name: "{name}"
description: "{description}"
welcome_message: |
  Welcome to the {name} room!

  {description}

suggestions:
  - "What can you help me with?"
  - "Show me an example"

agent:
  kind: "default"
  model_name: "{model_name}"
  system_prompt: |
    You are a helpful assistant in the {name} room.
    Your purpose: {description}
'''
    (room_path / "room_config.yaml").write_text(config)

    # Validate YAML
    try:
        yaml.safe_load(config)
    except yaml.YAMLError as e:
        shutil.rmtree(room_path)
        return {"error": f"Invalid YAML: {e}"}

    # Mark as managed
    try:
        editor = RoomConfigEditor(room_path)
        editor.load()
        editor.mark_as_managed()
    except Exception as e:
        shutil.rmtree(room_path)
        return {"error": f"Failed to mark as managed: {e}"}

    # Validate
    errors = editor.validate()
    if errors:
        shutil.rmtree(room_path)
        return {"error": f"Validation failed: {errors}"}

    return {"status": "created", "room_id": name_slug, "path": str(room_path)}


def edit_room(
    ctx: AnalysisContext,
    room_id: str,
    field: str,
    value: str,
) -> dict:
    """Edit a room's configuration field.

    Args:
        ctx: Analysis context
        room_id: The room to edit
        field: Field to edit (description, prompt, model, welcome)
        value: New value for the field

    Returns:
        Dict with status and any errors.
    """
    editor, error = get_room_editor(ctx, room_id, require_managed=True)
    if error:
        return {"status": "error", "message": error}

    field_map = {
        "description": editor.set_description,
        "prompt": editor.set_system_prompt,
        "model": editor.set_model_name,
        "welcome": editor.set_welcome_message,
    }

    if field not in field_map:
        valid = ", ".join(field_map.keys())
        return {"status": "error", "message": f"Unknown field: {field}. Valid: {valid}"}

    try:
        field_map[field](value)
        editor.save()

        # Validate after save
        errors = editor.validate()
        if errors:
            return {
                "status": "warning",
                "message": f"Saved but validation issues: {errors}",
            }

        return {"status": "success", "message": f"Updated {field} for {room_id}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to update: {e}"}


def manage_suggestion(
    ctx: AnalysisContext,
    room_id: str,
    action: str,
    value: str | int,
) -> dict:
    """Add or remove a suggestion from a room.

    Args:
        ctx: Analysis context
        room_id: The room to modify
        action: "add" or "remove"
        value: Text to add, or index to remove

    Returns:
        Dict with status and any errors.
    """
    editor, error = get_room_editor(ctx, room_id, require_managed=True)
    if error:
        return {"status": "error", "message": error}

    try:
        if action == "add":
            editor.add_suggestion(str(value))
            editor.save()
            return {"status": "success", "message": f"Added suggestion to {room_id}"}
        elif action == "remove":
            idx = int(value)
            if editor.remove_suggestion(idx):
                editor.save()
                return {"status": "success", "message": f"Removed suggestion {idx}"}
            else:
                return {"status": "error", "message": f"Invalid index: {idx}"}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    except ValueError:
        return {"status": "error", "message": "Remove requires a numeric index"}
    except Exception as e:
        return {"status": "error", "message": f"Failed: {e}"}
