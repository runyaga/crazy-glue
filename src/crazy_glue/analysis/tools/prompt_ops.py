"""Prompt library operations for the System Architect."""

from __future__ import annotations

import datetime
import typing

if typing.TYPE_CHECKING:
    from crazy_glue.analysis.context import AnalysisContext


def list_prompts(ctx: AnalysisContext) -> list[dict]:
    """List all saved prompts."""
    prompts = ctx.load_prompts()
    result = []
    for key, prompt in prompts.items():
        result.append({
            "id": key,
            "name": prompt.get("name", key),
            "preview": prompt.get("content", "")[:60] + "...",
            "created": prompt.get("created", ""),
        })
    return sorted(result, key=lambda p: p["name"])


def get_prompt(ctx: AnalysisContext, name: str) -> dict | None:
    """Get a specific prompt by name/id."""
    prompts = ctx.load_prompts()
    # Try exact match first
    if name in prompts:
        return prompts[name]
    # Try case-insensitive match
    name_lower = name.lower()
    for key, prompt in prompts.items():
        if key.lower() == name_lower:
            return prompt
    return None


def add_prompt(ctx: AnalysisContext, name: str, content: str) -> dict:
    """Add a new prompt to the library."""
    prompts = ctx.load_prompts()

    # Create slug from name
    slug = name.lower().replace(" ", "-").replace("_", "-")

    if slug in prompts:
        return {"status": "error", "message": f"Prompt '{slug}' already exists"}

    prompts[slug] = {
        "name": name,
        "content": content,
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    ctx.save_prompts(prompts)

    return {
        "status": "success",
        "message": f"Prompt '{name}' saved as '{slug}'",
        "id": slug,
    }


def remove_prompt(ctx: AnalysisContext, name: str) -> dict:
    """Remove a prompt from the library."""
    prompts = ctx.load_prompts()

    # Find the prompt
    key_to_remove = None
    name_lower = name.lower()
    for key in prompts:
        if key.lower() == name_lower:
            key_to_remove = key
            break

    if not key_to_remove:
        return {"status": "error", "message": f"Prompt '{name}' not found"}

    del prompts[key_to_remove]
    ctx.save_prompts(prompts)

    return {"status": "success", "message": f"Prompt '{key_to_remove}' removed"}


def use_prompt(ctx: AnalysisContext, room_id: str, prompt_name: str) -> dict:
    """Apply a prompt to a room's system prompt."""
    from crazy_glue.analysis.tools.room_ops import get_room_editor

    # Get the prompt
    prompt = get_prompt(ctx, prompt_name)
    if not prompt:
        return {"status": "error", "message": f"Prompt '{prompt_name}' not found"}

    # Get the room editor
    editor, error = get_room_editor(ctx, room_id, require_managed=True)
    if error:
        return {"status": "error", "message": error}

    try:
        editor.set_system_prompt(prompt["content"])
        editor.save()

        errors = editor.validate()
        if errors:
            return {
                "status": "warning",
                "message": f"Applied but validation issues: {errors}",
            }

        p_name = prompt.get("name", prompt_name)
        return {
            "status": "success",
            "message": f"Applied prompt '{p_name}' to {room_id}",
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed: {e}"}
