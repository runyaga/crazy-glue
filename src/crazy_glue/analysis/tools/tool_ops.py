"""Tool generation and management operations."""

from __future__ import annotations

import importlib
import typing

import yaml

from crazy_glue.analysis.validators import sanitize_identifier
from crazy_glue.analysis.validators import validate_python_compiles
from crazy_glue.analysis.validators import validate_python_syntax

if typing.TYPE_CHECKING:
    from crazy_glue.analysis.context import AnalysisContext

# Tools directory relative to project root
TOOLS_DIR = "src/crazy_glue/tools"

# Available tool types for rooms
AVAILABLE_TOOLS = {
    "soliplex.tools.search_documents": {
        "description": "RAG document search",
        "requires": "rag_lancedb_stem config",
    },
    "soliplex.tools.research_report": {
        "description": "RAG research report generation",
        "requires": "rag_lancedb_stem config",
    },
    "soliplex.tools.ask_with_rich_citations": {
        "description": "RAG Q&A with citations",
        "requires": "rag_lancedb_stem config",
    },
}


def list_available_tools() -> list[dict]:
    """List available tool types that can be added to rooms."""
    tools = []
    for name, info in AVAILABLE_TOOLS.items():
        tools.append({
            "name": name,
            "description": info["description"],
            "requires": info["requires"],
        })
    return tools


async def generate_tool(
    ctx: AnalysisContext,
    room_id: str,
    name: str,
    description: str,
) -> dict:
    """Generate a tool and stage it for approval."""
    from crazy_glue.analysis.tools.room_ops import get_room_editor

    # Sanitize and validate tool name first
    func_name, error = sanitize_identifier(name)
    if error:
        return {"status": "error", "message": error}

    # Validate room exists and is managed
    editor, error = get_room_editor(ctx, room_id, require_managed=True)
    if error:
        return {"status": "error", "message": error}

    # Generate the code and description using LLM
    code, generated_description = await _generate_tool_code(
        ctx, func_name, description
    )

    # Stage tool
    tool_data = {
        "room_id": room_id,
        "name": func_name,
        "original_name": name,
        "description": generated_description,
        "user_description": description,
        "code": code,
        "file_path": f"{TOOLS_DIR}/{func_name}.py",
    }
    ctx.save_pending_tool(tool_data)

    return {
        "status": "staged",
        "code": code,
        "description": generated_description,
        "file_path": tool_data["file_path"],
        "message": f"Tool '{name}' staged for {room_id}",
    }


async def _generate_tool_code(
    ctx: AnalysisContext,
    name: str,
    description: str,
    refinement: str | None = None,
) -> tuple[str, str]:
    """Generate tool code and description from user request using LLM."""
    import pydantic
    from pydantic_ai import Agent

    class GeneratedTool(pydantic.BaseModel):
        """LLM output containing generated tool code and description."""

        code: str = pydantic.Field(
            description="Complete Python code for the tool"
        )
        summary: str = pydantic.Field(
            description="Concise 1-sentence tool description"
        )

    func_name = name
    model_name = "".join(w.capitalize() for w in func_name.split("_"))
    model_name += "Result"

    refinement_line = ""
    if refinement:
        refinement_line = f"**Refinement**: {refinement}\n"

    prompt = f"""Generate a Python tool for pydantic-ai.

**Tool name**: {func_name}
**Description**: {description}
{refinement_line}
**Requirements**:
1. Create a Pydantic model `{model_name}` for structured output
2. Create an async function `{func_name}` that returns `{model_name}`
3. Add a docstring to the function describing what it does (IMPORTANT!)
4. Function should NOT use RunContext - just take necessary parameters
5. Implement the ACTUAL logic, not a placeholder
6. Handle errors gracefully, returning error info in the model
7. Use appropriate imports (pathlib, pydantic, etc.)

**Example pattern** (adapt fields/logic to the actual task):

import pydantic

class SearchResult(pydantic.BaseModel):
    found: bool
    matches: list[str]
    count: int
    error: str | None = None

async def search_files(pattern: str, path: str = ".") -> SearchResult:
    \"\"\"Search for files matching a glob pattern.\"\"\"
    from pathlib import Path
    try:
        target = Path(path).resolve()
        matches = [str(p) for p in target.rglob(pattern)]
        return SearchResult(
            found=len(matches) > 0,
            matches=matches,
            count=len(matches),
        )
    except Exception as e:
        return SearchResult(found=False, matches=[], count=0, error=str(e))

Provide:
1. **code**: Complete Python code (no markdown fences)
2. **summary**: Concise 1-sentence description (e.g., "Find large files")"""

    model = _get_model(ctx)
    agent: Agent[None, GeneratedTool] = Agent(model, output_type=GeneratedTool)
    result = await agent.run(prompt)
    tool_output = result.output

    code = tool_output.code
    summary = tool_output.summary

    # Clean up any markdown fences if present
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()

    # Add module docstring if missing
    if not code.startswith('"""'):
        header = f'"""\nTool: {name}\n\n{summary}\n"""\n\n'
        code = header + code

    return code, summary


def _get_model(ctx: AnalysisContext):
    """Create model using soliplex configuration."""
    from pydantic_ai.models import openai as openai_models
    from pydantic_ai.providers import ollama as ollama_providers

    if ctx.model is not None:
        return ctx.model

    provider_base_url = ctx.installation_config.get_environment("OLLAMA_BASE_URL")
    provider = ollama_providers.OllamaProvider(
        base_url=f"{provider_base_url}/v1",
    )
    ctx.model = openai_models.OpenAIChatModel(
        model_name=ctx.model_name,
        provider=provider,
    )
    return ctx.model


def apply_pending_tool(ctx: AnalysisContext) -> dict:
    """Apply the staged tool - validate, write, wire to room, check-config."""
    from crazy_glue.analysis.tools.room_ops import get_room_editor

    pending = ctx.load_pending_tool()
    if not pending:
        return {"status": "error", "message": "No pending tool to apply"}

    file_path = ctx.project_root / pending["file_path"]
    name = pending["name"]
    room_id = pending["room_id"]
    code = pending["code"]

    # Validate Python syntax
    if error := validate_python_syntax(code, str(file_path)):
        return {"status": "error", "message": error}

    # Validate code compiles
    if error := validate_python_compiles(code, str(file_path)):
        return {"status": "error", "message": error}

    # Build module path for tool registration
    file_path_str = pending["file_path"]
    module_path = file_path_str
    if module_path.startswith("src/"):
        module_path = module_path[4:]
    tool_module = module_path.replace("/", ".").replace(".py", "")
    tool_dotted_path = f"{tool_module}.{name}"

    try:
        # Write the tool file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code)

        # Validate module can be imported AND function exists
        try:
            module = importlib.import_module(tool_module)
            tool_func = getattr(module, name, None)
            if tool_func is None:
                if file_path.exists():
                    file_path.unlink()
                return {
                    "status": "error",
                    "message": f"Function '{name}' not found in {tool_module}",
                }
            if not callable(tool_func):
                if file_path.exists():
                    file_path.unlink()
                return {
                    "status": "error",
                    "message": f"'{name}' in {tool_module} is not callable",
                }
        except ImportError as e:
            if file_path.exists():
                file_path.unlink()
            return {
                "status": "error",
                "message": f"Import failed for {tool_module}: {e}",
            }

        # Wire tool into room config
        editor, error = get_room_editor(ctx, room_id, require_managed=True)
        if error:
            return {"status": "error", "message": error}

        editor.add_tool(tool_dotted_path)
        editor.save()

        # Validate saved YAML is parseable
        try:
            yaml.safe_load(editor.config_path.read_text())
        except yaml.YAMLError as e:
            if file_path.exists():
                file_path.unlink()
            return {"status": "error", "message": f"Invalid YAML after save: {e}"}

        # Run check-config to validate
        errors = editor.validate()
        if errors:
            editor.remove_tool(tool_dotted_path)
            editor.save()
            if file_path.exists():
                file_path.unlink()
            err_msg = "; ".join(errors)
            msg = f"Config validation failed: {err_msg}"
            return {"status": "error", "message": msg}

        # Success - clear staging
        ctx.clear_pending_tool()

        return {
            "status": "success",
            "message": f"Tool '{name}' added to {room_id}",
            "file_path": str(file_path),
            "tool_name": tool_dotted_path,
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to apply: {e}"}


def discard_pending_tool(ctx: AnalysisContext) -> dict:
    """Discard the staged tool."""
    pending = ctx.load_pending_tool()
    if not pending:
        return {"status": "error", "message": "No pending tool to discard"}

    name = pending["name"]
    ctx.clear_pending_tool()
    return {"status": "success", "message": f"Discarded pending tool '{name}'"}


async def refine_pending_tool(ctx: AnalysisContext, refinement: str) -> dict:
    """Refine the staged tool with additional requirements."""
    pending = ctx.load_pending_tool()
    if not pending:
        return {"status": "error", "message": "No pending tool to refine"}

    base_desc = pending.get("user_description", pending["description"])
    code, new_description = await _generate_tool_code(
        ctx, pending["name"], base_desc, refinement=refinement
    )

    pending["description"] = new_description
    pending["code"] = code
    ctx.save_pending_tool(pending)

    return {
        "status": "staged",
        "code": code,
        "description": new_description,
        "file_path": pending["file_path"],
        "message": f"Tool refined with: {refinement}",
    }


def manage_tool(
    ctx: AnalysisContext,
    room_id: str,
    action: str,
    tool_name: str,
) -> dict:
    """Add or remove a tool from a room."""
    from crazy_glue.analysis.tools.room_ops import get_room_editor

    editor, error = get_room_editor(ctx, room_id, require_managed=True)
    if error:
        return {"status": "error", "message": error}

    try:
        if action == "add":
            if tool_name not in AVAILABLE_TOOLS:
                avail = ", ".join(AVAILABLE_TOOLS.keys())
                return {
                    "status": "error",
                    "message": f"Unknown tool: {tool_name}. Available: {avail}",
                }

            editor.add_tool(tool_name)
            editor.save()

            errors = editor.validate()
            if errors:
                return {
                    "status": "warning",
                    "message": f"Added but validation issues: {errors}",
                }

            return {
                "status": "success",
                "message": f"Added {tool_name} to {room_id}",
            }

        elif action == "remove":
            if editor.remove_tool(tool_name):
                editor.save()
                return {
                    "status": "success",
                    "message": f"Removed {tool_name} from {room_id}",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Tool {tool_name} not found in {room_id}",
                }

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    except Exception as e:
        return {"status": "error", "message": f"Failed: {e}"}
