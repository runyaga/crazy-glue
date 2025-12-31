"""Knowledge graph operations for the System Architect."""

from __future__ import annotations

import importlib
import inspect
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from crazy_glue.analysis.context import AnalysisContext

# Reference implementations mapping
REFERENCE_IMPLEMENTATIONS = {
    "joker": ("soliplex.examples", "joker_agent_factory"),
    "faux": ("soliplex.examples", "FauxAgent"),
    "brainstorm": (None, "src/crazy_glue/factories/brainstorm_factory.py"),
}


async def refresh_knowledge_graph(ctx: AnalysisContext, force: bool = False) -> dict:
    """Scan codebase and build/update knowledge graph."""
    from agentic_patterns.domain_exploration import ExplorationBoundary
    from agentic_patterns.domain_exploration import explore_domain

    map_path = ctx.map_path
    map_path.parent.mkdir(parents=True, exist_ok=True)

    boundary = ExplorationBoundary(
        max_depth=ctx.max_depth,
        max_files=ctx.max_files,
        dry_run=True,
        include_patterns=["**/*.py", "**/*.yaml", "**/*.md"],
        exclude_patterns=[
            "**/__pycache__/**",
            "**/.git/**",
            "**/.venv/**",
            "**/node_modules/**",
            "**/test*/**",
        ],
    )

    results = []
    for root in ctx.roots:
        root_path = ctx.project_root / root
        if root_path.exists():
            km = await explore_domain(
                root_path=str(root_path),
                boundary=boundary,
                storage_path=str(map_path),
            )
            results.append({
                "root": root,
                "entities": len(km.entities),
                "links": len(km.links),
                "files_processed": km.files_processed,
            })

    total_entities = sum(r["entities"] for r in results)
    total_links = sum(r["links"] for r in results)

    return {
        "status": "complete",
        "map_path": str(map_path),
        "roots_scanned": results,
        "total_entities": total_entities,
        "total_links": total_links,
    }


def _load_store(ctx: AnalysisContext):
    """Load existing knowledge store if available."""
    from agentic_patterns.domain_exploration import KnowledgeStore

    if ctx.map_path.exists():
        return KnowledgeStore.load(ctx.map_path)
    return None


def query_graph(
    ctx: AnalysisContext,
    query: str,
    entity_type: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search knowledge graph for entities matching query."""
    store = _load_store(ctx)
    if store is None:
        return [{"error": "Knowledge graph not found. Run refresh first."}]

    km = store.to_knowledge_map()
    query_lower = query.lower()
    matches = []

    for entity in km.entities:
        if entity_type and entity.entity_type != entity_type:
            continue

        score = 0
        name_lower = entity.name.lower()
        summary_lower = entity.summary.lower()

        if query_lower in name_lower:
            score += 10
            if name_lower == query_lower:
                score += 20
        if query_lower in summary_lower:
            score += 5

        if score > 0:
            matches.append((score, entity))

    matches.sort(key=lambda x: x[0], reverse=True)
    results = []

    for _, entity in matches[:limit]:
        result = {
            "id": entity.id,
            "name": entity.name,
            "type": entity.entity_type,
            "summary": entity.summary,
            "location": entity.location,
        }
        if entity.metadata.get("line"):
            result["line"] = entity.metadata["line"]
        results.append(result)

    return results


def read_entity_source(ctx: AnalysisContext, entity_id: str) -> str:
    """Read source code for a specific entity."""
    store = _load_store(ctx)
    if store is None:
        return "Error: Knowledge graph not found. Run refresh first."

    km = store.to_knowledge_map()
    entity = None
    for e in km.entities:
        if e.id == entity_id:
            entity = e
            break

    if entity is None:
        return f"Error: Entity '{entity_id}' not found in graph."

    file_path = Path(entity.location)
    if not file_path.exists():
        return f"Error: Source file not found: {entity.location}"

    try:
        content = file_path.read_text()
        lines = content.split("\n")

        line_num = entity.metadata.get("line", 1)
        start = max(0, line_num - 1)
        end = min(len(lines), start + 50)

        snippet = "\n".join(
            f"{i+1:4}: {line}" for i, line in enumerate(lines[start:end], start)
        )

        header = f"# {entity.name} ({entity.entity_type})"
        location = f"# {entity.location}:{line_num}"
        return f"{header}\n{location}\n\n{snippet}"
    except Exception as e:
        return f"Error reading file: {e}"


def read_reference_implementation(ctx: AnalysisContext, name: str) -> str:
    """Get source code for a reference implementation."""
    name_lower = name.lower()
    if name_lower not in REFERENCE_IMPLEMENTATIONS:
        available = ", ".join(REFERENCE_IMPLEMENTATIONS.keys())
        return f"Unknown reference: '{name}'. Available: {available}"

    module_name, target = REFERENCE_IMPLEMENTATIONS[name_lower]

    if module_name is None:
        file_path = ctx.project_root / target
        if not file_path.exists():
            return f"Error: File not found: {target}"
        return f"# {target}\n\n{file_path.read_text()}"

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, target)
        source = inspect.getsource(obj)
        return f"# {module_name}.{target}\n\n{source}"
    except Exception as e:
        return f"Error loading reference: {e}"
