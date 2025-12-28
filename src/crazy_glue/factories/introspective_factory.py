"""Factory for Introspective Agent - uses soliplex internals to explore itself."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import tempfile
import typing
import uuid
from collections import abc
from pathlib import Path
from urllib.parse import urlparse

import httpx

# Configure logging with console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

import pydantic
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools
from pydantic_ai.models import openai as openai_models
from pydantic_ai.providers import ollama as ollama_providers

if typing.TYPE_CHECKING:
    from soliplex import config

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]


# Pattern knowledge base
PATTERN_KNOWLEDGE = {
    "routing": {
        "name": "Routing Pattern",
        "description": "Classify user intent and dispatch to specialized handlers based on detected intent.",
        "key_concepts": ["Intent classification", "Confidence scoring", "Handler dispatch"],
        "when_to_use": ["Multi-domain queries", "Support triage", "API gateways"],
        "diagram": """```mermaid
flowchart LR
    Query[User Query] --> Router{Intent<br/>Classifier}
    Router -->|orders| H1[Order Handler]
    Router -->|products| H2[Product Handler]
    Router -->|support| H3[Support Handler]
```""",
    },
    "reflection": {
        "name": "Reflection Pattern",
        "description": "Producer-critic improvement loops. Generate content, evaluate, iterate until quality threshold met.",
        "key_concepts": ["Producer agent", "Critic agent", "Quality criteria", "Iteration"],
        "when_to_use": ["Content generation", "Code review", "Quality assurance"],
        "diagram": """```mermaid
flowchart TB
    Input[Request] --> Producer[Producer Agent]
    Producer --> Draft[Draft Content]
    Draft --> Critic[Critic Agent]
    Critic --> Check{Quality OK?}
    Check -->|No| Feedback[Feedback]
    Feedback --> Producer
    Check -->|Yes| Output[Final Output]
```""",
    },
    "planning": {
        "name": "Planning Pattern",
        "description": "Goal decomposition into actionable steps, sequential execution, result synthesis.",
        "key_concepts": ["Goal decomposition", "Step execution", "Result synthesis"],
        "when_to_use": ["Complex multi-step tasks", "Research", "Project planning"],
        "diagram": """```mermaid
flowchart TB
    Goal[High-Level Goal] --> Planner[Planner Agent]
    Planner --> S1[Step 1]
    S1 --> S2[Step 2]
    S2 --> S3[Step 3]
    S3 --> Synth[Synthesizer]
    Synth --> Result[Final Result]
```""",
    },
    "parallelization": {
        "name": "Parallelization Pattern",
        "description": "Execute independent tasks concurrently using asyncio.gather. Strategies: sectioning, voting, map-reduce.",
        "key_concepts": ["Concurrent execution", "Sectioning", "Voting", "Map-reduce"],
        "when_to_use": ["Multi-section documents", "Consensus decisions", "Batch processing"],
        "diagram": """```mermaid
flowchart TB
    Input[Input] --> T1[Task 1] & T2[Task 2] & T3[Task 3]
    T1 & T2 & T3 --> Agg[Aggregate]
    Agg --> Output[Output]
```""",
    },
    "tool_use": {
        "name": "Tool Use Pattern",
        "description": "Equip agents with tools to interact with external systems and extend capabilities.",
        "key_concepts": ["Tool registration", "Tool execution", "Result handling"],
        "when_to_use": ["System integration", "Data access", "External APIs"],
        "diagram": """```mermaid
flowchart LR
    Agent[Agent] <--> Tools
    subgraph Tools
        T1[Tool 1]
        T2[Tool 2]
        T3[Tool 3]
    end
```""",
    },
    "multi_agent": {
        "name": "Multi-Agent Pattern",
        "description": "Multiple specialized agents collaborating. Topologies: supervisor, peer-to-peer, hierarchical.",
        "key_concepts": ["Agent specialization", "Coordination", "Message passing"],
        "when_to_use": ["Complex workflows", "Debates", "Collaborative tasks"],
        "diagram": """```mermaid
flowchart TB
    Coord[Coordinator] --> A1[Agent 1]
    Coord --> A2[Agent 2]
    Coord --> A3[Agent 3]
    A1 & A2 & A3 --> Coord
```""",
    },
}


# ============================================================
# PLANNING PATTERN MODELS
# ============================================================

class PlanStep(BaseModel):
    """A single step in an execution plan."""

    step_number: int = Field(description="Sequential step number (1-indexed)")
    tool_name: str = Field(description="Name of the tool to call")
    tool_args: dict[str, typing.Any] = Field(description="Arguments to pass to the tool")
    description: str = Field(description="What this step accomplishes")
    depends_on: list[int] = Field(default_factory=list, description="Step numbers this depends on")


class ExecutionPlan(BaseModel):
    """A plan of steps to accomplish a goal."""

    goal: str = Field(description="The original goal/task")
    reasoning: str = Field(description="Why this plan will accomplish the goal")
    steps: list[PlanStep] = Field(description="Ordered list of steps to execute")


class StepResult(BaseModel):
    """Result of executing a single step."""

    step_number: int
    tool_name: str
    success: bool
    result: typing.Any = None
    error: str | None = None


# ============================================================
# SELF-IMPROVEMENT MODELS
# ============================================================

class Learning(BaseModel):
    """A single learning extracted from a failed execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())
    task_category: str = Field(description="Category: introspection, rag, fetch, general")
    original_goal: str = Field(default="", description="The user's original goal")
    error_pattern: str = Field(description="What went wrong (e.g., 'incorrect room ID format')")
    failed_tool: str = Field(description="Tool that failed")
    failed_args: dict[str, typing.Any] = Field(default_factory=dict, description="Arguments that caused failure")
    correct_approach: str = Field(description="What should have been done")
    example_correction: str = Field(description="Concrete example of correct usage")
    times_applied: int = Field(default=0, description="How often this learning prevented errors")


class LearningStore(BaseModel):
    """Persistent store for agent learnings."""

    learnings: list[Learning] = Field(default_factory=list)
    version: str = Field(default="1.0")

    def add(self, learning: Learning) -> None:
        """Add a new learning to the store."""
        self.learnings.append(learning)

    def get_relevant(self, task_category: str, k: int = 3) -> list[Learning]:
        """Get the k most recent learnings for a task category."""
        matching = [l for l in self.learnings if l.task_category == task_category]
        # Return most recent first
        return sorted(matching, key=lambda x: x.timestamp, reverse=True)[:k]

    def get_all_categories(self) -> list[str]:
        """Get all unique task categories."""
        return list(set(l.task_category for l in self.learnings))


class LearningStoreManager:
    """Manages persistence of learning store to disk."""

    DEFAULT_PATH = Path("db/learnings.json")

    def __init__(self, path: Path | None = None):
        self.path = path or self.DEFAULT_PATH

    def load(self) -> LearningStore:
        """Load learning store from disk, or create empty if not exists."""
        if not self.path.exists():
            return LearningStore()
        try:
            data = json.loads(self.path.read_text())
            return LearningStore.model_validate(data)
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to load learnings: {e}")
            return LearningStore()

    def save(self, store: LearningStore) -> None:
        """Save learning store to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(store.model_dump_json(indent=2))
        logger.info(f"[LEARNING] Saved {len(store.learnings)} learnings to {self.path}")


# ============================================================
# CONVERSATION MEMORY MODELS
# ============================================================

class MemoryMessage(BaseModel):
    """A single message in conversation memory."""

    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message content")
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())


class WindowMemory(BaseModel):
    """Sliding window memory - keeps last N exchanges."""

    messages: list[MemoryMessage] = Field(default_factory=list)
    window_size: int = Field(default=20, description="Max messages to keep")

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(MemoryMessage(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message."""
        # Truncate long responses for memory
        truncated = content[:1000] + "..." if len(content) > 1000 else content
        self.messages.append(MemoryMessage(role="assistant", content=truncated))
        self._trim()

    def _trim(self) -> None:
        """Keep only the last window_size messages."""
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_context(self) -> str:
        """Format memory for prompt injection."""
        if not self.messages:
            return ""
        lines = []
        for msg in self.messages:
            role = "USER" if msg.role == "user" else "ASSISTANT"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def get_recent_summary(self, n: int = 3) -> str:
        """Get a brief summary of recent exchanges."""
        recent = self.messages[-n*2:] if len(self.messages) >= n*2 else self.messages
        if not recent:
            return "No previous conversation."
        lines = []
        for msg in recent:
            role = "You asked" if msg.role == "user" else "I responded"
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"- {role}: {preview}")
        return "\n".join(lines)


class MemoryManager:
    """Manages persistence of conversation memory."""

    DEFAULT_PATH = Path("db/introspective_memory.json")

    def __init__(self, path: Path | None = None):
        self.path = path or self.DEFAULT_PATH

    def load(self) -> WindowMemory:
        """Load memory from disk."""
        if not self.path.exists():
            return WindowMemory()
        try:
            data = json.loads(self.path.read_text())
            return WindowMemory.model_validate(data)
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to load memory: {e}")
            return WindowMemory()

    def save(self, memory: WindowMemory) -> None:
        """Save memory to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(memory.model_dump_json(indent=2))
        logger.debug(f"[MEMORY] Saved {len(memory.messages)} messages to {self.path}")


# ============================================================
# TOOL REGISTRY - Single source of truth for all tools
# ============================================================


@dataclasses.dataclass
class ToolSpec:
    """Specification for a tool in the registry."""

    name: str
    description: str
    args: dict[str, str]  # arg_name -> type hint string
    implementation: typing.Callable[..., typing.Awaitable[dict[str, typing.Any]]]


async def _tool_fetch_url_content(ctx: "IntrospectiveContext", url: str) -> dict[str, typing.Any]:
    """Fetch text/HTML content from a URL."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return {"success": True, "content": response.text, "url": url}
    except Exception as e:
        return {"error": str(e)}


async def _tool_fetch_url_to_file(ctx: "IntrospectiveContext", url: str) -> dict[str, typing.Any]:
    """Download URL to a temporary file."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            parsed = urlparse(url)
            filename = Path(parsed.path).name or "downloaded_file"
            temp_path = Path(tempfile.gettempdir()) / f"introspective_{uuid.uuid4().hex[:8]}_{filename}"
            temp_path.write_bytes(response.content)
            return {"success": True, "file_path": str(temp_path), "filename": filename}
    except Exception as e:
        return {"error": str(e)}


async def _tool_rag_info(ctx: "IntrospectiveContext") -> dict[str, typing.Any]:
    """Get RAG system status and document count."""
    if not ctx.rag_toolset:
        return {"error": "RAG toolset not configured. Start haiku-rag MCP server first."}
    try:
        async with ctx.rag_toolset:
            docs = await ctx.rag_toolset.direct_call_tool("list_documents", {"limit": 1000})
        doc_count = len(docs) if isinstance(docs, list) else 0
        return {"success": True, "status": "connected", "document_count": doc_count, "mcp_url": "http://127.0.0.1:8001/mcp"}
    except Exception as e:
        return {"error": str(e)}


async def _tool_rag_add_url(ctx: "IntrospectiveContext", url: str, title: str | None = None) -> dict[str, typing.Any]:
    """Index content from a URL into RAG."""
    if not ctx.rag_toolset:
        return {"error": "RAG toolset not configured"}
    try:
        async with ctx.rag_toolset:
            doc_id = await ctx.rag_toolset.direct_call_tool("add_document_from_url", {"url": url, "title": title})
        return {"success": True, "document_id": doc_id, "url": url}
    except Exception as e:
        return {"error": str(e)}


async def _tool_rag_add_file(ctx: "IntrospectiveContext", file_path: str, title: str | None = None) -> dict[str, typing.Any]:
    """Index a local file into RAG."""
    if not ctx.rag_toolset:
        return {"error": "RAG toolset not configured"}
    try:
        async with ctx.rag_toolset:
            doc_id = await ctx.rag_toolset.direct_call_tool("add_document_from_file", {"file_path": file_path, "title": title})
        return {"success": True, "document_id": doc_id, "file_path": file_path}
    except Exception as e:
        return {"error": str(e)}


async def _tool_rag_add_text(ctx: "IntrospectiveContext", content: str, title: str) -> dict[str, typing.Any]:
    """Index text content into RAG."""
    if not ctx.rag_toolset:
        return {"error": "RAG toolset not configured"}
    try:
        async with ctx.rag_toolset:
            doc_id = await ctx.rag_toolset.direct_call_tool("add_document_from_text", {"content": content, "title": title})
        return {"success": True, "document_id": doc_id}
    except Exception as e:
        return {"error": str(e)}


async def _tool_rag_search(ctx: "IntrospectiveContext", query: str, limit: int = 5) -> dict[str, typing.Any]:
    """Search the RAG knowledge base."""
    if not ctx.rag_toolset:
        return {"error": "RAG toolset not configured"}
    try:
        async with ctx.rag_toolset:
            results = await ctx.rag_toolset.direct_call_tool("search_documents", {"query": query, "limit": limit})
        return {"success": True, "results": results}
    except Exception as e:
        return {"error": str(e)}


async def _tool_rag_list_documents(ctx: "IntrospectiveContext", limit: int = 20) -> dict[str, typing.Any]:
    """List documents in the RAG knowledge base."""
    if not ctx.rag_toolset:
        return {"error": "RAG toolset not configured"}
    try:
        async with ctx.rag_toolset:
            docs = await ctx.rag_toolset.direct_call_tool("list_documents", {"limit": limit})
        return {"success": True, "documents": docs}
    except Exception as e:
        return {"error": str(e)}


async def _tool_rag_delete_document(ctx: "IntrospectiveContext", document_id: str) -> dict[str, typing.Any]:
    """Delete a document from the RAG knowledge base."""
    if not ctx.rag_toolset:
        return {"error": "RAG toolset not configured"}
    try:
        async with ctx.rag_toolset:
            result = await ctx.rag_toolset.direct_call_tool("delete_document", {"document_id": document_id})
        return {"success": True, "deleted_document_id": document_id, "result": result}
    except Exception as e:
        return {"error": str(e)}


async def _tool_list_all_rooms(ctx: "IntrospectiveContext") -> dict[str, typing.Any]:
    """List all rooms in the installation."""
    installation = ctx.installation_config
    rooms = [
        {"id": room_id, "name": room_config.name, "description": room_config.description}
        for room_id, room_config in installation.room_configs.items()
    ]
    return {"success": True, "rooms": sorted(rooms, key=lambda r: r["id"])}


async def _tool_inspect_room(ctx: "IntrospectiveContext", room_id: str) -> dict[str, typing.Any]:
    """Get detailed configuration for a specific room."""
    installation = ctx.installation_config
    if room_id not in installation.room_configs:
        return {"error": f"Room '{room_id}' not found"}
    room = installation.room_configs[room_id]
    return {"success": True, "id": room.id, "name": room.name, "description": room.description, "suggestions": room.suggestions}


async def _tool_explain_pattern(ctx: "IntrospectiveContext", pattern_name: str) -> dict[str, typing.Any]:
    """Explain an agentic pattern."""
    pattern_key = pattern_name.lower().replace(" ", "_")
    if pattern_key not in PATTERN_KNOWLEDGE:
        return {"error": f"Unknown pattern: {pattern_name}"}
    return {"success": True, **PATTERN_KNOWLEDGE[pattern_key]}


async def _tool_list_patterns(ctx: "IntrospectiveContext") -> dict[str, typing.Any]:
    """List all known agentic patterns."""
    return {
        "success": True,
        "patterns": [{"name": info["name"], "description": info["description"][:100]} for info in PATTERN_KNOWLEDGE.values()],
    }


# Build the registry
TOOL_REGISTRY: dict[str, ToolSpec] = {
    "fetch_url_content": ToolSpec(
        name="fetch_url_content",
        description="Fetch text/HTML content from a URL",
        args={"url": "str"},
        implementation=_tool_fetch_url_content,
    ),
    "fetch_url_to_file": ToolSpec(
        name="fetch_url_to_file",
        description="Download URL to a temporary file (for PDFs, etc.)",
        args={"url": "str"},
        implementation=_tool_fetch_url_to_file,
    ),
    "rag_info": ToolSpec(
        name="rag_info",
        description="Get RAG system status and document count",
        args={},
        implementation=_tool_rag_info,
    ),
    "rag_add_url": ToolSpec(
        name="rag_add_url",
        description="Index content from a URL into RAG",
        args={"url": "str", "title": "str | None"},
        implementation=_tool_rag_add_url,
    ),
    "rag_add_file": ToolSpec(
        name="rag_add_file",
        description="Index a local file into RAG",
        args={"file_path": "str", "title": "str | None"},
        implementation=_tool_rag_add_file,
    ),
    "rag_add_text": ToolSpec(
        name="rag_add_text",
        description="Index text content into RAG",
        args={"content": "str", "title": "str"},
        implementation=_tool_rag_add_text,
    ),
    "rag_search": ToolSpec(
        name="rag_search",
        description="Search the RAG knowledge base",
        args={"query": "str", "limit": "int = 5"},
        implementation=_tool_rag_search,
    ),
    "rag_list_documents": ToolSpec(
        name="rag_list_documents",
        description="List documents in RAG",
        args={"limit": "int = 20"},
        implementation=_tool_rag_list_documents,
    ),
    "rag_delete_document": ToolSpec(
        name="rag_delete_document",
        description="Delete a document from RAG",
        args={"document_id": "str"},
        implementation=_tool_rag_delete_document,
    ),
    "list_all_rooms": ToolSpec(
        name="list_all_rooms",
        description="List all rooms in the installation",
        args={},
        implementation=_tool_list_all_rooms,
    ),
    "inspect_room": ToolSpec(
        name="inspect_room",
        description="Get detailed configuration for a room",
        args={"room_id": "str"},
        implementation=_tool_inspect_room,
    ),
    "explain_pattern": ToolSpec(
        name="explain_pattern",
        description="Explain an agentic pattern",
        args={"pattern_name": "str"},
        implementation=_tool_explain_pattern,
    ),
    "list_patterns": ToolSpec(
        name="list_patterns",
        description="List all known agentic patterns",
        args={},
        implementation=_tool_list_patterns,
    ),
}


def _generate_tools_section() -> str:
    """Generate the tools documentation for PLANNER_SYSTEM_PROMPT."""
    lines = ["## AVAILABLE TOOLS\n"]

    # Group by category
    categories = {
        "Web Fetching": ["fetch_url_content", "fetch_url_to_file"],
        "RAG Knowledge Base": ["rag_info", "rag_add_url", "rag_add_file", "rag_add_text", "rag_search", "rag_list_documents", "rag_delete_document"],
        "System Introspection": ["list_all_rooms", "inspect_room", "explain_pattern", "list_patterns"],
    }

    for category, tool_names in categories.items():
        lines.append(f"**{category}:**")
        for name in tool_names:
            if name in TOOL_REGISTRY:
                spec = TOOL_REGISTRY[name]
                args_str = ", ".join(f"{k}: {v}" for k, v in spec.args.items()) if spec.args else ""
                lines.append(f"- {spec.name}({args_str}) - {spec.description}")
        lines.append("")

    return "\n".join(lines)


# Generate tools section from registry
_GENERATED_TOOLS_SECTION = _generate_tools_section()

PLANNER_SYSTEM_PROMPT = """You are a planning agent that breaks down tasks into tool calls.

""" + _GENERATED_TOOLS_SECTION + """
## OUTPUT FORMAT

You MUST output valid JSON matching this EXACT structure:

```json
{
  "goal": "The user's original request",
  "reasoning": "Why this plan will work",
  "steps": [
    {
      "step_number": 1,
      "tool_name": "fetch_url_content",
      "tool_args": {"url": "https://example.com/file.txt"},
      "description": "Fetch the file content",
      "depends_on": []
    },
    {
      "step_number": 2,
      "tool_name": "rag_add_url",
      "tool_args": {"url": "https://example.com/page1", "title": "Page 1"},
      "description": "Index first URL",
      "depends_on": [1]
    }
  ]
}
```

## RULES

1. Every step MUST have a tool_name - no null tools
2. Each step calls exactly ONE tool
3. Use depends_on to indicate dependencies on prior steps
4. For tasks like "add each URL from a list", use rag_add_url directly for each URL (it fetches automatically)
5. Use exact field names: step_number, tool_name, tool_args, description, depends_on

## EXAMPLE

Task: "Fetch https://example.com/llms.txt and add each URL to RAG"

```json
{
  "goal": "Fetch llms.txt and add each URL to RAG",
  "reasoning": "First fetch the file to see URLs, then add each one using rag_add_url which handles fetching",
  "steps": [
    {
      "step_number": 1,
      "tool_name": "fetch_url_content",
      "tool_args": {"url": "https://example.com/llms.txt"},
      "description": "Fetch the llms.txt file to see what URLs it contains",
      "depends_on": []
    },
    {
      "step_number": 2,
      "tool_name": "rag_add_url",
      "tool_args": {"url": "https://example.com/page1", "title": "Page 1"},
      "description": "Index first URL from the list",
      "depends_on": [1]
    },
    {
      "step_number": 3,
      "tool_name": "rag_add_url",
      "tool_args": {"url": "https://example.com/page2", "title": "Page 2"},
      "description": "Index second URL from the list",
      "depends_on": [1]
    }
  ]
}
```"""


SYNTHESIZER_SYSTEM_PROMPT = """You are a synthesis agent that summarizes execution results.

Given a goal and a list of step results, provide a clear summary of:
1. What was accomplished
2. Any failures or issues
3. Key data/outputs from the execution

Be concise but thorough."""


LEARNING_EXTRACTOR_PROMPT = """You are a failure analysis specialist. Extract ONE concise learning from failures.

CRITICAL: Keep ALL fields SHORT (1-2 sentences max). No bullet points or lists in field values.

## Output Format

{
  "task_category": "introspection|rag|fetch|general",
  "error_pattern": "SHORT description of what went wrong (max 15 words)",
  "failed_tool": "tool_name",
  "correct_approach": "SHORT fix (max 20 words)",
  "example_correction": "SHORT concrete example (max 15 words)"
}

## Example

Input:
- Goal: Inspect the brainstorm room
- Tool: inspect_room
- Error: Room 'brainstorm-room-id' not found

Output:
{
  "task_category": "introspection",
  "error_pattern": "Assumed room IDs have '-room-id' suffix",
  "failed_tool": "inspect_room",
  "correct_approach": "Room IDs are simple slugs. Use list_all_rooms() first.",
  "example_correction": "Use 'brainstorm' not 'brainstorm-room-id'"
}

## Another Example (incomplete plan)

Input:
- Goal: Fetch llms.txt and add each URL to RAG
- Synthesis: Only the first URL was processed

Output:
{
  "task_category": "rag",
  "error_pattern": "Plan only processed first URL instead of iterating all",
  "failed_tool": "rag_add_url",
  "correct_approach": "Create separate plan steps for EACH URL found in the file.",
  "example_correction": "Add step for each URL: rag_add_url(url1), rag_add_url(url2), etc."
}"""


SYSTEM_PROMPT = """You are the Introspective Agent - a self-aware AI that explores the soliplex installation and manages a RAG knowledge base.

## How You Work

You use the **Planning Pattern**:
1. Analyze the user's request
2. Break it into tool calls
3. Execute each step
4. Synthesize the results

For simple queries, you can call tools directly.

## Available Tools

### RAG Knowledge Base
- `rag_info()` - Get RAG status and document count
- `rag_add_url(url, title)` - Index content from a URL
- `rag_add_file(file_path, title)` - Index a local file
- `rag_add_text(content, title)` - Index text content
- `rag_search(query, limit)` - Search the knowledge base
- `rag_list_documents(limit)` - List indexed documents
- `rag_delete_document(document_id)` - Remove a document

### Web Fetching
- `fetch_url_content(url)` - Get text/HTML content from a URL
- `fetch_url_to_file(url)` - Download file to temp path (for PDFs)

### System Introspection
- `list_all_rooms()` - List all rooms in the installation
- `inspect_room(room_id)` - Get room configuration
- `explain_pattern(pattern_name)` - Explain an agentic pattern
- `list_patterns()` - List all known patterns
- `find_rooms_by_pattern(pattern_name)` - Find rooms using a pattern
- `generate_installation_diagram()` - Create mermaid architecture diagram
- `who_am_i()` - Get info about this agent
- `list_my_tools()` - List all available tools
- `run_health_check()` - Check installation health

## Guidelines

- Call tools to get information - don't guess
- For multi-step tasks, work through them sequentially
- Report results clearly and concisely
- If a tool fails, explain the error and suggest alternatives"""


def _extract_prompt(message_history: MessageHistory | None) -> str:
    """Extract the user prompt from message history."""
    if not message_history:
        return ""
    last_msg = message_history[-1]
    if isinstance(last_msg, ai_messages.ModelRequest):
        for part in last_msg.parts:
            if isinstance(part, ai_messages.UserPromptPart):
                return part.content
    return ""


@dataclasses.dataclass
class IntrospectiveContext:
    """Runtime context with access to soliplex internals."""

    agent_config: config.FactoryAgentConfig
    rag_toolset: typing.Any = None

    @property
    def installation_config(self) -> config.InstallationConfig:
        """Get the live installation config."""
        return self.agent_config._installation_config


@dataclasses.dataclass
class IntrospectiveAgent:
    """Agent that introspects the live soliplex installation."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None
    _model = None
    _agent = None
    _rag_toolset = None

    @property
    def model_name(self) -> str:
        return self.agent_config.extra_config.get("model_name", "gpt-oss:20b")

    def _get_rag_toolset(self):
        """Get haiku-rag MCP toolset from config."""
        if self._rag_toolset is None and self.mcp_client_toolset_configs:
            from soliplex import mcp_client

            rag_config = self.mcp_client_toolset_configs.get("haiku-rag")
            if rag_config:
                toolset_klass = mcp_client.TOOLSET_CLASS_BY_KIND[rag_config.kind]
                self._rag_toolset = toolset_klass(**rag_config.tool_kwargs)

        return self._rag_toolset

    def _create_planner_agent(
        self,
        learnings: list[Learning] | None = None,
        rag_context: str = "",
        memory_context: str = "",
    ) -> Agent:
        """Create the planner agent that decomposes tasks into steps."""
        model = self._get_model()
        system_prompt = self._get_planner_prompt_with_context(learnings, rag_context, memory_context)
        return Agent(
            model,
            system_prompt=system_prompt,
            output_type=ExecutionPlan,
            retries=2,
        )

    def _get_planner_prompt_with_context(
        self,
        learnings: list[Learning] | None,
        rag_context: str = "",
        memory_context: str = "",
    ) -> str:
        """Generate planner prompt with learnings and conversation memory."""
        prompt = PLANNER_SYSTEM_PROMPT

        # Add conversation memory (recent exchanges)
        if memory_context:
            prompt += "\n\n## RECENT CONVERSATION HISTORY\n\n"
            prompt += "The user has been having this conversation with you:\n\n"
            prompt += memory_context
            prompt += "\n\nUse this context to understand references like 'that', 'the same', 'again', etc."

        # Add RAG-retrieved learnings (semantic matches)
        if rag_context:
            prompt += rag_context

        # Add category-based learnings from JSON
        if learnings:
            prompt += "\n\n## LEARNED CORRECTIONS (category-based)\n\n"
            prompt += "**IMPORTANT**: Avoid these previously identified mistakes:\n\n"

            for i, learning in enumerate(learnings, 1):
                prompt += f"{i}. **{learning.error_pattern}**\n"
                prompt += f"   - Tool: `{learning.failed_tool}`\n"
                prompt += f"   - Correct approach: {learning.correct_approach}\n"
                prompt += f"   - Example: {learning.example_correction}\n\n"

        return prompt

    def _create_synthesizer_agent(self) -> Agent:
        """Create the synthesizer agent that summarizes results."""
        model = self._get_model()
        return Agent(
            model,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            output_type=str,
            retries=2,
        )

    def _create_learning_extractor_agent(self) -> Agent:
        """Create agent that extracts learnings from failed steps."""
        model = self._get_model()
        return Agent(
            model,
            system_prompt=LEARNING_EXTRACTOR_PROMPT,
            output_type=Learning,
            retries=2,
        )

    def _categorize_task(self, prompt: str) -> str:
        """Categorize a task for learning retrieval."""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["room", "inspect", "installation", "pattern", "diagram"]):
            return "introspection"
        elif any(word in prompt_lower for word in ["rag", "document", "search", "index", "knowledge"]):
            return "rag"
        elif any(word in prompt_lower for word in ["fetch", "url", "download", "http"]):
            return "fetch"
        else:
            return "general"

    def _get_learning_store(self) -> tuple[LearningStore, LearningStoreManager]:
        """Get the learning store and manager."""
        manager = LearningStoreManager()
        store = manager.load()
        return store, manager

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, typing.Any],
        ctx: "IntrospectiveContext",
        step_results: list[StepResult],
    ) -> dict[str, typing.Any]:
        """Execute a single tool by name using the tool registry."""
        logger.info(f"[TOOL_EXEC] Executing tool: {tool_name}")
        logger.debug(f"[TOOL_EXEC] Raw args: {tool_args}")

        # Resolve placeholders like <RESULT_FROM_STEP_1>
        resolved_args = self._resolve_placeholders(tool_args, step_results)
        logger.debug(f"[TOOL_EXEC] Resolved args: {resolved_args}")

        # Look up tool in registry
        if tool_name not in TOOL_REGISTRY:
            logger.warning(f"[TOOL_EXEC] Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            spec = TOOL_REGISTRY[tool_name]
            return await spec.implementation(ctx, **resolved_args)
        except Exception as e:
            logger.error(f"[TOOL_EXEC] Tool {tool_name} failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _resolve_placeholders(
        self,
        args: dict[str, typing.Any],
        step_results: list[StepResult],
    ) -> dict[str, typing.Any]:
        """Resolve placeholders like <RESULT_FROM_STEP_1> in arguments."""
        import re

        resolved = {}
        for key, value in args.items():
            if isinstance(value, str):
                # Check for placeholder pattern
                match = re.search(r'<(?:RESULT|URL|CONTENT)_FROM_STEP_(\d+)>', value)
                if match:
                    step_num = int(match.group(1))
                    # Find the step result
                    for sr in step_results:
                        if sr.step_number == step_num and sr.success and sr.result:
                            # Replace with actual value from result
                            resolved[key] = sr.result
                            break
                    else:
                        resolved[key] = value  # Keep placeholder if not found
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved

    def _get_model(self):
        """Create model using soliplex configuration."""
        if self._model is None:
            installation_config = self.agent_config._installation_config
            provider_base_url = installation_config.get_environment("OLLAMA_BASE_URL")
            provider = ollama_providers.OllamaProvider(
                base_url=f"{provider_base_url}/v1",
            )
            self._model = openai_models.OpenAIChatModel(
                model_name=self.model_name,
                provider=provider,
            )
        return self._model

    def _create_agent(self) -> Agent:
        """Create the introspective agent with tools."""
        if self._agent is not None:
            return self._agent

        model = self._get_model()

        agent = Agent(
            model,
            system_prompt=SYSTEM_PROMPT,
            deps_type=IntrospectiveContext,
            retries=3,
        )

        # ============================================================
        # ROOM INSPECTION TOOLS
        # ============================================================

        @agent.tool
        def list_all_rooms(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> list[dict[str, typing.Any]]:
            """List ALL rooms in the current soliplex installation with their details.

            Returns a list of room information including id, name, description, and agent type.
            """
            installation = ctx.deps.installation_config
            rooms = []

            for room_id, room_config in installation.room_configs.items():
                agent_config = room_config.agent_config
                rooms.append({
                    "id": room_id,
                    "name": room_config.name,
                    "description": room_config.description,
                    "agent_kind": agent_config.kind,
                    "factory_name": getattr(agent_config, "factory_name", None),
                    "suggestions_count": len(room_config.suggestions),
                    "has_welcome": bool(room_config.welcome_message),
                })

            return sorted(rooms, key=lambda r: r["id"])

        @agent.tool
        def inspect_room(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            room_id: str,
        ) -> dict[str, typing.Any]:
            """Get FULL details about a specific room including its complete configuration.

            Args:
                room_id: The room ID to inspect (e.g., 'shark-tank', 'debate')
            """
            installation = ctx.deps.installation_config
            room_configs = installation.room_configs

            if room_id not in room_configs:
                return {
                    "error": f"Room '{room_id}' not found",
                    "available_rooms": list(room_configs.keys()),
                }

            room = room_configs[room_id]
            agent_config = room.agent_config

            return {
                "id": room.id,
                "name": room.name,
                "description": room.description,
                "welcome_message": room.welcome_message or "(none)",
                "suggestions": room.suggestions,
                "agent": {
                    "kind": agent_config.kind,
                    "factory_name": getattr(agent_config, "factory_name", None),
                    "extra_config": getattr(agent_config, "extra_config", {}),
                },
                "tool_configs": list(room.tool_configs.keys()) if room.tool_configs else [],
                "enable_attachments": room.enable_attachments,
                "config_path": str(room._config_path) if hasattr(room, "_config_path") else None,
            }

        @agent.tool
        def get_room_suggestions(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            room_id: str,
        ) -> dict[str, typing.Any]:
            """Get the example prompts/suggestions for a specific room.

            Args:
                room_id: The room ID
            """
            installation = ctx.deps.installation_config
            room_configs = installation.room_configs

            if room_id not in room_configs:
                return {"error": f"Room '{room_id}' not found"}

            room = room_configs[room_id]
            return {
                "room_id": room_id,
                "room_name": room.name,
                "suggestions": room.suggestions,
            }

        # ============================================================
        # CONFIGURATION TOOLS
        # ============================================================

        @agent.tool
        def get_installation_info(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> dict[str, typing.Any]:
            """Get information about the current soliplex installation."""
            installation = ctx.deps.installation_config

            return {
                "installation_id": installation.id,
                "config_path": str(installation._config_path),
                "room_paths": [str(p) for p in installation.room_paths],
                "total_rooms": len(installation.room_configs),
                "room_ids": sorted(installation.room_configs.keys()),
            }

        @agent.tool
        def get_environment_variables(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> dict[str, typing.Any]:
            """Get environment variables used by this installation."""
            installation = ctx.deps.installation_config

            # Get configured environment
            env = {}
            for key in ["OLLAMA_BASE_URL", "OLLAMA_MODEL", "LOGFIRE_TOKEN"]:
                value = installation.get_environment(key)
                if value:
                    # Mask sensitive values
                    if "TOKEN" in key or "SECRET" in key or "PASSWORD" in key:
                        env[key] = value[:8] + "..." if len(value) > 8 else "***"
                    else:
                        env[key] = value
                else:
                    env[key] = "(not set)"

            return {
                "configured_environment": env,
                "installation_environment": dict(installation.environment),
            }

        @agent.tool
        def get_secrets_info(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> dict[str, typing.Any]:
            """Get information about configured secrets (names only, not values)."""
            installation = ctx.deps.installation_config

            secrets_info = []
            for secret in installation.secrets:
                secrets_info.append({
                    "name": secret.secret_name,
                    "source_type": type(secret).__name__,
                })

            return {
                "secrets_count": len(secrets_info),
                "secrets": secrets_info,
            }

        # ============================================================
        # FACTORY AGENT TOOLS
        # ============================================================

        @agent.tool
        def list_factory_agents(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> list[dict[str, typing.Any]]:
            """List all factory agents used in this installation."""
            installation = ctx.deps.installation_config
            factories = {}

            for room_id, room_config in installation.room_configs.items():
                agent_config = room_config.agent_config
                if agent_config.kind == "factory":
                    factory_name = agent_config.factory_name
                    if factory_name not in factories:
                        factories[factory_name] = {
                            "factory_name": factory_name,
                            "rooms_using": [],
                        }
                    factories[factory_name]["rooms_using"].append(room_id)

            return list(factories.values())

        @agent.tool
        def inspect_factory(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            factory_name: str,
        ) -> dict[str, typing.Any]:
            """Inspect a factory agent by name.

            Args:
                factory_name: Full dotted path like 'crazy_glue.factories.debate_factory.create_debate_agent'
            """
            try:
                import importlib

                module_name, func_name = factory_name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                factory_func = getattr(module, func_name, None)

                if factory_func is None:
                    return {"error": f"Factory function '{func_name}' not found in module"}

                # Get docstring and signature
                doc = inspect.getdoc(factory_func) or "(no documentation)"
                sig = str(inspect.signature(factory_func))

                # Look for the agent class in the module
                agent_class = None
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.endswith("Agent") and name != "Agent":
                        agent_class = obj
                        break

                agent_info = {}
                if agent_class:
                    agent_info = {
                        "agent_class": agent_class.__name__,
                        "agent_doc": inspect.getdoc(agent_class) or "(no documentation)",
                    }

                return {
                    "factory_name": factory_name,
                    "module": module_name,
                    "function": func_name,
                    "signature": sig,
                    "documentation": doc,
                    **agent_info,
                }

            except Exception as e:
                return {"error": str(e)}

        # ============================================================
        # PATTERN EXPLANATION TOOLS
        # ============================================================

        @agent.tool
        def explain_pattern(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            pattern_name: str,
        ) -> dict[str, typing.Any]:
            """Explain an agentic design pattern in detail.

            Args:
                pattern_name: Pattern to explain (routing, reflection, planning, parallelization, tool_use, multi_agent)
            """
            pattern_key = pattern_name.lower().replace(" ", "_").replace("-", "_")

            if pattern_key not in PATTERN_KNOWLEDGE:
                return {
                    "error": f"Unknown pattern '{pattern_name}'",
                    "available_patterns": list(PATTERN_KNOWLEDGE.keys()),
                }

            return PATTERN_KNOWLEDGE[pattern_key]

        @agent.tool
        def list_patterns(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> list[dict[str, str]]:
            """List all known agentic design patterns with brief descriptions."""
            return [
                {"name": info["name"], "description": info["description"][:100] + "..."}
                for info in PATTERN_KNOWLEDGE.values()
            ]

        @agent.tool
        def find_rooms_by_pattern(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            pattern_name: str,
        ) -> dict[str, typing.Any]:
            """Find which rooms use a specific pattern.

            Args:
                pattern_name: Pattern to search for
            """
            installation = ctx.deps.installation_config

            # Map factory names to patterns (based on known factories)
            factory_patterns = {
                "routing_factory": ["routing"],
                "reflection_factory": ["reflection"],
                "planning_factory": ["planning"],
                "parallelization_factory": ["parallelization"],
                "debate_factory": ["parallelization", "multi_agent"],
                "brainstorm_factory": ["parallelization", "multi_agent"],
                "code_review_factory": ["reflection", "multi_agent"],
                "shark_tank_factory": ["planning", "parallelization", "multi_agent"],
                "introspective_factory": ["tool_use"],
            }

            pattern_key = pattern_name.lower().replace(" ", "_").replace("-", "_")
            matching_rooms = []

            for room_id, room_config in installation.room_configs.items():
                agent_config = room_config.agent_config
                if agent_config.kind == "factory":
                    factory_key = agent_config.factory_name.split(".")[-1].replace("create_", "")
                    patterns = factory_patterns.get(factory_key, [])
                    if pattern_key in patterns:
                        matching_rooms.append({
                            "room_id": room_id,
                            "room_name": room_config.name,
                            "factory": agent_config.factory_name,
                        })

            return {
                "pattern": pattern_name,
                "rooms_count": len(matching_rooms),
                "rooms": matching_rooms,
            }

        # ============================================================
        # DIAGRAM GENERATION TOOLS
        # ============================================================

        @agent.tool
        def generate_installation_diagram(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> str:
            """Generate a mermaid diagram showing the installation architecture."""
            installation = ctx.deps.installation_config
            rooms = list(installation.room_configs.keys())

            room_nodes = "\n        ".join([f'R{i}["{r}"]' for i, r in enumerate(rooms)])

            return f"""```mermaid
flowchart TB
    subgraph Soliplex
        CLI[soliplex-cli serve]
        Core[Room Loader]
    end

    subgraph "Crazy Glue Installation"
        subgraph Rooms
        {room_nodes}
        end
        Factories[Factory Agents]
    end

    subgraph External
        Ollama[Ollama LLM]
    end

    CLI --> Core
    Core --> Rooms
    Rooms --> Factories
    Factories --> Ollama
```"""

        @agent.tool
        def generate_room_diagram(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            room_id: str,
        ) -> str:
            """Generate a mermaid diagram for a specific room's data flow.

            Args:
                room_id: The room to diagram
            """
            installation = ctx.deps.installation_config
            room_configs = installation.room_configs

            if room_id not in room_configs:
                return f"Error: Room '{room_id}' not found"

            room = room_configs[room_id]
            name = room.name

            return f"""```mermaid
flowchart TB
    User[User] -->|message| Room["{name}"]
    Room --> Factory[Factory Agent]
    Factory --> LLM[Ollama]
    LLM --> Factory
    Factory -->|AG-UI events| Room
    Room -->|response| User
```"""

        # ============================================================
        # HEALTH CHECK TOOLS
        # ============================================================

        @agent.tool
        def run_health_check(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> list[dict[str, typing.Any]]:
            """Run health checks on the installation."""
            installation = ctx.deps.installation_config
            results = []

            # Check rooms
            room_count = len(installation.room_configs)
            results.append({
                "component": "rooms",
                "status": "healthy" if room_count > 0 else "warning",
                "message": f"Found {room_count} rooms",
            })

            # Check Ollama URL
            ollama_url = installation.get_environment("OLLAMA_BASE_URL")
            results.append({
                "component": "ollama",
                "status": "healthy" if ollama_url else "error",
                "message": f"Ollama URL: {ollama_url or 'NOT CONFIGURED'}",
            })

            # Check installation config
            results.append({
                "component": "installation",
                "status": "healthy",
                "message": f"Installation ID: {installation.id}",
            })

            # Check for factory agents
            factory_count = sum(
                1
                for rc in installation.room_configs.values()
                if rc.agent_config.kind == "factory"
            )
            results.append({
                "component": "factories",
                "status": "healthy",
                "message": f"Found {factory_count} factory agents",
            })

            return results

        # ============================================================
        # SELF-AWARENESS TOOLS
        # ============================================================

        @agent.tool
        def who_am_i(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> dict[str, typing.Any]:
            """Get information about THIS agent (the introspective agent itself)."""
            agent_config = ctx.deps.agent_config

            return {
                "i_am": "The Introspective Agent",
                "my_purpose": "I can explore and explain the soliplex installation I'm running in",
                "my_factory": agent_config.factory_name,
                "my_config": {
                    "model": agent_config.extra_config.get("model_name", "gpt-oss:20b"),
                    "extra_config": agent_config.extra_config,
                },
                "my_capabilities": [
                    "List and inspect rooms",
                    "Show configuration and environment",
                    "Explain agentic patterns",
                    "Generate architecture diagrams",
                    "Run health checks",
                ],
                "pattern_i_use": "tool_use",
            }

        @agent.tool
        def list_my_tools(
            ctx: ai_tools.RunContext[IntrospectiveContext],
        ) -> list[dict[str, str]]:
            """List all the tools I (the introspective agent) have access to."""
            # This is meta - listing tools from within a tool!
            tools = [
                {"name": "list_all_rooms", "purpose": "List all rooms in the installation"},
                {"name": "inspect_room", "purpose": "Get full details about a specific room"},
                {"name": "get_room_suggestions", "purpose": "Get example prompts for a room"},
                {"name": "get_installation_info", "purpose": "Get installation configuration"},
                {"name": "get_environment_variables", "purpose": "Show environment variables"},
                {"name": "get_secrets_info", "purpose": "List configured secrets"},
                {"name": "list_factory_agents", "purpose": "List all factory agents"},
                {"name": "inspect_factory", "purpose": "Inspect a specific factory"},
                {"name": "explain_pattern", "purpose": "Explain an agentic pattern"},
                {"name": "list_patterns", "purpose": "List all known patterns"},
                {"name": "find_rooms_by_pattern", "purpose": "Find rooms using a pattern"},
                {"name": "generate_installation_diagram", "purpose": "Generate architecture diagram"},
                {"name": "generate_room_diagram", "purpose": "Generate room-specific diagram"},
                {"name": "run_health_check", "purpose": "Run health checks"},
                {"name": "who_am_i", "purpose": "Get info about this agent"},
                {"name": "list_my_tools", "purpose": "List all my tools (this one!)"},
                # RAG tools
                {"name": "rag_info", "purpose": "Get RAG system status and document count"},
                {"name": "rag_add_text", "purpose": "Add text content to the RAG index"},
                {"name": "rag_add_url", "purpose": "Add content from a URL to the RAG index"},
                {"name": "rag_add_file", "purpose": "Add a local file to the RAG index"},
                {"name": "rag_search", "purpose": "Search the RAG index"},
                {"name": "rag_list_documents", "purpose": "List documents in the RAG index"},
                {"name": "rag_delete_document", "purpose": "Delete a document from the RAG index"},
                # HTTP fetch tools
                {"name": "fetch_url_to_file", "purpose": "Download URL to temp file (for PDFs, etc.)"},
                {"name": "fetch_url_content", "purpose": "Fetch URL and return text content"},
            ]
            return tools

        # ============================================================
        # RAG TOOLS (via haiku-rag MCP) - Wrappers around TOOL_REGISTRY
        # ============================================================

        @agent.tool
        async def rag_info(ctx: ai_tools.RunContext[IntrospectiveContext]) -> dict[str, typing.Any]:
            """Get RAG system status and document count."""
            return await _tool_rag_info(ctx.deps)

        @agent.tool
        async def rag_add_text(ctx: ai_tools.RunContext[IntrospectiveContext], content: str, title: str) -> dict[str, typing.Any]:
            """Add text content to the RAG knowledge base."""
            return await _tool_rag_add_text(ctx.deps, content=content, title=title)

        @agent.tool
        async def rag_add_url(ctx: ai_tools.RunContext[IntrospectiveContext], url: str, title: str = None) -> dict[str, typing.Any]:
            """Add content from a URL to the RAG knowledge base."""
            return await _tool_rag_add_url(ctx.deps, url=url, title=title)

        @agent.tool
        async def rag_add_file(ctx: ai_tools.RunContext[IntrospectiveContext], file_path: str, title: str = None) -> dict[str, typing.Any]:
            """Add a local file to the RAG knowledge base."""
            return await _tool_rag_add_file(ctx.deps, file_path=file_path, title=title)

        @agent.tool
        async def rag_search(ctx: ai_tools.RunContext[IntrospectiveContext], query: str, limit: int = 5) -> dict[str, typing.Any]:
            """Search the RAG knowledge base."""
            return await _tool_rag_search(ctx.deps, query=query, limit=limit)

        @agent.tool
        async def rag_list_documents(ctx: ai_tools.RunContext[IntrospectiveContext], limit: int = 20) -> dict[str, typing.Any]:
            """List documents in the RAG knowledge base."""
            return await _tool_rag_list_documents(ctx.deps, limit=limit)

        @agent.tool
        async def rag_delete_document(ctx: ai_tools.RunContext[IntrospectiveContext], document_id: str) -> dict[str, typing.Any]:
            """Delete a document from the RAG knowledge base."""
            return await _tool_rag_delete_document(ctx.deps, document_id=document_id)

        # ============================================================
        # HTTP FETCH TOOLS
        # ============================================================

        @agent.tool
        async def fetch_url_to_file(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            url: str,
        ) -> dict[str, typing.Any]:
            """Fetch a remote URL and save it to a temporary file.

            Downloads content from the URL and saves it locally. Useful for
            fetching PDFs, documents, or other files that can then be added
            to the RAG knowledge base.

            Args:
                url: The URL to fetch (supports http/https)

            Returns:
                Dictionary with 'file_path' (local temp file), 'content_type',
                'size_bytes', and 'filename'.
            """
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    return {"error": f"Unsupported URL scheme: {parsed.scheme}"}

                async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                # Determine filename from URL or content-disposition
                filename = Path(parsed.path).name or "downloaded_file"
                content_disposition = response.headers.get("content-disposition", "")
                if "filename=" in content_disposition:
                    # Extract filename from header
                    parts = content_disposition.split("filename=")
                    if len(parts) > 1:
                        filename = parts[1].strip('"\'')

                # Determine extension from content-type if not in filename
                content_type = response.headers.get("content-type", "application/octet-stream")
                if "." not in filename:
                    ext_map = {
                        "application/pdf": ".pdf",
                        "text/html": ".html",
                        "text/plain": ".txt",
                        "application/json": ".json",
                        "text/markdown": ".md",
                        "image/png": ".png",
                        "image/jpeg": ".jpg",
                    }
                    for mime, ext in ext_map.items():
                        if mime in content_type:
                            filename += ext
                            break

                # Save to temp file (won't be auto-deleted)
                temp_dir = tempfile.gettempdir()
                temp_path = Path(temp_dir) / f"introspective_{uuid.uuid4().hex[:8]}_{filename}"

                temp_path.write_bytes(response.content)

                return {
                    "success": True,
                    "file_path": str(temp_path),
                    "filename": filename,
                    "content_type": content_type.split(";")[0],
                    "size_bytes": len(response.content),
                    "url": url,
                }

            except httpx.HTTPStatusError as e:
                return {"error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}"}
            except httpx.RequestError as e:
                return {"error": f"Request failed: {str(e)}"}
            except Exception as e:
                return {"error": str(e)}

        @agent.tool
        async def fetch_url_content(
            ctx: ai_tools.RunContext[IntrospectiveContext],
            url: str,
            max_length: int = 50000,
        ) -> dict[str, typing.Any]:
            """Fetch a URL and return its text content directly.

            Best for HTML pages, text files, JSON, etc. For binary files
            like PDFs, use fetch_url_to_file instead.

            Args:
                url: The URL to fetch
                max_length: Maximum characters to return (default 50000)

            Returns:
                Dictionary with 'content', 'content_type', and 'url'.
            """
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    return {"error": f"Unsupported URL scheme: {parsed.scheme}"}

                async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                # Check if binary content
                if any(t in content_type for t in ["image/", "audio/", "video/", "application/pdf", "application/zip"]):
                    return {
                        "error": f"Binary content type ({content_type}). Use fetch_url_to_file instead.",
                        "content_type": content_type,
                    }

                text = response.text
                truncated = len(text) > max_length

                return {
                    "success": True,
                    "content": text[:max_length],
                    "content_type": content_type.split(";")[0],
                    "length": len(text),
                    "truncated": truncated,
                    "url": url,
                }

            except httpx.HTTPStatusError as e:
                return {"error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}"}
            except httpx.RequestError as e:
                return {"error": f"Request failed: {str(e)}"}
            except Exception as e:
                return {"error": str(e)}

        self._agent = agent
        return agent

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Execute using Planning pattern: Plan → Execute → Synthesize."""
        prompt = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        logger.info(f"[INTROSPECTIVE] Starting run_stream_events")
        logger.info(f"[INTROSPECTIVE] Prompt: {prompt[:100]}...")
        logger.info(f"[INTROSPECTIVE] Emitter present: {emitter is not None}")

        rag_toolset = self._get_rag_toolset()
        ctx = IntrospectiveContext(
            agent_config=self.agent_config,
            rag_toolset=rag_toolset,
        )

        # Load conversation memory
        memory_manager = MemoryManager()
        memory = memory_manager.load()
        memory_context = memory.get_recent_summary(n=5)  # Last 5 exchanges
        logger.info(f"[MEMORY] Loaded {len(memory.messages)} messages from memory")

        # Save user's message to memory
        memory.add_user_message(prompt)

        part_index = 0

        # ============================================================
        # PHASE 1: PLANNING (with injected learnings and memory)
        # ============================================================
        if emitter:
            emitter.update_activity(
                "introspective",
                {"status": "planning", "query": prompt[:100], "message": "Creating execution plan..."},
                activity_id,
            )

        think_part = ai_messages.ThinkingPart("Planning...")
        logger.debug(f"[EVENT] Yielding PartStartEvent for ThinkingPart (index={part_index})")
        yield ai_messages.PartStartEvent(index=part_index, part=think_part)

        # Fetch relevant learnings from previous failures
        # 1. Try semantic search in RAG first (if available)
        rag_learning_context = ""
        if rag_toolset:
            try:
                async with rag_toolset:
                    search_results = await rag_toolset.direct_call_tool(
                        "search_documents",
                        {"query": f"LEARNING planning {prompt[:100]}", "limit": 3},
                    )
                if search_results:
                    rag_learning_context = "\n\n## RELEVANT PAST LEARNINGS (from RAG)\n\n"
                    for i, result in enumerate(search_results, 1):
                        # Extract content from search result
                        content = result.get("content", result.get("text", str(result)))
                        if "LEARNING:" in str(content):
                            rag_learning_context += f"{i}. {content[:500]}\n\n"
                    if rag_learning_context.strip().endswith("(from RAG)"):
                        rag_learning_context = ""  # No actual learnings found
                    else:
                        logger.info(f"[LEARNING] Found {len(search_results)} semantic matches in RAG")
            except Exception as e:
                logger.warning(f"[LEARNING] RAG search failed: {e}")

        # 2. Fall back to category-based retrieval from JSON
        store, _ = self._get_learning_store()
        task_category = self._categorize_task(prompt)
        relevant_learnings = store.get_relevant(task_category, k=3)

        if relevant_learnings:
            logger.info(f"[LEARNING] Injecting {len(relevant_learnings)} category-based learnings for '{task_category}'")

        planner = self._create_planner_agent(
            learnings=relevant_learnings,
            rag_context=rag_learning_context,
            memory_context=memory_context,
        )
        logger.info(f"[INTROSPECTIVE] Created planner agent, calling run()")
        try:
            plan_result = await planner.run(f"Create a plan for: {prompt}")
            plan = plan_result.output
            logger.info(f"[INTROSPECTIVE] Plan created: {len(plan.steps)} steps")
            for step in plan.steps:
                logger.info(f"[INTROSPECTIVE]   Step {step.step_number}: {step.tool_name}({step.tool_args})")
        except Exception as e:
            logger.error(f"[INTROSPECTIVE] Planning failed: {e}")
            logger.info(f"[INTROSPECTIVE] Falling back to simple agent execution")
            # Fallback: use the regular agent with tools instead of planning pattern
            yield ai_messages.PartEndEvent(index=0, part=think_part)
            agent = self._create_agent()
            ctx = IntrospectiveContext(
                agent_config=self.agent_config,
                rag_toolset=rag_toolset,
            )
            async for event in agent.iter(prompt, deps=ctx):
                yield event
            return

        yield ai_messages.PartDeltaEvent(
            index=0,
            delta=ai_messages.ThinkingPartDelta(
                content_delta=f"\n✓ Plan created: {len(plan.steps)} steps\n{plan.reasoning}"
            ),
        )

        # ============================================================
        # PHASE 2: EXECUTION
        # ============================================================
        if emitter:
            emitter.update_activity(
                "introspective",
                {
                    "status": "executing",
                    "query": prompt[:100],
                    "total_steps": len(plan.steps),
                    "message": f"Executing {len(plan.steps)} steps...",
                },
                activity_id,
            )

        step_results: list[StepResult] = []

        for step in plan.steps:
            yield ai_messages.PartDeltaEvent(
                index=0,
                delta=ai_messages.ThinkingPartDelta(
                    content_delta=f"\n\nStep {step.step_number}: {step.tool_name}"
                ),
            )

            if emitter:
                emitter.update_activity(
                    "introspective",
                    {
                        "status": "executing",
                        "current_step": step.step_number,
                        "total_steps": len(plan.steps),
                        "tool": step.tool_name,
                        "message": step.description,
                    },
                    activity_id,
                )

            # Emit tool call event
            part_index += 1
            tc_args_str = json.dumps(step.tool_args)
            tc_part = ai_messages.ToolCallPart(tool_name=step.tool_name, args=tc_args_str)
            logger.info(f"[EVENT] Yielding ToolCallPart START for '{step.tool_name}' args={tc_args_str} (index={part_index})")
            yield ai_messages.PartStartEvent(index=part_index, part=tc_part)

            # Execute the tool
            logger.info(f"[EXECUTION] Calling _execute_tool for step {step.step_number}: {step.tool_name}")
            result = await self._execute_tool(
                step.tool_name,
                step.tool_args,
                ctx,
                step_results,
            )
            logger.info(f"[EXECUTION] Tool '{step.tool_name}' returned: {str(result)[:200]}...")

            success = "error" not in result
            step_results.append(StepResult(
                step_number=step.step_number,
                tool_name=step.tool_name,
                success=success,
                result=result if success else None,
                error=result.get("error") if not success else None,
            ))
            logger.info(f"[EXECUTION] Step {step.step_number} success={success}")

            logger.info(f"[EVENT] Yielding ToolCallPart END for '{step.tool_name}' (index={part_index})")
            yield ai_messages.PartEndEvent(index=part_index, part=tc_part)

            status = "✓" if success else "✗"
            yield ai_messages.PartDeltaEvent(
                index=0,
                delta=ai_messages.ThinkingPartDelta(
                    content_delta=f" {status}"
                ),
            )

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # ============================================================
        # PHASE 3: SYNTHESIS
        # ============================================================
        if emitter:
            emitter.update_activity(
                "introspective",
                {"status": "synthesizing", "message": "Summarizing results..."},
                activity_id,
            )

        logger.info(f"[INTROSPECTIVE] Creating synthesizer agent")
        synthesizer = self._create_synthesizer_agent()

        # Build synthesis prompt
        results_summary = []
        for sr in step_results:
            if sr.success:
                results_summary.append(f"Step {sr.step_number} ({sr.tool_name}): SUCCESS\n  Result: {sr.result}")
            else:
                results_summary.append(f"Step {sr.step_number} ({sr.tool_name}): FAILED\n  Error: {sr.error}")

        synthesis_prompt = f"""Goal: {plan.goal}

Plan: {plan.reasoning}

Execution Results:
{chr(10).join(results_summary)}

Provide a summary of what was accomplished."""

        try:
            logger.info(f"[INTROSPECTIVE] Running synthesizer with {len(results_summary)} step results")
            synthesis_result = await synthesizer.run(synthesis_prompt)
            response = synthesis_result.output
            logger.info(f"[INTROSPECTIVE] Synthesis complete, response length={len(response)}")
        except Exception as e:
            logger.error(f"[INTROSPECTIVE] Synthesis failed: {e}", exc_info=True)
            response = f"Execution completed but synthesis failed: {e}\n\nRaw results:\n" + "\n".join(results_summary)

        # ============================================================
        # PHASE 4: SELF-IMPROVEMENT
        # ============================================================
        failed_steps = [sr for sr in step_results if not sr.success]
        learnings_text = ""

        # Also check if synthesis identified issues (incomplete plan, partial success)
        synthesis_indicates_issues = any(
            phrase in response.lower()
            for phrase in ["failure", "failed", "issue", "incomplete", "only one", "only the first", "not all", "missing"]
        )

        should_extract_learnings = failed_steps or synthesis_indicates_issues

        if should_extract_learnings:
            if emitter:
                emitter.update_activity(
                    "introspective",
                    {"status": "learning", "message": "Extracting learnings from failures..."},
                    activity_id,
                )

            learning_extractor = self._create_learning_extractor_agent()
            store, manager = self._get_learning_store()
            extracted_learnings = []

            # Extract learnings from failed steps
            if failed_steps:
                logger.info(f"[LEARNING] Extracting learnings from {len(failed_steps)} failed steps")

            for failed_step in failed_steps:
                try:
                    # Find original plan step for context
                    original_step = next(
                        (s for s in plan.steps if s.step_number == failed_step.step_number),
                        None
                    )

                    extraction_prompt = f"""
Goal: {plan.goal}
Failed Tool: {failed_step.tool_name}
Arguments: {json.dumps(failed_step.result) if failed_step.result else '{}'}
Error: {failed_step.error}
Step Description: {original_step.description if original_step else 'N/A'}
"""
                    learning_result = await learning_extractor.run(extraction_prompt)
                    learning = learning_result.output
                    learning.original_goal = plan.goal
                    learning.failed_args = original_step.tool_args if original_step else {}

                    store.add(learning)
                    extracted_learnings.append(learning)
                    logger.info(f"[LEARNING] Extracted: {learning.error_pattern}")

                except Exception as e:
                    logger.warning(f"[LEARNING] Failed to extract learning: {e}")

            # Extract learnings from synthesis-identified issues (incomplete plans, partial success)
            if synthesis_indicates_issues and not failed_steps:
                logger.info(f"[LEARNING] Extracting learnings from synthesis-identified issues")
                try:
                    extraction_prompt = f"""
Goal: {plan.goal}
Plan Steps: {len(plan.steps)} steps were planned
Synthesis Result: {response}

The synthesis indicates issues even though individual steps succeeded.
Analyze what went wrong with the PLANNING (not execution) and extract a learning.
Focus on: Was the plan incomplete? Did it miss requirements? Did it only handle part of the task?
"""
                    learning_result = await learning_extractor.run(extraction_prompt)
                    learning = learning_result.output
                    learning.original_goal = plan.goal
                    learning.task_category = task_category

                    store.add(learning)
                    extracted_learnings.append(learning)
                    logger.info(f"[LEARNING] Extracted planning issue: {learning.error_pattern}")

                except Exception as e:
                    logger.warning(f"[LEARNING] Failed to extract learning from synthesis: {e}")

            if extracted_learnings:
                manager.save(store)

                # Also index learnings in RAG for semantic retrieval
                if rag_toolset:
                    for learning in extracted_learnings:
                        try:
                            learning_text = f"""LEARNING: {learning.error_pattern}

Goal: {learning.original_goal}
Problem: {learning.error_pattern}
Tool: {learning.failed_tool}
Solution: {learning.correct_approach}
Example: {learning.example_correction}
Category: {learning.task_category}"""

                            async with rag_toolset:
                                await rag_toolset.direct_call_tool(
                                    "add_document_from_text",
                                    {
                                        "content": learning_text,
                                        "title": f"Learning: {learning.error_pattern[:50]}",
                                    },
                                )
                            logger.info(f"[LEARNING] Indexed learning in RAG: {learning.id}")
                        except Exception as e:
                            logger.warning(f"[LEARNING] Failed to index learning in RAG: {e}")

                # Build learnings section for response
                learnings_text = "\n\n## What I Learned\n\n"
                for i, learning in enumerate(extracted_learnings, 1):
                    learnings_text += f"{i}. **{learning.error_pattern}**\n"
                    learnings_text += f"   - Correct approach: {learning.correct_approach}\n"
                    learnings_text += f"   - Example: {learning.example_correction}\n\n"

                response = response + learnings_text

        if emitter:
            emitter.update_activity(
                "introspective",
                {
                    "status": "complete",
                    "steps_completed": len([s for s in step_results if s.success]),
                    "steps_failed": len([s for s in step_results if not s.success]),
                    "learnings_extracted": len(failed_steps) if failed_steps else 0,
                    "message": "Complete",
                },
                activity_id,
            )

        # Yield final response
        part_index += 1
        text_part = ai_messages.TextPart(response)
        logger.info(f"[EVENT] Yielding final TextPart (index={part_index}), length={len(response)}")
        yield ai_messages.PartStartEvent(index=part_index, part=text_part)
        yield ai_messages.PartEndEvent(index=part_index, part=text_part)

        # Save assistant's response to memory
        memory.add_assistant_message(response)
        memory_manager.save(memory)
        logger.info(f"[MEMORY] Saved conversation to memory ({len(memory.messages)} messages)")

        logger.info(f"[EVENT] Yielding AgentRunResultEvent")
        yield ai_run.AgentRunResultEvent(result=response)
        logger.info(f"[INTROSPECTIVE] run_stream_events complete")


def create_introspective_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> IntrospectiveAgent:
    """Factory function to create the introspective agent."""
    return IntrospectiveAgent(agent_config, tool_configs, mcp_client_toolset_configs)
