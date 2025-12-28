"""Factory for Introspective Agent - uses soliplex internals to explore itself."""

from __future__ import annotations

import dataclasses
import inspect
import os
import typing
import uuid
from collections import abc
from pathlib import Path

import pydantic
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


SYSTEM_PROMPT = """You are the Introspective Agent - a self-aware AI that can explore and explain the soliplex installation you're running in.

You have powerful tools to inspect the REAL, LIVE soliplex configuration:
- List and inspect actual rooms from the running installation
- Access environment variables and configuration
- Examine factory agents and their capabilities
- Generate architecture diagrams
- Explain agentic design patterns

When users ask questions, USE YOUR TOOLS to get accurate, real-time information.
Be helpful and explain technical concepts clearly.

You are running inside the crazy-glue installation which demonstrates agentic design patterns.
Use your tools liberally - they give you real access to the system!

IMPORTANT: Always use tools to answer questions about rooms, config, or the installation.
Don't guess - look it up!"""


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

    @property
    def model_name(self) -> str:
        return self.agent_config.extra_config.get("model_name", "gpt-oss:20b")

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
            ]
            return tools

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
        """Stream events from the introspective agent with tool call visibility."""
        prompt = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        if emitter:
            emitter.update_activity(
                "introspective",
                {
                    "status": "thinking",
                    "query": prompt[:100],
                    "message": "Analyzing query...",
                },
                activity_id,
            )

        agent = self._create_agent()
        ctx = IntrospectiveContext(agent_config=self.agent_config)

        # Start with thinking part
        part_index = 0
        think_part = ai_messages.ThinkingPart("Analyzing your question...")
        yield ai_messages.PartStartEvent(index=part_index, part=think_part)

        # Run the agent
        result = await agent.run(prompt, deps=ctx)

        # Extract tool calls from the result messages
        tool_calls_made = []
        for msg in result.all_messages():
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if isinstance(part, ai_messages.ToolCallPart):
                        tool_calls_made.append(part.tool_name)

        # Yield tool call events for each tool that was called
        for tool_name in tool_calls_made:
            if emitter:
                emitter.update_activity(
                    "introspective",
                    {
                        "status": "tool_call",
                        "query": prompt[:100],
                        "tool": tool_name,
                        "tools_used": tool_calls_made,
                        "message": f"Used tool: {tool_name}",
                    },
                    activity_id,
                )

            # Update thinking part
            delta = f"\n→ Called {tool_name}"
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(
                index=0,
                delta=ai_messages.ThinkingPartDelta(content_delta=delta),
            )

            # Yield tool call events
            part_index += 1
            tc_part = ai_messages.ToolCallPart(tool_name=tool_name)
            yield ai_messages.PartStartEvent(index=part_index, part=tc_part)
            yield ai_messages.PartEndEvent(index=part_index, part=tc_part)

        # End thinking part
        tool_count = len(tool_calls_made)
        delta = f"\n\nCompleted with {tool_count} tool call(s)."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0,
            delta=ai_messages.ThinkingPartDelta(content_delta=delta),
        )
        yield ai_messages.PartEndEvent(index=0, part=think_part)

        if emitter:
            emitter.update_activity(
                "introspective",
                {
                    "status": "complete",
                    "query": prompt[:100],
                    "tool_count": tool_count,
                    "tools_used": tool_calls_made,
                    "message": "Complete",
                },
                activity_id,
            )

        # Yield the text response
        response = result.output if hasattr(result, "output") else str(result.data)
        part_index += 1
        text_part = ai_messages.TextPart(response)
        yield ai_messages.PartStartEvent(index=part_index, part=text_part)
        yield ai_messages.PartEndEvent(index=part_index, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response)


def create_introspective_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> IntrospectiveAgent:
    """Factory function to create the introspective agent."""
    return IntrospectiveAgent(agent_config, tool_configs, mcp_client_toolset_configs)
