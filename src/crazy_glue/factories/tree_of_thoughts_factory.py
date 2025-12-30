"""Factory for Tree of Thoughts Room (17b - Multi-level Exploration)."""

from __future__ import annotations

import dataclasses
import typing
import uuid
from collections import abc

from agentic_patterns.thought_candidates import OutputConfig
from agentic_patterns.thought_candidates import ProblemStatement
from agentic_patterns.thought_candidates import create_evaluator_agent
from agentic_patterns.thought_candidates import create_generator_agent
from agentic_patterns.tree_of_thoughts import TreeConfig
from agentic_patterns.tree_of_thoughts import TreeExplorationResult
from agentic_patterns.tree_of_thoughts import create_synthesizer_agent
from agentic_patterns.tree_of_thoughts import run_tree_of_thoughts
from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools
from pydantic_ai.models import openai as openai_models
from pydantic_ai.providers import ollama as ollama_providers
from soliplex import config

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = (
    ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]
)


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


def _format_result(result: TreeExplorationResult) -> str:
    """Format the tree exploration result as markdown."""
    lines = [
        "# Tree of Thoughts Exploration\n",
        f"**Problem:** {result.problem.description}\n",
    ]

    if result.problem.constraints:
        constraints = ", ".join(result.problem.constraints)
        lines.append(f"**Constraints:** {constraints}\n")

    cfg = result.config
    lines.append(
        f"\n**Config:** depth={cfg.max_depth}, branch={cfg.branch_factor}, "
        f"prune>={cfg.prune_threshold}, beam={cfg.beam_width}\n"
    )

    # Statistics
    lines.append("\n## Exploration Statistics\n")
    lines.append(f"- **Nodes explored:** {result.nodes_explored}")
    lines.append(f"- **Nodes pruned:** {result.nodes_pruned}")
    lines.append(f"- **Best path length:** {len(result.best_path)}\n")

    # Best path
    lines.append("\n## Best Reasoning Path\n")
    for i, node in enumerate(result.best_path):
        score = node.score
        content = node.scored_thought.thought.content
        lines.append(f"\n### Step {i + 1} (Score: {score:.1f}/10)\n")
        lines.append(f"{content}\n")
        lines.append(f"*Reasoning: {node.scored_thought.thought.reasoning}*\n")

    # Final solution
    lines.append("\n---\n")
    lines.append("## Final Solution\n")
    lines.append(f"**Confidence:** {result.solution.confidence:.0%}\n")
    lines.append(f"\n{result.solution.solution}\n")
    lines.append(f"\n**Derivation:** {result.solution.reasoning}\n")

    # Tree visualization (simplified)
    lines.append("\n---\n")
    lines.append("## Tree Overview\n")
    lines.append("```")

    best_path_ids = {n.id for n in result.best_path}

    # Group nodes by depth
    by_depth: dict[int, list] = {}
    for node in result.all_nodes:
        by_depth.setdefault(node.depth, []).append(node)

    for depth in sorted(by_depth.keys()):
        nodes = sorted(by_depth[depth], key=lambda n: n.score, reverse=True)
        lines.append(f"\nDepth {depth}:")
        for node in nodes:
            marker = "*" if node.id in best_path_ids else " "
            prune = " [pruned]" if node.is_pruned else ""
            lines.append(f"  {marker} [{node.score:.1f}] {node.id}{prune}")

    lines.append("```")
    lines.append("\n*Nodes marked with * are on the best path*\n")

    return "\n".join(lines)


@dataclasses.dataclass
class TreeOfThoughtsAgent:
    """Agent that runs Tree of Thoughts exploration with AG-UI activities."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None
    _fast_model: typing.Any = dataclasses.field(default=None, repr=False)
    _strong_model: typing.Any = dataclasses.field(default=None, repr=False)

    @property
    def fast_model_name(self) -> str:
        """Model for generation (fast, creative) - from room config."""
        return self.agent_config.extra_config["fast_model_name"]

    @property
    def strong_model_name(self) -> str:
        """Model for evaluation/synthesis (strong) - from room config."""
        return self.agent_config.extra_config["strong_model_name"]

    def _get_fast_model(self) -> typing.Any:
        """Create fast model using soliplex configuration."""
        if self._fast_model is None:
            installation_config = self.agent_config._installation_config
            base_url = installation_config.get_environment("OLLAMA_BASE_URL")
            provider = ollama_providers.OllamaProvider(
                base_url=f"{base_url}/v1",
            )
            self._fast_model = openai_models.OpenAIChatModel(
                model_name=self.fast_model_name,
                provider=provider,
            )
        return self._fast_model

    def _get_strong_model(self) -> typing.Any:
        """Create strong model using soliplex configuration."""
        if self._strong_model is None:
            installation_config = self.agent_config._installation_config
            base_url = installation_config.get_environment("OLLAMA_BASE_URL")
            provider = ollama_providers.OllamaProvider(
                base_url=f"{base_url}/v1",
            )
            self._strong_model = openai_models.OpenAIChatModel(
                model_name=self.strong_model_name,
                provider=provider,
            )
        return self._strong_model

    @property
    def max_depth(self) -> int:
        return self.agent_config.extra_config.get("max_depth", 3)

    @property
    def branch_factor(self) -> int:
        return self.agent_config.extra_config.get("branch_factor", 3)

    @property
    def prune_threshold(self) -> float:
        return self.agent_config.extra_config.get("prune_threshold", 5.0)

    @property
    def beam_width(self) -> int:
        return self.agent_config.extra_config.get("beam_width", 2)

    @property
    def max_words(self) -> int:
        return self.agent_config.extra_config.get("max_words", 100)

    @property
    def constraints(self) -> list[str]:
        return self.agent_config.extra_config.get("constraints", [])

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Stream the tree exploration with AG-UI updates."""
        user_prompt = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        # Build problem statement
        problem = ProblemStatement(
            description=user_prompt,
            constraints=self.constraints,
        )

        tree_config = TreeConfig(
            max_depth=self.max_depth,
            branch_factor=self.branch_factor,
            prune_threshold=self.prune_threshold,
            beam_width=self.beam_width,
        )

        output_config = OutputConfig(
            max_words=self.max_words,
            ascii_only=True,
        )

        # Initial activity
        if emitter:
            emitter.update_activity(
                "tree_of_thoughts",
                {
                    "status": "starting",
                    "max_depth": self.max_depth,
                    "branch_factor": self.branch_factor,
                    "problem": user_prompt[:50],
                },
                activity_id,
            )

        # Start thinking
        depth, branch = self.max_depth, self.branch_factor
        think_part = ai_messages.ThinkingPart(
            f"Tree of Thoughts: depth={depth}, branch={branch}"
        )
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        delta = f"\nProblem: {user_prompt[:40]}..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Exploring
        if emitter:
            emitter.update_activity(
                "tree_of_thoughts",
                {
                    "status": "exploring",
                    "max_depth": self.max_depth,
                    "current_depth": 0,
                },
                activity_id,
            )

        delta = "\nGenerating root candidates..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Run tree of thoughts with injected models
        result = await run_tree_of_thoughts(
            problem,
            tree_config,
            output_config,
            generator=create_generator_agent(self._get_fast_model()),
            evaluator=create_evaluator_agent(self._get_strong_model()),
            synthesizer=create_synthesizer_agent(self._get_strong_model()),
        )

        # Synthesizing
        if emitter:
            emitter.update_activity(
                "tree_of_thoughts",
                {
                    "status": "synthesizing",
                    "nodes_explored": result.nodes_explored,
                    "nodes_pruned": result.nodes_pruned,
                },
                activity_id,
            )

        explored, pruned = result.nodes_explored, result.nodes_pruned
        delta = f"\nExplored {explored} nodes, pruned {pruned}"
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        delta = f"\nBest path: {len(result.best_path)} steps"
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Complete
        if emitter:
            emitter.update_activity(
                "tree_of_thoughts",
                {
                    "status": "complete",
                    "nodes_explored": result.nodes_explored,
                    "nodes_pruned": result.nodes_pruned,
                    "best_path_length": len(result.best_path),
                    "confidence": result.solution.confidence,
                    "problem": user_prompt[:50],
                },
                activity_id,
            )

        delta = f"\nSolution confidence: {result.solution.confidence:.0%}"
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Final response
        response = _format_result(result)
        text_part = ai_messages.TextPart(response)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response)


def create_tree_of_thoughts_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> TreeOfThoughtsAgent:
    """Factory function to create the tree of thoughts agent."""
    return TreeOfThoughtsAgent(
        agent_config, tool_configs, mcp_client_toolset_configs
    )
