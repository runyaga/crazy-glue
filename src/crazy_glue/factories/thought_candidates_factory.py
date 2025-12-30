"""Factory for Thought Candidates Room (17a - Best-of-N Sampling)."""

from __future__ import annotations

import dataclasses
import typing
import uuid
from collections import abc

from agentic_patterns.thought_candidates import BestOfNResult
from agentic_patterns.thought_candidates import OutputConfig
from agentic_patterns.thought_candidates import ProblemStatement
from agentic_patterns.thought_candidates import create_evaluator_agent
from agentic_patterns.thought_candidates import create_generator_agent
from agentic_patterns.thought_candidates import run_best_of_n
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


def _format_result(result: BestOfNResult) -> str:
    """Format the result as markdown."""
    lines = [
        "# Thought Candidates\n",
        f"**Problem:** {result.problem.description}\n",
    ]

    if result.problem.constraints:
        constraints = ", ".join(result.problem.constraints)
        lines.append(f"**Constraints:** {constraints}\n")

    lines.append(f"\n## Candidates ({result.generation_count} generated)\n")

    for i, candidate in enumerate(result.candidates):
        is_best = candidate == result.best
        marker = "**[BEST]**" if is_best else f"{i + 1}."
        valid = "valid" if candidate.evaluation.is_valid else "invalid"

        score = candidate.score
        lines.append(f"\n### {marker} Score: {score:.1f}/10 ({valid})\n")
        lines.append(f"**Approach:** {candidate.thought.content}\n")
        lines.append(f"**Reasoning:** {candidate.thought.reasoning}\n")
        lines.append(f"**Feedback:** {candidate.evaluation.feedback}\n")

    lines.append("\n---\n")
    lines.append("## Best Candidate\n")
    lines.append(f"**Score:** {result.best.score:.1f}/10\n")
    lines.append(f"**Approach:** {result.best.thought.content}\n")

    return "\n".join(lines)


@dataclasses.dataclass
class ThoughtCandidatesAgent:
    """Agent that runs Best-of-N thought generation with AG-UI activities."""

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
        """Model for evaluation (strong, analytical) - from room config."""
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
    def num_candidates(self) -> int:
        return self.agent_config.extra_config.get("num_candidates", 5)

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
        """Stream the thought generation with AG-UI updates."""
        user_prompt = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        # Build problem statement
        problem = ProblemStatement(
            description=user_prompt,
            constraints=self.constraints,
        )

        output_config = OutputConfig(
            max_words=self.max_words,
            ascii_only=True,
        )

        # Initial activity
        if emitter:
            emitter.update_activity(
                "thought_candidates",
                {
                    "status": "generating",
                    "num_candidates": self.num_candidates,
                    "problem": user_prompt[:50],
                },
                activity_id,
            )

        # Start thinking
        think_part = ai_messages.ThinkingPart(
            f"Generating {self.num_candidates} candidates..."
        )
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        # Update thinking as we progress
        delta = f"\nProblem: {user_prompt[:40]}..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Run best-of-n (this runs all candidates in parallel internally)
        if emitter:
            emitter.update_activity(
                "thought_candidates",
                {"status": "evaluating", "n": self.num_candidates},
                activity_id,
            )

        delta = "\nEvaluating candidates in parallel..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        result = await run_best_of_n(
            problem,
            n=self.num_candidates,
            config=output_config,
            generator=create_generator_agent(self._get_fast_model()),
            evaluator=create_evaluator_agent(self._get_strong_model()),
        )

        # Complete
        if emitter:
            emitter.update_activity(
                "thought_candidates",
                {
                    "status": "complete",
                    "num_candidates": self.num_candidates,
                    "best_score": result.best.score,
                    "problem": user_prompt[:50],
                },
                activity_id,
            )

        delta = f"\nBest score: {result.best.score:.1f}/10"
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


def create_thought_candidates_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> ThoughtCandidatesAgent:
    """Factory function to create the thought candidates agent."""
    return ThoughtCandidatesAgent(
        agent_config, tool_configs, mcp_client_toolset_configs
    )
