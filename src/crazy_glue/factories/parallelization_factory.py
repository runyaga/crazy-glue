"""Factory for Parallelization Pattern agent with AG-UI state and activities."""

from __future__ import annotations

import dataclasses
import re
import typing
import uuid
from collections import abc

import pydantic
from agentic_patterns.parallelization import ReducedSummary
from agentic_patterns.parallelization import SynthesizedResult
from agentic_patterns.parallelization import VotingOutcome
from agentic_patterns.parallelization import run_map_reduce
from agentic_patterns.parallelization import run_sectioning
from agentic_patterns.parallelization import run_voting
from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools
from soliplex import config

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]

COMMAND_PATTERN = re.compile(r"^/(\w+)\s+(.+)$", re.DOTALL)


# AG-UI State model for parallelization workflow
class ParallelState(pydantic.BaseModel):
    """Track the parallelization workflow state for AG-UI."""

    strategy: str | None = None  # sectioning, voting, mapreduce
    stage: str = "idle"  # idle, spawning, executing, aggregating, complete
    worker_count: int = 0
    workers_complete: int = 0
    input_preview: str | None = None
    # Strategy-specific data
    sections: list[str] | None = None
    votes: list[dict] | None = None
    documents: list[str] | None = None


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


def _parse_command(prompt: str, default_strategy: str) -> tuple[str, str]:
    """Parse /command from prompt, return (strategy, remaining_prompt)."""
    match = COMMAND_PATTERN.match(prompt.strip())
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return default_strategy, prompt


def _format_sectioning_result(topic: str, result: SynthesizedResult) -> str:
    """Format sectioning result."""
    lines = [
        "## Parallel Sectioning Complete",
        "",
        f"**Topic:** {topic}",
        f"**Sections processed:** {result.section_count} in parallel",
        "",
        "### Summary",
        result.summary,
        "",
        "### Key Points",
    ]
    for point in result.all_key_points:
        lines.append(f"- {point}")
    return "\n".join(lines)


def _format_voting_result(question: str, result: VotingOutcome) -> str:
    """Format voting result."""
    lines = [
        "## Parallel Voting Complete",
        "",
        f"**Question:** {question}",
        "",
        f"### Winner: {result.winning_answer}",
        f"*Votes: {result.vote_count}/{result.total_votes}*",
        "",
        "### All Votes",
    ]
    for i, answer in enumerate(result.all_answers, 1):
        marker = "→" if answer == result.winning_answer else " "
        lines.append(f"{marker} Agent {i}: {answer}")
    return "\n".join(lines)


def _format_mapreduce_result(result: ReducedSummary) -> str:
    """Format map-reduce result."""
    return "\n".join([
        "## Map-Reduce Complete",
        "",
        f"**Documents processed:** {result.total_documents} in parallel",
        f"**Total words analyzed:** {result.total_words}",
        "",
        "### Combined Summary",
        result.combined_summary,
    ])


@dataclasses.dataclass
class ParallelizationAgent:
    """Agent that wraps parallelization patterns with AG-UI state/activities."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None

    @property
    def default_strategy(self) -> str:
        """Get default strategy from extra_config."""
        return self.agent_config.extra_config.get("default_strategy", "sectioning")

    @property
    def num_voters(self) -> int:
        """Get num_voters from extra_config."""
        return self.agent_config.extra_config.get("num_voters", 3)

    async def _run_sectioning(
        self,
        prompt: str,
        state: ParallelState,
        agui_emitter: typing.Any,
        activity_id: str,
    ) -> tuple[str, ParallelState]:
        """Run sectioning strategy with state updates. Returns (result, state)."""
        if ":" in prompt:
            topic, sections_str = prompt.split(":", 1)
            sections = [s.strip() for s in sections_str.split(",")]
        else:
            topic = prompt
            sections = ["Overview", "Details", "Summary"]

        state.sections = sections
        state.worker_count = len(sections)

        if agui_emitter:
            agui_emitter.update_activity(
                "parallel_workers",
                {
                    "status": "spawning",
                    "strategy": "sectioning",
                    "workers": [
                        {"id": i, "name": s, "status": "pending"}
                        for i, s in enumerate(sections)
                    ],
                },
                activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        result = await run_sectioning(topic.strip(), sections)

        state.workers_complete = len(sections)
        if agui_emitter:
            agui_emitter.update_activity(
                "parallel_workers",
                {
                    "status": "complete",
                    "workers": [
                        {"id": i, "name": s, "status": "complete"}
                        for i, s in enumerate(sections)
                    ],
                },
                activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        return _format_sectioning_result(topic.strip(), result), state

    async def _run_voting(
        self,
        prompt: str,
        state: ParallelState,
        agui_emitter: typing.Any,
        activity_id: str,
    ) -> tuple[str, ParallelState]:
        """Run voting strategy with state updates. Returns (result, state)."""
        state.worker_count = self.num_voters
        state.votes = []

        if agui_emitter:
            agui_emitter.update_activity(
                "parallel_workers",
                {
                    "status": "voting",
                    "strategy": "voting",
                    "question": prompt[:50],
                    "workers": [
                        {"id": i, "name": f"Voter {i+1}", "status": "thinking"}
                        for i in range(self.num_voters)
                    ],
                },
                activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        result = await run_voting(prompt, self.num_voters)

        state.workers_complete = self.num_voters
        state.votes = [
            {"voter": i + 1, "answer": a}
            for i, a in enumerate(result.all_answers)
        ]

        if agui_emitter:
            agui_emitter.update_activity(
                "parallel_workers",
                {
                    "status": "complete",
                    "winner": result.winning_answer,
                    "vote_count": result.vote_count,
                    "workers": [
                        {
                            "id": i,
                            "name": f"Voter {i+1}",
                            "status": "voted",
                            "vote": a,
                        }
                        for i, a in enumerate(result.all_answers)
                    ],
                },
                activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        return _format_voting_result(prompt, result), state

    async def _run_mapreduce(
        self,
        prompt: str,
        state: ParallelState,
        agui_emitter: typing.Any,
        activity_id: str,
    ) -> tuple[str, ParallelState]:
        """Run map-reduce strategy with state updates. Returns (result, state)."""
        docs = []
        for part in prompt.split("|"):
            if ":" in part:
                doc_id, content = part.split(":", 1)
                docs.append((doc_id.strip(), content.strip()))
            else:
                docs.append((f"doc{len(docs)+1}", part.strip()))

        state.worker_count = len(docs)
        state.documents = [d[0] for d in docs]

        if agui_emitter:
            agui_emitter.update_activity(
                "parallel_workers",
                {
                    "status": "mapping",
                    "strategy": "mapreduce",
                    "phase": "map",
                    "workers": [
                        {"id": i, "name": d[0], "status": "summarizing"}
                        for i, d in enumerate(docs)
                    ],
                },
                activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        result = await run_map_reduce(docs)

        state.workers_complete = len(docs)

        if agui_emitter:
            agui_emitter.update_activity(
                "parallel_workers",
                {
                    "status": "complete",
                    "phase": "reduce",
                    "total_words": result.total_words,
                    "workers": [
                        {"id": i, "name": d[0], "status": "complete"}
                        for i, d in enumerate(docs)
                    ],
                },
                activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        return _format_mapreduce_result(result), state

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Stream events with rich AG-UI state and activities."""
        raw_prompt = _extract_prompt(message_history)
        strategy, prompt = _parse_command(raw_prompt, self.default_strategy)
        agui_emitter = getattr(deps, "agui_emitter", None) if deps else None

        # Initialize state
        state = ParallelState(
            strategy=strategy,
            stage="spawning",
            input_preview=prompt[:80] + "..." if len(prompt) > 80 else prompt,
        )

        activity_id = str(uuid.uuid4())

        if agui_emitter:
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        # Emit thinking
        strategy_desc = {
            "sectioning": "Dividing into parallel sections",
            "voting": f"Spawning {self.num_voters} parallel voters",
            "mapreduce": "Mapping documents to parallel workers",
        }.get(strategy, f"Running {strategy}")

        think_part = ai_messages.ThinkingPart(f"{strategy_desc}...")
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        # Execute selected strategy
        state.stage = "executing"
        if agui_emitter:
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        try:
            if strategy == "sectioning":
                result, state = await self._run_sectioning(
                    prompt, state, agui_emitter, activity_id
                )
            elif strategy == "voting":
                result, state = await self._run_voting(
                    prompt, state, agui_emitter, activity_id
                )
            elif strategy == "mapreduce":
                result, state = await self._run_mapreduce(
                    prompt, state, agui_emitter, activity_id
                )
            else:
                result = (
                    f"Unknown strategy: {strategy}\n\n"
                    f"**Available strategies:**\n"
                    f"- `/sectioning Topic: Section1, Section2`\n"
                    f"- `/voting Your question here?`\n"
                    f"- `/mapreduce doc1:text|doc2:text`"
                )
        except Exception as e:
            result = f"Parallelization failed: {e}"

        # Complete
        state.stage = "complete"
        if agui_emitter:
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        delta = f"\nCompleted with {state.workers_complete} parallel workers"
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0,
            delta=ai_messages.ThinkingPartDelta(content_delta=delta),
        )
        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Emit text response
        text_part = ai_messages.TextPart(result)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=result)


def create_parallelization_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> ParallelizationAgent:
    """Factory function to create the parallelization agent."""
    return ParallelizationAgent(
        agent_config, tool_configs, mcp_client_toolset_configs
    )
