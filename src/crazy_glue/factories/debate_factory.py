"""Factory for Multi-Agent Debate Room with AG-UI state and activities."""

from __future__ import annotations

import asyncio
import dataclasses
import typing
import uuid
from collections import abc

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


class JudgeScore(pydantic.BaseModel):
    """Judge's evaluation of an argument."""

    score: float = pydantic.Field(ge=1, le=10, description="Score from 1-10")
    reasoning: str = pydantic.Field(description="Brief explanation of score")


PRO_SYSTEM_PROMPT = """You are the PRO debater arguing IN FAVOR of the topic.
Keep arguments to 2-3 sentences. Be persuasive but concise."""

CON_SYSTEM_PROMPT = """You are the CON debater arguing AGAINST the topic.
Keep arguments to 2-3 sentences. Challenge the opponent's points."""

JUDGE_SYSTEM_PROMPT = """You are a debate judge. Score the argument from 1-10.
Give a brief 1-sentence explanation."""


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
class DebateAgent:
    """Agent that runs a multi-agent debate with AG-UI state/activities."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None
    _model = None

    @property
    def num_rounds(self) -> int:
        return self.agent_config.extra_config.get("num_rounds", 2)

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

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Stream the debate with AG-UI updates."""
        topic = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        # Create agents with retries for output validation
        model = self._get_model()
        pro_agent = Agent(model, system_prompt=PRO_SYSTEM_PROMPT, retries=3)
        con_agent = Agent(model, system_prompt=CON_SYSTEM_PROMPT, retries=3)
        judge_agent = Agent(model, output_type=JudgeScore, system_prompt=JUDGE_SYSTEM_PROMPT, retries=3)

        # Initial activity (skip state updates - delta computation has issues with new keys)
        if emitter:
            emitter.update_activity("debate", {"status": "starting", "topic": topic[:50]}, activity_id)

        # Start thinking
        think_part = ai_messages.ThinkingPart(f"Debate: {topic[:30]}...")
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        transcript = [f"# Debate: {topic}\n\n"]
        pro_total = 0.0
        con_total = 0.0

        for round_num in range(1, self.num_rounds + 1):
            # PRO turn
            if emitter:
                emitter.update_activity("debate", {
                    "status": "pro_speaking", "round": round_num,
                    "pro_score": pro_total, "con_score": con_total,
                }, activity_id)

            delta = f"\nRound {round_num}: PRO..."
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta))

            pro_prompt = f"Topic: {topic}\nArgue IN FAVOR in 2-3 sentences."
            pro_result = await pro_agent.run(pro_prompt)
            pro_arg = pro_result.output
            transcript.append(f"## Round {round_num}\n\n**PRO:** {pro_arg}\n\n")

            # CON turn
            if emitter:
                emitter.update_activity("debate", {
                    "status": "con_speaking", "round": round_num,
                    "pro_score": pro_total, "con_score": con_total,
                }, activity_id)

            delta = f"\nRound {round_num}: CON..."
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta))

            con_prompt = f"Topic: {topic}\nOpponent said: {pro_arg}\nArgue AGAINST in 2-3 sentences."
            con_result = await con_agent.run(con_prompt)
            con_arg = con_result.output
            transcript.append(f"**CON:** {con_arg}\n\n")

            # Judge
            if emitter:
                emitter.update_activity("debate", {
                    "status": "judging", "round": round_num,
                    "pro_score": pro_total, "con_score": con_total,
                }, activity_id)

            delta = f"\nJudging (parallel)..."
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta))

            # Score both arguments in parallel
            pro_score_task = judge_agent.run(f"Score this PRO argument: {pro_arg}")
            con_score_task = judge_agent.run(f"Score this CON argument: {con_arg}")
            pro_score_res, con_score_res = await asyncio.gather(pro_score_task, con_score_task)

            pro_score = pro_score_res.output.score
            con_score = con_score_res.output.score
            pro_total += pro_score
            con_total += con_score

            transcript.append(f"*Scores: PRO {pro_score}/10, CON {con_score}/10*\n\n")

        # Verdict
        winner = "PRO" if pro_total > con_total else ("CON" if con_total > pro_total else "TIE")

        if emitter:
            # Use activity for final quantitative data (NOT state snapshot)
            # State snapshots have timing issues - can be rejected if RUN_FINISHED
            # arrives at parser before STATE_SNAPSHOT due to multiplex race condition
            emitter.update_activity("debate", {
                "status": "complete",
                "winner": winner,
                "topic": topic[:50],
                "round": self.num_rounds,
                "total_rounds": self.num_rounds,
                "pro_score": pro_total,
                "con_score": con_total,
            }, activity_id)

        transcript.append(f"---\n\n**Winner: {winner}** (PRO: {pro_total:.1f}, CON: {con_total:.1f})")

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Final response
        response = "".join(transcript)
        text_part = ai_messages.TextPart(response)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response)


def create_debate_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> DebateAgent:
    """Factory function to create the debate agent."""
    return DebateAgent(agent_config, tool_configs, mcp_client_toolset_configs)
