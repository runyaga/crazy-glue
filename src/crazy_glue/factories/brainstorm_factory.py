"""
Factory for Brainstorm Arena - Parallelization pattern with voting.

Pattern: Parallelization + Voting
Purpose: Multi-persona idea generation with consensus voting

Flow diagram:

```mermaid
--8<-- [start:flow]
flowchart TB
    Topic[Brainstorm Topic] --> Personas

    subgraph Personas[4 Personas Generate Ideas]
        P1[Visionary]
        P2[Pragmatist]
        P3["Devil's Advocate"]
        P4[Innovator]
    end

    P1 --> Ideas
    P2 --> Ideas
    P3 --> Ideas
    P4 --> Ideas

    Ideas[All Ideas] --> Voting

    subgraph Voting[Parallel Voting]
        V1[Vote 1]
        V2[Vote 2]
        V3[Vote 3]
        V4[Vote 4]
    end

    Voting --> Winner[Best Idea]
--8<-- [end:flow]
```

Sequence diagram:

```mermaid
--8<-- [start:sequence]
sequenceDiagram
    participant Coord as Coordinator
    participant P1 as Visionary
    participant P2 as Pragmatist
    participant P3 as "Devil's Advocate"
    participant P4 as Innovator

    par Generate Ideas
        Coord->>P1: Generate idea
        Coord->>P2: Generate idea
        Coord->>P3: Generate idea
        Coord->>P4: Generate idea
    end

    P1-->>Coord: Idea 1
    P2-->>Coord: Idea 2
    P3-->>Coord: Idea 3
    P4-->>Coord: Idea 4

    par Vote on Ideas
        Coord->>P1: Vote
        Coord->>P2: Vote
        Coord->>P3: Vote
        Coord->>P4: Vote
    end

    Coord->>Coord: Tally votes
--8<-- [end:sequence]
```
"""

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
from soliplex import config

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]


class IdeaList(pydantic.BaseModel):
    """Structured output for idea generation."""

    ideas: list[str] = pydantic.Field(description="List of 3-4 creative ideas")


class VoteResult(pydantic.BaseModel):
    """Structured output for voting."""

    rankings: list[str] = pydantic.Field(description="Ideas ranked best to worst")
    reasoning: str = pydantic.Field(description="Brief explanation of ranking")


# Agent personas with distinct thinking styles
PERSONAS = {
    "practical": {
        "name": "Practical Pete",
        "emoji": "🎯",
        "prompt": """You are Practical Pete, a pragmatic thinker who focuses on
actionable, implementable ideas. You prefer solutions that are cost-effective,
realistic, and can be done with existing technology. Generate 3-4 practical ideas.""",
    },
    "creative": {
        "name": "Wild Card Wendy",
        "emoji": "🚀",
        "prompt": """You are Wild Card Wendy, a creative visionary who thinks outside
the box. You love unconventional, surprising, and even slightly absurd ideas that
challenge assumptions. Generate 3-4 wild and creative ideas.""",
    },
    "contrarian": {
        "name": "Contrarian Carl",
        "emoji": "🔄",
        "prompt": """You are Contrarian Carl, who questions the premise and looks at
problems from the opposite angle. You find value in what others dismiss and challenge
conventional wisdom. Generate 3-4 contrarian or reverse-thinking ideas.""",
    },
    "analytical": {
        "name": "Analytical Alex",
        "emoji": "📊",
        "prompt": """You are Analytical Alex, a data-driven thinker who optimizes for
efficiency and measurable outcomes. You consider second-order effects and systemic
solutions. Generate 3-4 analytical, optimization-focused ideas.""",
    },
}

VOTE_PROMPT = """You are {name}. Review these ideas from other brainstormers and rank
them from best to worst based on your perspective. Be true to your thinking style.

Ideas to rank:
{ideas}

Rank ALL ideas and briefly explain your reasoning."""


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
class BrainstormAgent:
    """Agent that runs parallel brainstorming with voting."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None
    _model = None

    @property
    def num_voters(self) -> int:
        return self.agent_config.extra_config.get("num_voters", 4)

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
        """Stream the brainstorm session with parallel generation and voting."""
        challenge = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        model = self._get_model()

        # Initial activity
        if emitter:
            emitter.update_activity("brainstorm", {
                "status": "starting",
                "challenge": challenge[:60],
                "phase": "ideation",
            }, activity_id)

        # Start thinking
        think_part = ai_messages.ThinkingPart(f"Brainstorming: {challenge[:40]}...")
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        transcript = [f"# Brainstorm: {challenge}\n\n"]

        # Phase 1: Parallel idea generation
        if emitter:
            emitter.update_activity("brainstorm", {
                "status": "generating",
                "phase": "ideation",
                "active_agents": list(PERSONAS.keys()),
            }, activity_id)

        delta = "\nPhase 1: Generating ideas in parallel..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Create agents for each persona with retries for output validation
        persona_agents = {}
        for key, persona in PERSONAS.items():
            persona_agents[key] = Agent(
                model,
                output_type=IdeaList,
                system_prompt=persona["prompt"],
                retries=3,
            )

        # Run all agents in parallel
        async def generate_ideas(key: str, agent: Agent) -> tuple[str, IdeaList]:
            prompt = f"Challenge: {challenge}\n\nGenerate your ideas:"
            result = await agent.run(prompt)
            return key, result.output

        tasks = [generate_ideas(k, a) for k, a in persona_agents.items()]
        results = await asyncio.gather(*tasks)

        # Collect all ideas
        all_ideas: dict[str, list[str]] = {}
        transcript.append("## Ideas Generated\n\n")

        for key, idea_list in results:
            persona = PERSONAS[key]
            all_ideas[key] = idea_list.ideas
            transcript.append(f"### {persona['emoji']} {persona['name']}\n\n")
            for idea in idea_list.ideas:
                transcript.append(f"- {idea}\n")
            transcript.append("\n")

            delta = f"\n{persona['emoji']} {persona['name']}: {len(idea_list.ideas)} ideas"
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(
                index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
            )

        # Phase 2: Voting
        if emitter:
            emitter.update_activity("brainstorm", {
                "status": "voting",
                "phase": "voting",
                "total_ideas": sum(len(ideas) for ideas in all_ideas.values()),
            }, activity_id)

        delta = "\n\nPhase 2: Voting on ideas..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Flatten ideas for voting (exclude own ideas from each voter)
        def get_votable_ideas(voter_key: str) -> str:
            lines = []
            for key, ideas in all_ideas.items():
                if key != voter_key:
                    persona = PERSONAS[key]
                    for idea in ideas:
                        lines.append(f"- [{persona['name']}] {idea}")
            return "\n".join(lines)

        # Create voting agents with retries
        vote_agents = {}
        for key, persona in PERSONAS.items():
            vote_agents[key] = Agent(
                model,
                output_type=VoteResult,
                system_prompt=f"You are {persona['name']}. {persona['prompt'].split('.')[1]}",
                retries=3,
            )

        # Run voting in parallel
        async def cast_vote(key: str, agent: Agent) -> tuple[str, VoteResult]:
            persona = PERSONAS[key]
            ideas_text = get_votable_ideas(key)
            prompt = VOTE_PROMPT.format(name=persona["name"], ideas=ideas_text)
            result = await agent.run(prompt)
            return key, result.output

        vote_tasks = [cast_vote(k, a) for k, a in vote_agents.items()]
        vote_results = await asyncio.gather(*vote_tasks)

        # Tally votes (simple scoring: 1st place = n points, 2nd = n-1, etc.)
        idea_scores: dict[str, float] = {}
        transcript.append("## Voting Results\n\n")

        for key, vote in vote_results:
            persona = PERSONAS[key]
            transcript.append(f"### {persona['emoji']} {persona['name']}'s Rankings\n\n")
            for i, idea in enumerate(vote.rankings[:5], 1):  # Top 5
                score = 6 - i  # 5 points for 1st, 4 for 2nd, etc.
                idea_scores[idea] = idea_scores.get(idea, 0) + score
                transcript.append(f"{i}. {idea}\n")
            transcript.append(f"\n*{vote.reasoning}*\n\n")

            delta = f"\n{persona['emoji']} voted"
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(
                index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
            )

        # Final ranking
        sorted_ideas = sorted(idea_scores.items(), key=lambda x: x[1], reverse=True)
        top_ideas = sorted_ideas[:5]

        transcript.append("## Final Rankings\n\n")
        for i, (idea, score) in enumerate(top_ideas, 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i - 1]
            transcript.append(f"{medal} **{idea}** ({score:.0f} points)\n\n")

        # Final activity update
        if emitter:
            emitter.update_activity("brainstorm", {
                "status": "complete",
                "phase": "complete",
                "challenge": challenge[:60],
                "total_ideas": sum(len(ideas) for ideas in all_ideas.values()),
                "top_idea": top_ideas[0][0] if top_ideas else "",
                "top_score": top_ideas[0][1] if top_ideas else 0,
            }, activity_id)

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Final response
        response = "".join(transcript)
        text_part = ai_messages.TextPart(response)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response)


def create_brainstorm_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> BrainstormAgent:
    """Factory function to create the brainstorm agent."""
    return BrainstormAgent(agent_config, tool_configs, mcp_client_toolset_configs)
