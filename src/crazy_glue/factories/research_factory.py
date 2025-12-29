"""Factory for Collaborative Research agent using Exa.ai via MCP."""

from __future__ import annotations

import asyncio
import dataclasses
import re
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


# --- Pydantic Models ---

class ResearchPlanStep(pydantic.BaseModel):
    query: str = pydantic.Field(description="Search query for this step")
    description: str = pydantic.Field(description="What we're looking for")


class NumberedResearchPlanStep(pydantic.BaseModel):
    """Step with number for internal use after LLM parsing."""
    step_number: int
    query: str
    description: str


class ResearchPlan(pydantic.BaseModel):
    """Plan with search queries."""
    goal: str
    steps: list[ResearchPlanStep]
    reasoning: str


class ExaSearchResult(pydantic.BaseModel):
    """Single search result from Exa."""
    title: str
    url: str
    text: str | None = None
    published_date: str | None = None
    author: str | None = None


class ResearchSessionState(pydantic.BaseModel):
    """Persisted state of the research session."""
    phase: str = "initial"  # initial, review, executing, complete
    original_goal: str | None = None
    plan: list[NumberedResearchPlanStep] = pydantic.Field(default_factory=list)

    def to_json_block(self) -> str:
        return f"\n\n<!-- STATE_BLOCK: {self.model_dump_json()} -->"

    @classmethod
    def from_history(cls, history: MessageHistory | None) -> ResearchSessionState:
        if not history:
            return cls()
        for msg in reversed(history):
            if isinstance(msg, ai_messages.ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ai_messages.TextPart):
                        if "<!-- STATE_BLOCK:" in part.content:
                            try:
                                json_str = part.content.split("<!-- STATE_BLOCK:")[1].split("-->")[0].strip()
                                return cls.model_validate_json(json_str)
                            except Exception:
                                pass
        return cls()


# --- Helpers ---

def _extract_prompt(message_history: MessageHistory | None) -> str:
    if not message_history:
        return ""
    last_msg = message_history[-1]
    if isinstance(last_msg, ai_messages.ModelRequest):
        for part in last_msg.parts:
            if isinstance(part, ai_messages.UserPromptPart):
                return part.content
    return ""


def _is_approval(text: str) -> bool:
    text = text.lower().strip()
    return text in ["ok", "yes", "approve", "approved", "go", "proceed", "looks good", "start"]


def _parse_exa_mcp_results(mcp_result: typing.Any) -> list[ExaSearchResult]:
    """Parse MCP tool result into ExaSearchResult list."""
    results = []

    # Handle string format from Exa MCP (text-based output)
    if isinstance(mcp_result, str):
        text = mcp_result.strip()

        # Try to extract title, url, text from the string
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', text)
        url_match = re.search(r'URL:\s*(\S+)', text)
        text_match = re.search(r'Text:\s*(.+)', text, re.DOTALL)

        if title_match or url_match:
            results.append(ExaSearchResult(
                title=title_match.group(1).strip() if title_match else "",
                url=url_match.group(1).strip() if url_match else "",
                text=text_match.group(1).strip() if text_match else text,
            ))
        else:
            # If no structured format, use whole text as content
            results.append(ExaSearchResult(
                title="Search Result",
                url="",
                text=text,
            ))
        return results

    # Handle dict format
    if isinstance(mcp_result, dict):
        raw_results = mcp_result.get("results", [])
    elif isinstance(mcp_result, list):
        raw_results = mcp_result
    else:
        return results

    for r in raw_results:
        if isinstance(r, dict):
            results.append(ExaSearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                text=r.get("text"),
                published_date=r.get("publishedDate"),
                author=r.get("author"),
            ))

    return results


# --- The Agent ---

@dataclasses.dataclass
class ResearchAgent:
    """Collaborative Research Agent with Exa.ai via MCP."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None
    _model = None
    _exa_toolset = None

    @property
    def model_name(self) -> str:
        return self.agent_config.extra_config.get("model_name", "gpt-oss:20b")

    def _get_exa_toolset(self):
        """Create Exa MCP toolset from config."""
        if self._exa_toolset is None and self.mcp_client_toolset_configs:
            from soliplex import mcp_client

            exa_config = self.mcp_client_toolset_configs.get("exa")
            if exa_config:
                toolset_klass = mcp_client.TOOLSET_CLASS_BY_KIND[exa_config.kind]
                self._exa_toolset = toolset_klass(**exa_config.tool_kwargs)

        return self._exa_toolset

    def _get_model(self):
        if self._model is None:
            installation = self.agent_config._installation_config
            provider_base_url = installation.get_environment("OLLAMA_BASE_URL")
            provider = ollama_providers.OllamaProvider(
                base_url=f"{provider_base_url}/v1",
            )
            self._model = openai_models.OpenAIChatModel(
                model_name=self.model_name,
                provider=provider,
            )
        return self._model

    async def _create_plan(self, goal: str) -> ResearchPlan:
        """Use LLM to create a research plan with search queries."""
        model = self._get_model()
        planner = Agent(
            model,
            system_prompt=(
                "You are a research planner. Given a research goal, create a plan with "
                "3-4 specific web search queries that would help gather information. "
                "Each step should have a clear search query and description of what "
                "information we're looking for."
            ),
            output_type=ResearchPlan,
        )
        result = await planner.run(f"Create a research plan for: {goal}")
        return result.output

    async def _synthesize_report(self, goal: str, research_data: list[dict]) -> str:
        """Use LLM to synthesize research into a report."""
        model = self._get_model()
        synthesizer = Agent(
            model,
            system_prompt=(
                "You are a research report writer that ONLY uses information from the provided sources. "
                "CRITICAL RULES:\n"
                "1. ONLY include facts, claims, and data that appear in the provided search results\n"
                "2. NEVER add information from your own knowledge - if it's not in the sources, don't include it\n"
                "3. ALWAYS cite sources using [Source Title](URL) format for every claim\n"
                "4. If the sources don't contain enough information to answer a question, say so explicitly\n"
                "5. Quote relevant passages when possible to show the source of information\n\n"
                "Your report should synthesize the provided web search results into a coherent answer."
            ),
            output_type=str,
        )

        # Format research data with full text for better grounding
        formatted = []
        for item in research_data:
            formatted.append(f"### Search Query: {item['query']}\n")
            for r in item['results']:
                formatted.append(f"**Source: [{r.title}]({r.url})**")
                if r.text:
                    # Include more text for better grounding (up to 1500 chars)
                    formatted.append(f"Content:\n> {r.text[:1500]}")
                formatted.append("")
            formatted.append("---\n")

        prompt = f"Research Goal: {goal}\n\n## Sources from Web Search:\n\n" + "\n".join(formatted)
        prompt += "\n\nWrite a report answering the research goal using ONLY the above sources. Cite every claim."

        result = await synthesizer.run(prompt)
        return result.output

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:

        user_input = _extract_prompt(message_history)
        agui_emitter = getattr(deps, "agui_emitter", None) if deps else None
        session = ResearchSessionState.from_history(message_history)
        activity_id = str(uuid.uuid4())
        part_index = 0

        # Check for MCP toolset
        exa_toolset = self._get_exa_toolset()
        if not exa_toolset:
            error_msg = "Exa MCP toolset not configured. Add 'exa' to mcp_client_toolsets in room config."
            text_part = ai_messages.TextPart(error_msg)
            yield ai_messages.PartStartEvent(index=0, part=text_part)
            yield ai_messages.PartEndEvent(index=0, part=text_part)
            yield ai_run.AgentRunResultEvent(result=error_msg)
            return

        # CASE A: INITIAL - Create plan
        if session.phase == "initial" or session.phase == "complete":
            session.original_goal = user_input
            session.phase = "review"

            think_part = ai_messages.ThinkingPart(f"Planning research for: {user_input}")
            yield ai_messages.PartStartEvent(index=part_index, part=think_part)

            if agui_emitter:
                agui_emitter.update_activity("research", {"status": "planning", "goal": user_input[:50]}, activity_id)

            # Create plan using soliplex-configured model
            plan = await self._create_plan(user_input)

            session.plan = [
                NumberedResearchPlanStep(
                    step_number=i,
                    query=s.query,
                    description=s.description,
                )
                for i, s in enumerate(plan.steps, 1)
            ]

            yield ai_messages.PartDeltaEvent(
                index=part_index,
                delta=ai_messages.ThinkingPartDelta(content_delta=f"\nCreated {len(session.plan)} search queries"),
            )
            yield ai_messages.PartEndEvent(index=part_index, part=think_part)

            response_text = (
                f"## Research Plan for: {user_input}\n\n"
                + "\n".join([f"{s.step_number}. **{s.query}**\n   _{s.description}_" for s in session.plan])
                + "\n\n**Proceed with this plan?** (say 'yes' or provide feedback)"
                + session.to_json_block()
            )

            part_index += 1
            text_part = ai_messages.TextPart(response_text)
            yield ai_messages.PartStartEvent(index=part_index, part=text_part)
            yield ai_messages.PartEndEvent(index=part_index, part=text_part)
            yield ai_run.AgentRunResultEvent(result=response_text)
            return

        # CASE B: REVIEW - Approve or modify
        elif session.phase == "review":
            if _is_approval(user_input):
                session.phase = "executing"
                # Fall through to execution
            else:
                think_part = ai_messages.ThinkingPart("Revising plan based on feedback...")
                yield ai_messages.PartStartEvent(index=part_index, part=think_part)

                new_goal = f"{session.original_goal} (Note: {user_input})"
                plan = await self._create_plan(new_goal)
                session.plan = [
                    NumberedResearchPlanStep(step_number=i, query=s.query, description=s.description)
                    for i, s in enumerate(plan.steps, 1)
                ]

                yield ai_messages.PartEndEvent(index=part_index, part=think_part)

                response_text = (
                    f"## Revised Research Plan\n\n"
                    + "\n".join([f"{s.step_number}. **{s.query}**\n   _{s.description}_" for s in session.plan])
                    + "\n\n**Does this look better?**"
                    + session.to_json_block()
                )

                part_index += 1
                text_part = ai_messages.TextPart(response_text)
                yield ai_messages.PartStartEvent(index=part_index, part=text_part)
                yield ai_messages.PartEndEvent(index=part_index, part=text_part)
                yield ai_run.AgentRunResultEvent(result=response_text)
                return

        # CASE C: EXECUTION - Run searches and synthesize
        if session.phase == "executing":
            think_part = ai_messages.ThinkingPart(f"Executing {len(session.plan)} web searches via Exa MCP...")
            yield ai_messages.PartStartEvent(index=part_index, part=think_part)

            if agui_emitter:
                agui_emitter.update_activity(
                    "research",
                    {"status": "searching", "queries": len(session.plan)},
                    activity_id,
                )

            # Execute searches via MCP toolset
            research_data = []

            async def mcp_search(query: str) -> list[ExaSearchResult]:
                """Execute search via MCP toolset."""
                result = await exa_toolset.direct_call_tool(
                    "web_search_exa",
                    {"query": query, "numResults": 3},
                )
                return _parse_exa_mcp_results(result)

            # Use MCP connection for all searches
            async with exa_toolset:
                search_tasks = [mcp_search(step.query) for step in session.plan]
                results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

            for step, results in zip(session.plan, results_list):
                if isinstance(results, Exception):
                    yield ai_messages.PartDeltaEvent(
                        index=0,  # think_part is at index 0
                        delta=ai_messages.ThinkingPartDelta(content_delta=f"\n! Search failed for: {step.query} - {results}"),
                    )
                    research_data.append({"query": step.query, "results": []})
                else:
                    yield ai_messages.PartDeltaEvent(
                        index=0,  # think_part is at index 0
                        delta=ai_messages.ThinkingPartDelta(content_delta=f"\n+ Found {len(results)} results for: {step.query}"),
                    )
                    research_data.append({"query": step.query, "results": results})

                # Emit tool call events for AG-UI visibility
                part_index += 1
                tc_part = ai_messages.ToolCallPart(tool_name="web_search_exa", args={"query": step.query})
                yield ai_messages.PartStartEvent(index=part_index, part=tc_part)
                yield ai_messages.PartEndEvent(index=part_index, part=tc_part)

            if agui_emitter:
                agui_emitter.update_activity(
                    "research",
                    {"status": "synthesizing", "results_count": sum(len(d["results"]) for d in research_data)},
                    activity_id,
                )

            yield ai_messages.PartDeltaEvent(
                index=0,
                delta=ai_messages.ThinkingPartDelta(content_delta="\n\nSynthesizing research into report..."),
            )

            # Synthesize report using soliplex-configured model
            report = await self._synthesize_report(session.original_goal, research_data)

            yield ai_messages.PartEndEvent(index=0, part=think_part)

            session.phase = "complete"

            # Build sources section
            sources = []
            for data in research_data:
                for r in data["results"]:
                    sources.append(f"- [{r.title}]({r.url})")

            response_text = (
                f"# Research Report: {session.original_goal}\n\n"
                f"{report}\n\n"
                "---\n"
                "## Sources\n"
                + "\n".join(sources[:10])  # Top 10 sources
                + session.to_json_block()
            )

            if agui_emitter:
                agui_emitter.update_activity(
                    "research",
                    {"status": "complete", "sources": len(sources)},
                    activity_id,
                )

            part_index += 1
            text_part = ai_messages.TextPart(response_text)
            yield ai_messages.PartStartEvent(index=part_index, part=text_part)
            yield ai_messages.PartEndEvent(index=part_index, part=text_part)
            yield ai_run.AgentRunResultEvent(result=response_text)


def create_research_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> ResearchAgent:
    return ResearchAgent(agent_config, tool_configs, mcp_client_toolset_configs)
