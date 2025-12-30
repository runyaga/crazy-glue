"""Factory for Shark Tank - Planning + Parallelization + Voting patterns."""

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


class PitchBreakdown(pydantic.BaseModel):
    """Planning phase output - break pitch into analyzable components."""

    company_name: str = pydantic.Field(description="Suggested company name if not provided")
    one_liner: str = pydantic.Field(description="One sentence pitch summary")
    market_questions: list[str] = pydantic.Field(description="Key market questions to analyze")
    tech_questions: list[str] = pydantic.Field(description="Key technology questions to analyze")
    business_questions: list[str] = pydantic.Field(description="Key business model questions")
    risk_questions: list[str] = pydantic.Field(description="Key risks to evaluate")


class SharkAnalysis(pydantic.BaseModel):
    """Individual shark's analysis of the pitch."""

    strengths: list[str] = pydantic.Field(description="Key strengths identified")
    concerns: list[str] = pydantic.Field(description="Key concerns or weaknesses")
    questions: list[str] = pydantic.Field(description="Questions for the founder")
    excitement_level: int = pydantic.Field(ge=1, le=10, description="How excited 1-10")
    summary: str = pydantic.Field(description="2-3 sentence overall assessment")


class InvestmentDecision(pydantic.BaseModel):
    """Shark's final investment decision."""

    investing: bool = pydantic.Field(description="Whether to invest")
    amount: int = pydantic.Field(description="Investment amount in dollars (0 if not investing)")
    equity: float = pydantic.Field(description="Equity percentage requested (0 if not investing)")
    conditions: list[str] = pydantic.Field(description="Conditions or terms for the deal")
    reasoning: str = pydantic.Field(description="Why investing or passing")


# The Sharks - each with distinct investment philosophy
SHARKS = {
    "market": {
        "name": "Marina Market",
        "emoji": "🎯",
        "title": "Market Maven",
        "style": """You are Marina Market, a shark who obsesses over market size and timing.
You've made billions by spotting trends early. You care about TAM/SAM/SOM, competition,
and whether the market is ready. You're skeptical of 'creating new markets' but love
category leaders. You typically invest $100K-$500K for 10-20% equity.""",
    },
    "tech": {
        "name": "Trevor Tech",
        "emoji": "🔧",
        "title": "Tech Titan",
        "style": """You are Trevor Tech, a shark who built and sold 3 tech companies.
You evaluate technical feasibility, scalability, and defensibility. You love patents,
proprietary algorithms, and network effects. You're harsh on 'just an app' pitches
but generous with technical founders. You typically invest $200K-$1M for 15-25% equity.""",
    },
    "money": {
        "name": "Morgan Money",
        "emoji": "💰",
        "title": "Money Mogul",
        "style": """You are Morgan Money, a shark focused purely on unit economics.
You care about CAC, LTV, margins, and path to profitability. You've seen too many
startups burn cash chasing growth. You love recurring revenue and hate businesses
that need constant fundraising. You typically invest $150K-$750K for 12-22% equity.""",
    },
    "risk": {
        "name": "Rita Risk",
        "emoji": "⚠️",
        "title": "Risk Ranger",
        "style": """You are Rita Risk, a shark who made her fortune by avoiding disasters.
You identify regulatory risks, liability issues, competitive threats, and founder red flags.
You're the voice of caution but will invest big when risks are manageable. You often
co-invest with other sharks. You typically invest $100K-$400K for 10-18% equity.""",
    },
}

PLANNER_SYSTEM = """You are a startup pitch analyst. Break down the pitch into key areas
for evaluation. Extract or suggest a company name, create a one-liner, and identify
the critical questions that need answering in each domain."""

ANALYSIS_PROMPT = """As {name} ({title}), analyze this startup pitch:

**Company:** {company}
**Pitch:** {pitch}
**One-liner:** {one_liner}

Focus on your area of expertise. Be specific and insightful. Consider:
{questions}

Give your honest assessment."""

DECISION_PROMPT = """As {name} ({title}), make your final investment decision.

**Company:** {company}
**Pitch:** {pitch}

Your analysis found:
**Strengths:** {strengths}
**Concerns:** {concerns}

Other sharks are leaning: {other_leanings}

Decide: Are you IN or OUT? If in, how much and for what equity?
Be true to your investment style and typical deal ranges."""


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
class SharkTankAgent:
    """Agent that runs a Shark Tank pitch session."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None
    _model = None

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
        """Stream the Shark Tank session."""
        pitch = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        model = self._get_model()

        # Initial activity
        if emitter:
            emitter.update_activity("shark_tank", {
                "status": "starting",
                "phase": "intro",
                "pitch_preview": pitch[:80],
            }, activity_id)

        # Start thinking
        think_part = ai_messages.ThinkingPart(f"Shark Tank: {pitch[:40]}...")
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        transcript = ["# 🦈 Shark Tank\n\n"]
        transcript.append(f"**The Pitch:** {pitch}\n\n")

        # ========== PHASE 1: PLANNING ==========
        if emitter:
            emitter.update_activity("shark_tank", {
                "status": "planning",
                "phase": "planning",
                "pitch": pitch,
                "sharks": {k: {"name": v["name"], "emoji": v["emoji"], "title": v["title"], "status": "waiting"} for k, v in SHARKS.items()},
            }, activity_id)

        delta = "\nPhase 1: Analyzing pitch structure..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        planner = Agent(model, output_type=PitchBreakdown, system_prompt=PLANNER_SYSTEM, retries=3)
        plan_result = await planner.run(f"Analyze this startup pitch:\n\n{pitch}")
        breakdown = plan_result.output

        # Update with planning results
        if emitter:
            emitter.update_activity("shark_tank", {
                "status": "planned",
                "phase": "planning_complete",
                "pitch": pitch,
                "company": breakdown.company_name,
                "one_liner": breakdown.one_liner,
                "analysis_areas": {
                    "market": breakdown.market_questions,
                    "tech": breakdown.tech_questions,
                    "business": breakdown.business_questions,
                    "risk": breakdown.risk_questions,
                },
                "sharks": {k: {"name": v["name"], "emoji": v["emoji"], "title": v["title"], "status": "preparing"} for k, v in SHARKS.items()},
            }, activity_id)

        transcript.append(f"## 📋 {breakdown.company_name}\n\n")
        transcript.append(f"*{breakdown.one_liner}*\n\n")
        transcript.append("---\n\n")

        # ========== PHASE 2: PARALLEL SHARK ANALYSIS ==========
        if emitter:
            emitter.update_activity("shark_tank", {
                "status": "analyzing",
                "phase": "analysis",
                "pitch": pitch,
                "company": breakdown.company_name,
                "one_liner": breakdown.one_liner,
                "sharks": {k: {"name": v["name"], "emoji": v["emoji"], "title": v["title"], "status": "analyzing"} for k, v in SHARKS.items()},
                "analyses": {},
            }, activity_id)

        delta = "\nPhase 2: Sharks analyzing in parallel..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Create analysis agents for each shark
        question_map = {
            "market": breakdown.market_questions,
            "tech": breakdown.tech_questions,
            "money": breakdown.business_questions,
            "risk": breakdown.risk_questions,
        }

        async def analyze_pitch(shark_key: str) -> tuple[str, SharkAnalysis]:
            shark = SHARKS[shark_key]
            agent = Agent(
                model,
                output_type=SharkAnalysis,
                system_prompt=shark["style"],
                retries=3,
            )
            prompt = ANALYSIS_PROMPT.format(
                name=shark["name"],
                title=shark["title"],
                company=breakdown.company_name,
                pitch=pitch,
                one_liner=breakdown.one_liner,
                questions="\n".join(f"- {q}" for q in question_map[shark_key]),
            )
            result = await agent.run(prompt)
            return shark_key, result.output

        # Run all analyses in parallel
        analysis_tasks = [analyze_pitch(key) for key in SHARKS]
        analyses = dict(await asyncio.gather(*analysis_tasks))

        # Update activity with all analyses
        if emitter:
            sharks_status = {}
            analyses_data = {}
            for shark_key, analysis in analyses.items():
                shark = SHARKS[shark_key]
                sharks_status[shark_key] = {
                    "name": shark["name"],
                    "emoji": shark["emoji"],
                    "title": shark["title"],
                    "status": "analyzed",
                    "excitement": analysis.excitement_level,
                }
                analyses_data[shark_key] = {
                    "excitement_level": analysis.excitement_level,
                    "strengths": analysis.strengths,
                    "concerns": analysis.concerns,
                    "questions": analysis.questions,
                    "summary": analysis.summary,
                }
            emitter.update_activity("shark_tank", {
                "status": "analyzed",
                "phase": "analysis_complete",
                "pitch": pitch,
                "company": breakdown.company_name,
                "one_liner": breakdown.one_liner,
                "sharks": sharks_status,
                "analyses": analyses_data,
            }, activity_id)

        transcript.append("## 🔍 Shark Analysis\n\n")

        for shark_key, analysis in analyses.items():
            shark = SHARKS[shark_key]
            transcript.append(f"### {shark['emoji']} {shark['name']} ({shark['title']})\n\n")
            transcript.append(f"**Excitement Level:** {'🔥' * analysis.excitement_level}{'⚪' * (10 - analysis.excitement_level)} ({analysis.excitement_level}/10)\n\n")
            transcript.append("**Strengths:**\n")
            for s in analysis.strengths:
                transcript.append(f"- {s}\n")
            transcript.append("\n**Concerns:**\n")
            for c in analysis.concerns:
                transcript.append(f"- {c}\n")
            transcript.append(f"\n*\"{analysis.summary}\"*\n\n")

            delta = f"\n{shark['emoji']} {shark['name']}: {analysis.excitement_level}/10 excitement"
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(
                index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
            )

        # ========== PHASE 3: INVESTMENT DECISIONS (PARALLEL) ==========
        if emitter:
            sharks_deciding = {}
            for shark_key in SHARKS:
                shark = SHARKS[shark_key]
                analysis = analyses[shark_key]
                sharks_deciding[shark_key] = {
                    "name": shark["name"],
                    "emoji": shark["emoji"],
                    "title": shark["title"],
                    "status": "deciding",
                    "excitement": analysis.excitement_level,
                }
            emitter.update_activity("shark_tank", {
                "status": "deciding",
                "phase": "investment",
                "pitch": pitch,
                "company": breakdown.company_name,
                "one_liner": breakdown.one_liner,
                "sharks": sharks_deciding,
                "analyses": analyses_data,
                "deals": {},
                "total_investment": 0,
                "total_equity": 0,
            }, activity_id)

        delta = "\n\nPhase 3: Sharks making investment decisions..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
        )

        # Get leanings for social proof
        leanings = []
        for shark_key, analysis in analyses.items():
            shark = SHARKS[shark_key]
            lean = "interested" if analysis.excitement_level >= 6 else "skeptical"
            leanings.append(f"{shark['name']} is {lean}")

        async def make_decision(shark_key: str) -> tuple[str, InvestmentDecision]:
            shark = SHARKS[shark_key]
            analysis = analyses[shark_key]
            agent = Agent(
                model,
                output_type=InvestmentDecision,
                system_prompt=shark["style"],
                retries=3,
            )
            prompt = DECISION_PROMPT.format(
                name=shark["name"],
                title=shark["title"],
                company=breakdown.company_name,
                pitch=pitch,
                strengths="\n".join(f"- {s}" for s in analysis.strengths),
                concerns="\n".join(f"- {c}" for c in analysis.concerns),
                other_leanings=", ".join(l for l in leanings if shark["name"] not in l),
            )
            result = await agent.run(prompt)
            return shark_key, result.output

        # Run all decisions in parallel
        decision_tasks = [make_decision(key) for key in SHARKS]
        decisions = dict(await asyncio.gather(*decision_tasks))

        transcript.append("---\n\n## 💵 Investment Decisions\n\n")

        total_investment = 0
        total_equity = 0.0
        investors = []

        for shark_key, decision in decisions.items():
            shark = SHARKS[shark_key]
            if decision.investing:
                transcript.append(f"### {shark['emoji']} {shark['name']}: **I'M IN!**\n\n")
                transcript.append(f"💰 **${decision.amount:,}** for **{decision.equity}%** equity\n\n")
                if decision.conditions:
                    transcript.append("**Conditions:**\n")
                    for cond in decision.conditions:
                        transcript.append(f"- {cond}\n")
                transcript.append(f"\n*\"{decision.reasoning}\"*\n\n")
                total_investment += decision.amount
                total_equity += decision.equity
                investors.append(shark["name"])

                delta = f"\n{shark['emoji']} {shark['name']}: IN ${decision.amount:,} for {decision.equity}%"
            else:
                transcript.append(f"### {shark['emoji']} {shark['name']}: **I'm out.**\n\n")
                transcript.append(f"*\"{decision.reasoning}\"*\n\n")

                delta = f"\n{shark['emoji']} {shark['name']}: OUT"

            think_part.content += delta
            yield ai_messages.PartDeltaEvent(
                index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
            )

        # ========== FINAL VERDICT ==========
        transcript.append("---\n\n## 🏆 Final Verdict\n\n")

        if investors:
            transcript.append(f"**{len(investors)}/4 sharks are IN!**\n\n")
            transcript.append(f"**Combined Offer:** ${total_investment:,} for {total_equity:.1f}% equity\n\n")
            transcript.append(f"**Investors:** {', '.join(investors)}\n\n")
            verdict = "FUNDED"
            if total_investment >= 500000:
                transcript.append("🎉 *Congratulations! You've got a significant war chest!*\n")
            elif total_investment >= 200000:
                transcript.append("👍 *Solid start! You've got believers.*\n")
            else:
                transcript.append("🌱 *Seed funding secured. Time to prove yourself!*\n")
        else:
            transcript.append("**No deals today.**\n\n")
            transcript.append("*The sharks have spoken. But remember - many successful companies were rejected by investors at first!*\n")
            verdict = "NO DEAL"

        # Final activity update with all the bling
        if emitter:
            final_sharks = {}
            final_deals = {}
            for shark_key, decision in decisions.items():
                shark = SHARKS[shark_key]
                analysis = analyses[shark_key]
                final_sharks[shark_key] = {
                    "name": shark["name"],
                    "emoji": shark["emoji"],
                    "title": shark["title"],
                    "status": "in" if decision.investing else "out",
                    "excitement": analysis.excitement_level,
                }
                final_deals[shark_key] = {
                    "investing": decision.investing,
                    "amount": decision.amount,
                    "equity": decision.equity,
                    "conditions": decision.conditions,
                    "reasoning": decision.reasoning,
                }
            emitter.update_activity("shark_tank", {
                "status": "complete",
                "phase": "complete",
                "pitch": pitch,
                "company": breakdown.company_name,
                "one_liner": breakdown.one_liner,
                "verdict": verdict,
                "sharks": final_sharks,
                "analyses": analyses_data,
                "deals": final_deals,
                "investors": investors,
                "num_investors": len(investors),
                "total_investment": total_investment,
                "total_equity": total_equity,
                "valuation_implied": int(total_investment / (total_equity / 100)) if total_equity > 0 else 0,
            }, activity_id)

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Final response
        response = "".join(transcript)
        text_part = ai_messages.TextPart(response)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response)


def create_shark_tank_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> SharkTankAgent:
    """Factory function to create the shark tank agent."""
    return SharkTankAgent(agent_config, tool_configs, mcp_client_toolset_configs)
