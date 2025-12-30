"""Factory for Code Review Dojo - Reflection pattern with producer-critic loop."""

from __future__ import annotations

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


class CodeSubmission(pydantic.BaseModel):
    """Junior developer's code submission."""

    code: str = pydantic.Field(description="The code implementation")
    explanation: str = pydantic.Field(description="Brief explanation of approach")


class ReviewResult(pydantic.BaseModel):
    """Senior reviewer's feedback."""

    approved: bool = pydantic.Field(description="Whether the code is approved")
    issues: list[str] = pydantic.Field(description="List of issues found (empty if approved)")
    suggestions: list[str] = pydantic.Field(description="Specific improvement suggestions")
    praise: str = pydantic.Field(description="What was done well")


JUNIOR_SYSTEM = """You are a junior developer writing code. You are skilled but still learning.
Write clean, working code that solves the problem. Include comments where helpful.
When given feedback, incorporate it thoughtfully and explain your changes."""

JUNIOR_INITIAL = """Write code to solve this task:

{task}

Provide your implementation and briefly explain your approach."""

JUNIOR_REVISION = """Your previous code received this feedback:

**Issues:**
{issues}

**Suggestions:**
{suggestions}

**What was good:**
{praise}

Revise your code to address the feedback. Explain what you changed."""

SENIOR_SYSTEM = """You are a senior developer reviewing code. You are thorough but fair.
Look for: bugs, edge cases, security issues, readability, best practices, and performance.
Be specific with feedback. Acknowledge what's done well. Only approve code that's production-ready."""

SENIOR_REVIEW = """Review this code submission for the task: "{task}"

**Code:**
```
{code}
```

**Developer's explanation:**
{explanation}

Evaluate the code. Be specific about any issues and suggestions. Note what was done well."""


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
class CodeReviewAgent:
    """Agent that runs code review iterations."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None
    _model = None

    @property
    def max_rounds(self) -> int:
        return self.agent_config.extra_config.get("max_rounds", 3)

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
        """Stream the code review session with reflection loop."""
        task = _extract_prompt(message_history)
        emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        model = self._get_model()

        # Create agents with retries for output validation failures
        junior_agent = Agent(
            model,
            output_type=CodeSubmission,
            system_prompt=JUNIOR_SYSTEM,
            retries=3,
        )
        senior_agent = Agent(
            model,
            output_type=ReviewResult,
            system_prompt=SENIOR_SYSTEM,
            retries=3,
        )

        # Initial activity
        if emitter:
            emitter.update_activity("code_review", {
                "status": "starting",
                "task": task[:60],
                "round": 0,
                "max_rounds": self.max_rounds,
            }, activity_id)

        # Start thinking
        think_part = ai_messages.ThinkingPart(f"Code Review: {task[:40]}...")
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        transcript = [f"# Code Review Dojo\n\n**Task:** {task}\n\n"]

        current_submission: CodeSubmission | None = None
        approved = False

        for round_num in range(1, self.max_rounds + 1):
            # Junior writes/revises code
            if emitter:
                emitter.update_activity("code_review", {
                    "status": "coding",
                    "task": task,
                    "round": round_num,
                    "max_rounds": self.max_rounds,
                    "phase": "junior_coding",
                    "code": current_submission.code if current_submission else "",
                    "review": None,
                }, activity_id)

            delta = f"\n\nRound {round_num}: Junior coding..."
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(
                index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
            )

            if round_num == 1:
                # Initial submission
                prompt = JUNIOR_INITIAL.format(task=task)
            else:
                # Revision based on feedback
                prompt = JUNIOR_REVISION.format(
                    issues="\n".join(f"- {i}" for i in last_review.issues),
                    suggestions="\n".join(f"- {s}" for s in last_review.suggestions),
                    praise=last_review.praise,
                )

            junior_result = await junior_agent.run(prompt)
            current_submission = junior_result.output

            # Update activity with new code
            if emitter:
                emitter.update_activity("code_review", {
                    "status": "code_submitted",
                    "task": task,
                    "round": round_num,
                    "max_rounds": self.max_rounds,
                    "phase": "junior_submitted",
                    "code": current_submission.code,
                    "explanation": current_submission.explanation,
                    "review": None,
                }, activity_id)

            transcript.append(f"## Round {round_num}\n\n")
            transcript.append("### ✍️ Junior Developer\n\n")
            transcript.append(f"```\n{current_submission.code}\n```\n\n")
            transcript.append(f"*{current_submission.explanation}*\n\n")

            # Senior reviews
            if emitter:
                emitter.update_activity("code_review", {
                    "status": "reviewing",
                    "task": task,
                    "round": round_num,
                    "max_rounds": self.max_rounds,
                    "phase": "senior_review",
                    "code": current_submission.code,
                    "explanation": current_submission.explanation,
                    "review": None,
                }, activity_id)

            delta = f"\nRound {round_num}: Senior reviewing..."
            think_part.content += delta
            yield ai_messages.PartDeltaEvent(
                index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
            )

            review_prompt = SENIOR_REVIEW.format(
                task=task,
                code=current_submission.code,
                explanation=current_submission.explanation,
            )
            senior_result = await senior_agent.run(review_prompt)
            last_review = senior_result.output

            # Update activity with review results
            if emitter:
                emitter.update_activity("code_review", {
                    "status": "reviewed",
                    "task": task,
                    "round": round_num,
                    "max_rounds": self.max_rounds,
                    "phase": "review_complete",
                    "code": current_submission.code,
                    "explanation": current_submission.explanation,
                    "review": {
                        "approved": last_review.approved,
                        "issues": last_review.issues,
                        "suggestions": last_review.suggestions,
                        "praise": last_review.praise,
                    },
                }, activity_id)

            transcript.append("### 👀 Senior Reviewer\n\n")

            if last_review.approved:
                transcript.append("✅ **APPROVED**\n\n")
                transcript.append(f"*{last_review.praise}*\n\n")
                approved = True

                delta = f"\n✅ Approved in round {round_num}!"
                think_part.content += delta
                yield ai_messages.PartDeltaEvent(
                    index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
                )
                break
            else:
                transcript.append("🔄 **Needs Revision**\n\n")
                transcript.append("**Issues:**\n")
                for issue in last_review.issues:
                    transcript.append(f"- {issue}\n")
                transcript.append("\n**Suggestions:**\n")
                for suggestion in last_review.suggestions:
                    transcript.append(f"- {suggestion}\n")
                transcript.append(f"\n**What's good:** {last_review.praise}\n\n")

                delta = f"\n🔄 Round {round_num}: {len(last_review.issues)} issues"
                think_part.content += delta
                yield ai_messages.PartDeltaEvent(
                    index=0, delta=ai_messages.ThinkingPartDelta(content_delta=delta)
                )

        # Final summary
        transcript.append("---\n\n## Summary\n\n")
        if approved:
            transcript.append(f"✅ **Code approved after {round_num} round(s)**\n\n")
            transcript.append("### Final Code\n\n")
            transcript.append(f"```\n{current_submission.code}\n```\n")
        else:
            transcript.append(f"⚠️ **Max rounds ({self.max_rounds}) reached without approval**\n\n")
            transcript.append("### Latest Code\n\n")
            transcript.append(f"```\n{current_submission.code}\n```\n\n")
            transcript.append("### Remaining Issues\n\n")
            for issue in last_review.issues:
                transcript.append(f"- {issue}\n")

        # Final activity with complete code and review data
        if emitter:
            emitter.update_activity("code_review", {
                "status": "complete",
                "task": task,
                "round": round_num,
                "rounds_used": round_num,
                "max_rounds": self.max_rounds,
                "phase": "complete",
                "approved": approved,
                "code": current_submission.code if current_submission else "",
                "explanation": current_submission.explanation if current_submission else "",
                "review": {
                    "approved": last_review.approved,
                    "issues": last_review.issues if not approved else [],
                    "suggestions": last_review.suggestions if not approved else [],
                    "praise": last_review.praise,
                },
            }, activity_id)

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Final response
        response = "".join(transcript)
        text_part = ai_messages.TextPart(response)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response)


def create_code_review_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> CodeReviewAgent:
    """Factory function to create the code review agent."""
    return CodeReviewAgent(agent_config, tool_configs, mcp_client_toolset_configs)
