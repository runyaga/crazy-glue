"""
Factory for Reflection Pattern agent with AG-UI state and activities.

Pattern: Reflection
Purpose: Generate content with producer-critic improvement loops

Flow diagram:

```mermaid
--8<-- [start:flow]
flowchart TB
    Input[User Request] --> Producer[Producer Agent]
    Producer --> Content[Generated Content]
    Content --> Critic[Critic Agent]
    Critic --> Eval{Quality OK?}
    Eval -->|No| Feedback[Feedback]
    Feedback --> Producer
    Eval -->|Yes| Output[Final Content]
--8<-- [end:flow]
```

Sequence diagram:

```mermaid
--8<-- [start:sequence]
sequenceDiagram
    participant User
    participant Producer
    participant Critic
    participant UI

    User->>Producer: Request
    Producer->>UI: activity: generating (round: 1)
    Producer->>Critic: Draft content
    Critic->>UI: activity: evaluating (round: 1)
    Critic->>Producer: Feedback (not approved)
    Producer->>UI: activity: generating (round: 2)
    Producer->>Critic: Revised content
    Critic->>UI: activity: evaluating (round: 2)
    Critic->>UI: activity: complete (approved: true)
--8<-- [end:sequence]
```
"""

from __future__ import annotations

import dataclasses
import typing
import uuid
from collections import abc

import pydantic
from agentic_patterns.reflection import ProducerOutput
from agentic_patterns.reflection import run_reflection
from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools
from soliplex import config

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]


# AG-UI State model for reflection workflow
class ReflectionState(pydantic.BaseModel):
    """Track the reflection workflow state for AG-UI."""

    stage: str = "idle"  # idle, producing, critiquing, refining, complete
    task_preview: str | None = None
    iteration: int = 0
    max_iterations: int = 3
    quality_threshold: float = 8.0
    current_score: float | None = None
    critique_history: list[dict] = pydantic.Field(default_factory=list)
    accepted: bool = False


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


def _format_result(result: ProducerOutput) -> str:
    """Format the reflection result as a readable response."""
    return "\n".join([
        "## Content Generated",
        "",
        result.content,
        "",
        "---",
        "",
        f"*{result.reasoning}*",
    ])


@dataclasses.dataclass
class ReflectionAgentRun:
    """Represents a single run of the reflection agent."""

    prompt: str
    _agent: ReflectionAgent
    _result: ProducerOutput | None = None

    def new_messages(self) -> list[ai_messages.ModelMessage]:
        """Return messages for the conversation history."""
        response_text = (
            _format_result(self._result)
            if self._result
            else "Reflection failed."
        )
        return [
            ai_messages.ModelRequest(
                parts=[ai_messages.UserPromptPart(content=self.prompt)]
            ),
            ai_messages.ModelResponse(
                parts=[ai_messages.TextPart(content=response_text)]
            ),
        ]


@dataclasses.dataclass
class ReflectionAgent:
    """Agent that wraps the reflection pattern with AG-UI state/activities."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None

    async def run(
        self,
        prompt: str,
        message_history: MessageHistory | None = None,
        deps: ai_tools.AgentDepsT = None,
    ) -> ReflectionAgentRun:
        """Run the reflection pattern."""
        run = ReflectionAgentRun(prompt, self)
        run._result = await run_reflection(prompt)
        return run

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Stream events with rich AG-UI state and activities."""
        prompt = _extract_prompt(message_history)
        agui_emitter = getattr(deps, "agui_emitter", None) if deps else None

        # Initialize state
        state = ReflectionState(
            stage="producing",
            task_preview=prompt[:80] + "..." if len(prompt) > 80 else prompt,
            iteration=1,
        )

        # Activity IDs
        producer_activity_id = str(uuid.uuid4())
        critic_activity_id = str(uuid.uuid4())

        # --- Stage 1: Initial Production ---
        if agui_emitter:
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking
            agui_emitter.update_activity(
                "producer",
                {
                    "status": "generating",
                    "iteration": 1,
                    "message": "Generating initial content...",
                },
                producer_activity_id,
            )

        # Emit thinking part
        think_part = ai_messages.ThinkingPart(
            "Starting reflection process...\n"
            "Producer generating initial content..."
        )
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        # Update to critiquing stage
        state.stage = "critiquing"
        if agui_emitter:
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking
            agui_emitter.update_activity(
                "critic",
                {
                    "status": "evaluating",
                    "iteration": 1,
                    "threshold": state.quality_threshold,
                    "message": "Critic evaluating quality...",
                },
                critic_activity_id,
            )

        delta = "\nCritic evaluating content quality..."
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0,
            delta=ai_messages.ThinkingPartDelta(content_delta=delta),
        )

        # Execute reflection (handles retries internally)
        try:
            result = await run_reflection(prompt)

            # Mark as accepted
            state.accepted = True
            state.stage = "complete"

            if agui_emitter:
                agui_emitter.update_activity(
                    "producer",
                    {
                        "status": "complete",
                        "iteration": state.iteration,
                        "accepted": True,
                    },
                    producer_activity_id,
                )
                agui_emitter.update_activity(
                    "critic",
                    {
                        "status": "approved",
                        "score": "8.0+",
                        "message": "Content meets quality threshold",
                    },
                    critic_activity_id,
                )
                agui_emitter.update_state(state)
                state = state.model_copy(deep=True)  # Deep copy for delta tracking

            delta2 = "\nContent approved by critic!"
            think_part.content += delta2
            yield ai_messages.PartDeltaEvent(
                index=0,
                delta=ai_messages.ThinkingPartDelta(content_delta=delta2),
            )

            response_text = _format_result(result)

        except Exception as e:
            # Reflection failed after max retries
            state.accepted = False
            state.stage = "complete"

            if agui_emitter:
                agui_emitter.update_activity(
                    "critic",
                    {
                        "status": "failed",
                        "message": f"Failed after {state.max_iterations} iterations",
                    },
                    critic_activity_id,
                )
                agui_emitter.update_state(state)
                state = state.model_copy(deep=True)  # Deep copy for delta tracking

            delta2 = f"\nReflection failed: {e}"
            think_part.content += delta2
            yield ai_messages.PartDeltaEvent(
                index=0,
                delta=ai_messages.ThinkingPartDelta(content_delta=delta2),
            )

            response_text = (
                f"## Reflection Failed\n\n"
                f"Could not meet quality threshold after "
                f"{state.max_iterations} iterations.\n\n"
                f"Error: {e}"
            )

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Emit text response
        text_part = ai_messages.TextPart(response_text)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response_text)


def create_reflection_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> ReflectionAgent:
    """Factory function to create the reflection agent."""
    return ReflectionAgent(agent_config, tool_configs, mcp_client_toolset_configs)
