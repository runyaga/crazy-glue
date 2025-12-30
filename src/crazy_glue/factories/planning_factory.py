"""Factory for Planning Pattern agent with AG-UI state and activities."""

from __future__ import annotations

import dataclasses
import typing
import uuid
from collections import abc

import pydantic
from agentic_patterns.planning import PlanExecutionResult
from agentic_patterns.planning import create_plan
from agentic_patterns.planning import execute_plan
from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools
from soliplex import config

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]


# AG-UI State model for planning workflow
class PlanningState(pydantic.BaseModel):
    """Track the planning workflow state for AG-UI."""

    stage: str = "idle"  # idle, planning, executing, synthesizing, complete
    goal: str | None = None
    plan_steps: list[dict] = pydantic.Field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    replanned: bool = False
    execution_progress: float = 0.0


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


def _format_result(result: PlanExecutionResult) -> str:
    """Format the planning result as a readable response."""
    status_icon = "✅" if result.success else "⚠️"

    lines = [
        f"## {status_icon} Plan Execution {'Complete' if result.success else 'Partial'}",
        "",
        f"**Goal:** {result.goal}",
        "",
        f"**Progress:** {result.completed_steps}/{result.total_steps} steps completed",
    ]

    if result.replanned:
        lines.append("*Plan was adjusted during execution*")

    lines.extend(["", "### Step Results", ""])

    for step_result in result.step_results:
        icon = "✓" if step_result.success else "✗"
        output_preview = step_result.output[:100]
        if len(step_result.output) > 100:
            output_preview += "..."
        lines.append(f"{icon} **Step {step_result.step_number}:** {output_preview}")

    lines.extend([
        "",
        "---",
        "",
        "### Final Output",
        "",
        result.final_output,
    ])

    return "\n".join(lines)


@dataclasses.dataclass
class PlanningAgentRun:
    """Represents a single run of the planning agent."""

    prompt: str
    _agent: PlanningAgent
    _result: PlanExecutionResult | None = None

    def new_messages(self) -> list[ai_messages.ModelMessage]:
        """Return messages for the conversation history."""
        response_text = (
            _format_result(self._result) if self._result else "Planning failed."
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
class PlanningAgent:
    """Agent that wraps the planning pattern with AG-UI state/activities."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None

    @property
    def max_steps(self) -> int:
        """Get max_steps from extra_config."""
        return self.agent_config.extra_config.get("max_steps", 5)

    @property
    def allow_replan(self) -> bool:
        """Get allow_replan from extra_config."""
        return self.agent_config.extra_config.get("allow_replan", True)

    async def run(
        self,
        prompt: str,
        message_history: MessageHistory | None = None,
        deps: ai_tools.AgentDepsT = None,
    ) -> PlanningAgentRun:
        """Run the planning pattern."""
        run = PlanningAgentRun(prompt, self)
        plan = await create_plan(prompt, self.max_steps)
        run._result = await execute_plan(plan, self.allow_replan)
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
        state = PlanningState(
            stage="planning",
            goal=prompt[:100] + "..." if len(prompt) > 100 else prompt,
        )

        # Activity IDs
        planning_activity_id = str(uuid.uuid4())
        execution_activity_id = str(uuid.uuid4())
        synthesis_activity_id = str(uuid.uuid4())

        # --- Stage 1: Creating Plan ---
        if agui_emitter:
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking
            agui_emitter.update_activity(
                "planning",
                {
                    "status": "creating",
                    "message": "Decomposing goal into actionable steps...",
                    "goal": state.goal,
                    "max_steps": self.max_steps,
                },
                planning_activity_id,
            )

        # Emit thinking part
        think_part = ai_messages.ThinkingPart(
            f"Creating plan for: {prompt[:50]}..."
        )
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        # Create the plan
        plan = await create_plan(prompt, self.max_steps)

        # Update state with plan
        state.total_steps = len(plan.steps)
        state.plan_steps = [
            {
                "step_number": s.step_number,
                "description": s.description,
                "expected_output": s.expected_output,
                "status": "pending",
            }
            for s in plan.steps
        ]
        state.stage = "executing"

        if agui_emitter:
            agui_emitter.update_activity(
                "planning",
                {
                    "status": "complete",
                    "steps_created": len(plan.steps),
                    "reasoning": plan.reasoning,
                },
                planning_activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        # Update thinking with plan
        delta = f"\nPlan created with {len(plan.steps)} steps:\n"
        for step in plan.steps:
            delta += f"  {step.step_number}. {step.description[:40]}...\n"
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0,
            delta=ai_messages.ThinkingPartDelta(content_delta=delta),
        )

        # --- Stage 2: Executing Plan ---
        if agui_emitter:
            agui_emitter.update_activity(
                "execution",
                {
                    "status": "running",
                    "current_step": 0,
                    "total_steps": len(plan.steps),
                    "progress": 0.0,
                },
                execution_activity_id,
            )

        # Execute the plan (this runs all steps)
        result = await execute_plan(plan, self.allow_replan)

        # Update state with results
        state.completed_steps = result.completed_steps
        state.failed_steps = result.total_steps - result.completed_steps
        state.replanned = result.replanned
        state.execution_progress = 1.0
        state.stage = "synthesizing"

        # Update plan_steps with results
        for step_result in result.step_results:
            if step_result.step_number <= len(state.plan_steps):
                state.plan_steps[step_result.step_number - 1]["status"] = (
                    "complete" if step_result.success else "failed"
                )
                state.plan_steps[step_result.step_number - 1]["output"] = (
                    step_result.output[:100]
                )

        if agui_emitter:
            agui_emitter.update_activity(
                "execution",
                {
                    "status": "complete",
                    "completed_steps": result.completed_steps,
                    "failed_steps": state.failed_steps,
                    "replanned": result.replanned,
                },
                execution_activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        # Update thinking with execution results
        delta2 = f"\nExecuted {result.completed_steps}/{result.total_steps} steps"
        if result.replanned:
            delta2 += " (plan was adjusted)"
        think_part.content += delta2
        yield ai_messages.PartDeltaEvent(
            index=0,
            delta=ai_messages.ThinkingPartDelta(content_delta=delta2),
        )

        # --- Stage 3: Synthesis ---
        state.stage = "complete"
        if agui_emitter:
            agui_emitter.update_activity(
                "synthesis",
                {
                    "status": "complete",
                    "success": result.success,
                },
                synthesis_activity_id,
            )
            agui_emitter.update_state(state)
            state = state.model_copy(deep=True)  # Deep copy for delta tracking

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Emit text response
        response_text = _format_result(result)
        text_part = ai_messages.TextPart(response_text)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response_text)


def create_planning_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> PlanningAgent:
    """Factory function to create the planning agent."""
    return PlanningAgent(agent_config, tool_configs, mcp_client_toolset_configs)
