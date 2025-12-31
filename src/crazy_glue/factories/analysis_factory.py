"""
Factory for Analysis Room - The System Architect.

Provides room introspection, creation, and tool management using
a simple command handler dispatch pattern.
"""

from __future__ import annotations

import dataclasses
import typing
from collections import abc

from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from soliplex import config

from crazy_glue.analysis import HANDLERS
from crazy_glue.analysis import AnalysisContext
from crazy_glue.analysis import parse_command
from crazy_glue.analysis.handlers import handle_unknown
from crazy_glue.analysis.planner import execute_plan
from crazy_glue.analysis.planner import is_complex_request
from crazy_glue.analysis.planner import plan_request

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]


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
class AnalysisAgent:
    """Agent that manages rooms and tools."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None

    def _build_context(self) -> AnalysisContext:
        """Build context for handlers."""
        return AnalysisContext(
            installation_config=self.agent_config._installation_config,
            agent_config=self.agent_config,
        )

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: typing.Any = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Stream events from command processing."""
        from crazy_glue.analysis import formatters
        from crazy_glue.analysis.planner import ExecutionPlan

        user_prompt = _extract_prompt(message_history)
        ctx = self._build_context()

        # Check for plan confirmation
        if user_prompt.lower().strip() in ("y", "yes", "proceed"):
            plan_data = ctx.load_pending_plan()
            if plan_data:
                plan = ExecutionPlan(**plan_data)
                async for event in execute_plan(ctx, plan):
                    yield event
                ctx.clear_pending_plan()
                yield ai_run.AgentRunResultEvent(result="")
                return

        # Check for complex request needing planning
        if is_complex_request(user_prompt):
            # Show planning indicator
            async for event in self._yield_text("Planning your request...\n\n"):
                yield event

            try:
                plan = await plan_request(ctx, user_prompt)

                # If clarifications needed, ask questions
                if plan.clarifications:
                    text = formatters.format_clarifications(plan)
                    async for event in self._yield_text(text):
                        yield event
                    yield ai_run.AgentRunResultEvent(result="")
                    return

                # Show plan and wait for confirmation
                text = formatters.format_plan(plan)
                async for event in self._yield_text(text):
                    yield event

                # Save plan for confirmation
                ctx.save_pending_plan(plan.model_dump())
                yield ai_run.AgentRunResultEvent(result="")
                return

            except Exception as e:
                # Fall back to simple parsing on error
                err = f"Planning failed: {e}\nFalling back to simple parsing.\n\n"
                async for event in self._yield_text(err):
                    yield event

        # Simple command - use keyword parser
        cmd = parse_command(user_prompt)
        handler = HANDLERS.get(cmd.command, handle_unknown)

        async for event in handler(ctx, cmd):
            yield event

        yield ai_run.AgentRunResultEvent(result="")

    async def _yield_text(self, text: str):
        """Helper to yield text as stream events."""
        text_part = ai_messages.TextPart(text)
        yield ai_messages.PartStartEvent(index=0, part=text_part)
        yield ai_messages.PartEndEvent(index=0, part=text_part)


def create_analysis_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> AnalysisAgent:
    """Factory function to create the analysis agent."""
    return AnalysisAgent(agent_config, tool_configs, mcp_client_toolset_configs)
