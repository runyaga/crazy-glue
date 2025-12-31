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
        user_prompt = _extract_prompt(message_history)
        ctx = self._build_context()

        # Parse command
        cmd = parse_command(user_prompt)

        # Get handler
        handler = HANDLERS.get(cmd.command, handle_unknown)

        # Execute and yield events
        async for event in handler(ctx, cmd):
            yield event

        # Final result event
        yield ai_run.AgentRunResultEvent(result="")


def create_analysis_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> AnalysisAgent:
    """Factory function to create the analysis agent."""
    return AnalysisAgent(agent_config, tool_configs, mcp_client_toolset_configs)
