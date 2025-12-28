"""Factory for Routing Pattern agent with AG-UI state and activities."""

from __future__ import annotations

import dataclasses
import typing
import uuid
from collections import abc

from pydantic_ai import messages as ai_messages
from pydantic_ai import run as ai_run
from pydantic_ai import tools as ai_tools

from agentic_patterns.routing import Intent
from agentic_patterns.routing import RouteDecision
from agentic_patterns.routing import RouteResponse
from agentic_patterns.routing import route_query

if typing.TYPE_CHECKING:
    from soliplex import config

MessageHistory = typing.Sequence[ai_messages.ModelMessage]
NativeEvent = ai_messages.AgentStreamEvent | ai_run.AgentRunResultEvent[typing.Any]


HANDLER_INFO = {
    Intent.ORDER_STATUS: {
        "name": "Order Status Specialist",
        "icon": "package",
    },
    Intent.PRODUCT_INFO: {
        "name": "Product Information Expert",
        "icon": "info",
    },
    Intent.TECHNICAL_SUPPORT: {
        "name": "Technical Support Engineer",
        "icon": "wrench",
    },
    Intent.CLARIFICATION: {
        "name": "Customer Service Assistant",
        "icon": "help-circle",
    },
}


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


def _format_response(decision: RouteDecision, response: RouteResponse) -> str:
    """Format the routing result as a readable response."""
    handler = HANDLER_INFO.get(decision.intent, {})
    handler_name = handler.get("name", decision.intent.value)

    lines = [
        f"**Routed to: {handler_name}**",
        f"*Intent: {decision.intent.value} ({decision.confidence:.0%} confidence)*",
        "",
        f"> {decision.reasoning}",
        "",
        "---",
        "",
    ]

    response_dict = response.model_dump()
    for field, value in response_dict.items():
        if isinstance(value, list) and value:
            lines.append(f"**{field.replace('_', ' ').title()}:**")
            for item in value:
                lines.append(f"- {item}")
        elif value:
            lines.append(f"**{field.replace('_', ' ').title()}:** {value}")

    return "\n".join(lines)


@dataclasses.dataclass
class RoutingAgentRun:
    """Represents a single run of the routing agent."""

    prompt: str
    _agent: RoutingAgent
    _decision: RouteDecision | None = None
    _response: RouteResponse | None = None

    def new_messages(self) -> list[ai_messages.ModelMessage]:
        """Return messages for the conversation history."""
        response_text = (
            _format_response(self._decision, self._response)
            if self._decision
            else "Unable to process query."
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
class RoutingAgent:
    """Agent that wraps the routing pattern with AG-UI state/activities."""

    agent_config: config.FactoryAgentConfig
    tool_configs: config.ToolConfigMap = None
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None

    output_type = None

    async def run(
        self,
        prompt: str,
        message_history: MessageHistory | None = None,
        deps: ai_tools.AgentDepsT = None,
    ) -> RoutingAgentRun:
        """Run the routing pattern."""
        run = RoutingAgentRun(prompt, self)
        run._decision, run._response = await route_query(prompt)
        return run

    async def run_stream_events(
        self,
        output_type: typing.Any = None,
        message_history: MessageHistory | None = None,
        deferred_tool_results: typing.Any = None,
        deps: ai_tools.AgentDepsT = None,
        **kwargs: typing.Any,
    ) -> abc.AsyncIterator[NativeEvent]:
        """Stream events with AG-UI state updates."""
        prompt = _extract_prompt(message_history)
        agui_emitter = getattr(deps, "agui_emitter", None) if deps else None
        activity_id = str(uuid.uuid4())

        # Stage 1: Classifying
        if agui_emitter:
            # Use activities only - state has delta and timing issues
            agui_emitter.update_activity(
                "routing",
                {
                    "status": "classifying",
                    "stage": "classifying",
                    "query_preview": prompt[:80] if prompt else "",
                    "message": "Analyzing intent...",
                },
                activity_id,
            )

        # Emit thinking
        think_part = ai_messages.ThinkingPart("Analyzing query intent...")
        yield ai_messages.PartStartEvent(index=0, part=think_part)

        # Execute routing
        decision, response = await route_query(prompt)
        handler_info = HANDLER_INFO.get(decision.intent, {})
        handler_name = handler_info.get("name", decision.intent.value)

        # Stage 2: Routing
        if agui_emitter:
            agui_emitter.update_activity(
                "routing",
                {
                    "status": "routing",
                    "stage": "routing",
                    "query_preview": prompt[:80] if prompt else "",
                    "detected_intent": decision.intent.value,
                    "confidence": decision.confidence,
                    "handler_name": handler_name,
                },
                activity_id,
            )

        delta = f"\nDetected: {decision.intent.value} ({decision.confidence:.0%})"
        delta += f"\nRouting to: {handler_name}"
        think_part.content += delta
        yield ai_messages.PartDeltaEvent(
            index=0,
            delta=ai_messages.ThinkingPartDelta(content_delta=delta),
        )

        # Stage 3: Complete
        if agui_emitter:
            agui_emitter.update_activity(
                "routing",
                {
                    "status": "complete",
                    "stage": "complete",
                    "query_preview": prompt[:80] if prompt else "",
                    "detected_intent": decision.intent.value,
                    "confidence": decision.confidence,
                    "handler_name": handler_name,
                },
                activity_id,
            )

        yield ai_messages.PartEndEvent(index=0, part=think_part)

        # Emit text response
        response_text = _format_response(decision, response)
        text_part = ai_messages.TextPart(response_text)
        yield ai_messages.PartStartEvent(index=1, part=text_part)
        yield ai_messages.PartEndEvent(index=1, part=text_part)

        yield ai_run.AgentRunResultEvent(result=response_text)


def create_routing_agent(
    agent_config: config.FactoryAgentConfig,
    tool_configs: config.ToolConfigMap = None,
    mcp_client_toolset_configs: config.MCP_ClientToolsetConfigMap = None,
) -> RoutingAgent:
    """Factory function to create the routing agent."""
    return RoutingAgent(agent_config, tool_configs, mcp_client_toolset_configs)
