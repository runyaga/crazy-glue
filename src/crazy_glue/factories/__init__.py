"""Factory agents for agentic pattern rooms."""

from crazy_glue.factories.brainstorm_factory import create_brainstorm_agent
from crazy_glue.factories.code_review_factory import create_code_review_agent
from crazy_glue.factories.debate_factory import create_debate_agent
from crazy_glue.factories.introspective_factory import create_introspective_agent
from crazy_glue.factories.parallelization_factory import (
    create_parallelization_agent,
)
from crazy_glue.factories.planning_factory import create_planning_agent
from crazy_glue.factories.reflection_factory import create_reflection_agent
from crazy_glue.factories.research_factory import create_research_agent
from crazy_glue.factories.routing_factory import create_routing_agent
from crazy_glue.factories.shark_tank_factory import create_shark_tank_agent

__all__ = [
    "create_brainstorm_agent",
    "create_code_review_agent",
    "create_debate_agent",
    "create_introspective_agent",
    "create_parallelization_agent",
    "create_planning_agent",
    "create_reflection_agent",
    "create_research_agent",
    "create_routing_agent",
    "create_shark_tank_agent",
]
