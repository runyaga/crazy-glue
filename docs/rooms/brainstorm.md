# Brainstorm Arena Room

**Pattern**: Parallelization + Voting
**Purpose**: Multi-persona idea generation with consensus voting

## How It Works

```mermaid
--8<-- "src/crazy_glue/factories/brainstorm_factory.py:flow"
```

## The Personas

| Persona | Focus | Approach |
|---------|-------|----------|
| **Visionary** | Big picture | Bold, transformative ideas |
| **Pragmatist** | Feasibility | Practical, implementable solutions |
| **Devil's Advocate** | Challenges | Contrarian, unconventional angles |
| **Innovator** | Novelty | Creative combinations, new approaches |

## Example Session

**Topic**: "How can we reduce food waste in restaurants?"

**Ideas Generated**:

1. **Visionary**: "AI-powered demand forecasting that predicts orders before customers arrive"
2. **Pragmatist**: "Partner with local farms for same-day composting pickup"
3. **Devil's Advocate**: "Charge customers for uneaten food to change behavior"
4. **Innovator**: "Gamify waste reduction with leaderboards between restaurant locations"

**Voting**:
- Visionary votes: Pragmatist (implementable impact)
- Pragmatist votes: Innovator (engaging + practical)
- Devil's Advocate votes: Visionary (biggest potential)
- Innovator votes: Pragmatist (immediate results)

**Winner**: Pragmatist's composting partnership (2 votes)

## Parallel Execution

```mermaid
--8<-- "src/crazy_glue/factories/brainstorm_factory.py:sequence"
```

## AG-UI Activities

```python
# Idea generation phase
emitter.update_activity("brainstorm", {
    "status": "generating",
    "topic": topic,
    "personas": [
        {"name": "Visionary", "status": "thinking"},
        {"name": "Pragmatist", "status": "thinking"},
        {"name": "Devil's Advocate", "status": "thinking"},
        {"name": "Innovator", "status": "thinking"},
    ],
}, activity_id)

# After ideas generated
emitter.update_activity("brainstorm", {
    "status": "voting",
    "ideas": [
        {"persona": "Visionary", "idea": "AI forecasting..."},
        {"persona": "Pragmatist", "idea": "Composting..."},
        ...
    ],
}, activity_id)

# Final result
emitter.update_activity("brainstorm", {
    "status": "complete",
    "winner": {
        "persona": "Pragmatist",
        "idea": "Partner with local farms...",
        "votes": 2,
    },
    "all_ideas": [...],
    "vote_breakdown": [...],
}, activity_id)
```

## Configuration

```yaml
id: "brainstorm"
name: "Brainstorm Arena"
description: "Multi-persona idea generation with voting"

agent:
  kind: "factory"
  factory_name: "crazy_glue.factories.brainstorm_factory.create_brainstorm_agent"
  with_agent_config: true
  extra_config:
    model_name: "gpt-oss:20b"

suggestions:
  - "How can we reduce food waste in restaurants?"
  - "Ways to make public transit more appealing"
  - "Ideas for engaging remote team building"
  - "Solutions for reducing plastic packaging"
```

## Factory Implementation

```python
@dataclasses.dataclass
class BrainstormAgent:
    PERSONAS = [
        ("Visionary", "Think big picture, transformative ideas"),
        ("Pragmatist", "Focus on practical, implementable solutions"),
        ("Devil's Advocate", "Challenge assumptions, unconventional angles"),
        ("Innovator", "Creative combinations, novel approaches"),
    ]

    async def run_stream_events(self, ...):
        topic = _extract_prompt(message_history)

        # Generate ideas in parallel
        idea_tasks = [
            self._generate_idea(persona, topic)
            for persona in self.PERSONAS
        ]
        ideas = await asyncio.gather(*idea_tasks)

        # Vote in parallel
        vote_tasks = [
            self._vote(persona, ideas)
            for persona in self.PERSONAS
        ]
        votes = await asyncio.gather(*vote_tasks)

        # Tally and determine winner
        winner = self._tally_votes(ideas, votes)
```

## Use Cases

- **Product brainstorming**: Generate diverse feature ideas
- **Problem solving**: Multiple perspectives on challenges
- **Creative writing**: Different story angles
- **Strategic planning**: Varied approaches to goals

## Related Patterns

- **Debate**: Two-sided instead of multi-perspective
- **Parallelization**: Base pattern for concurrent execution
- **Shark Tank**: Combines brainstorming with evaluation
