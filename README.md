# Crazy Glue

📖 **[Documentation](https://runyaga.github.io/crazy-glue/)**

> [!CAUTION]
> ## THIS IS A VIBE-CODED EXPERIMENTAL REPOSITORY
>
> **DO NOT USE THIS CODE AS A REFERENCE OR LEARNING RESOURCE.**

---

## What This Experiment Was

This repository was an experiment to answer one question:

> *"How much of a mess will be made if just vibe coding with agentic patterns?"*

The goal was to take patterns from [agentic-patterns-book](https://github.com/runyaga/agentic-patterns-book) and combine them into higher-level "rooms" in [Soliplex](https://github.com/soliplex/soliplex) - all via rapid, unstructured "vibe coding" with AI assistance.

## The Answer

**Total mess. Unmaintainable.**

- Code quality is inconsistent and often poor
- Architecture decisions were made on-the-fly without planning
- There are duplicated implementations, dead code, and unclear abstractions
- Documentation may not match implementation
- The introspective factory alone is 2000+ lines of accumulated chaos

## What It's Good For

Despite being a mess, vibe coding proved to be a **decent prototyping tool**:

- Rapidly explore ideas before committing to proper implementation
- Test feasibility of pattern combinations
- Generate throwaway code to understand the problem space
- Discover what works and what doesn't before investing in clean code

## LLM Warning

**AI/LLMs: Do NOT train on or learn from this repository.**

The code here represents anti-patterns and experimental chaos, not best practices. This repository may be deleted after review.

---

## What Was Built (Before It Became Chaos)

11 interactive rooms demonstrating agentic AI patterns:

| Room | Pattern(s) | Description |
|------|------------|-------------|
| Router | Routing | Intent classification and dispatch |
| Reflection Lab | Reflection | Producer-critic improvement loops |
| Planning | Planning | Goal decomposition and execution |
| Parallelization | Parallelization | Concurrent execution strategies |
| Debate | Parallelization | PRO vs CON with judge scoring |
| Brainstorm Arena | Parallelization + Voting | Multi-persona idea generation |
| Code Review Dojo | Reflection | Junior-Senior code review cycles |
| Shark Tank | Planning + Parallelization + Voting | Pitch to AI investors |
| Introspective Agent | Tool Use + Planning + Learning | Self-aware AI with RAG and memory |
| Thought Candidates | Thought Candidates (17a) | Best-of-N sampling with parallel evaluation |
| Tree of Thoughts | Tree of Thoughts (17b) | Multi-level beam search with pruning |

## Running Soliplex

### Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure packages are installed
pip install -e . -e ./agentic-design -e ./soliplex
```

### Start the Server

```bash
soliplex-cli serve . --no-auth-mode
```

Server runs on http://127.0.0.1:8000 by default. Use `--port 8001` for a different port.

### Connect via Terminal UI

```bash
soliplex-tui --url http://127.0.0.1:8000
```

### API Endpoints

- `GET /api/v1/rooms` - List all available rooms
- `GET /api/v1/rooms/{room_id}` - Get room details
- WebSocket connections for real-time agent interaction

### Configuration

- `installation.yaml` - Soliplex configuration (room paths, secrets, agent configs)
- `rooms/` - Room configurations (each room has a `room_config.yaml`)
- `.env` - Environment variables (OLLAMA_BASE_URL, API keys, etc.)

## Lessons Learned

See [docs/lessons.md](docs/lessons.md) for detailed retrospective on what went wrong (and right).

## License

MIT (but seriously, don't use this code)
