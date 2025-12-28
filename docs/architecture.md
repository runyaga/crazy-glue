# Architecture Overview

Crazy Glue connects three systems: **Soliplex** (room framework), **agentic-design** (pattern implementations), and **AG-UI** (protocol for UI state updates).

## System Architecture

```mermaid
flowchart TB
    subgraph Browser
        UI[Soliplex UI]
    end

    subgraph "Soliplex Server"
        CLI[soliplex-cli serve]
        Loader[Room Loader]
        Stream[Event Stream]
    end

    subgraph "Crazy Glue"
        RC[room_config.yaml]
        FA[Factory Agent]
        EM[AGUIEmitter]
    end

    subgraph "agentic-design"
        PAT[Patterns]
        MOD[Models]
    end

    subgraph LLM
        OL[Ollama]
    end

    UI <-->|WebSocket| Stream
    CLI --> Loader
    Loader --> RC
    RC --> FA
    FA --> PAT
    PAT --> MOD
    FA --> EM
    EM --> Stream
    PAT <-->|API| OL
```

## Factory Agent Flow

Factory agents implement a streaming interface that produces events compatible with pydantic-ai:

```mermaid
sequenceDiagram
    participant User
    participant Soliplex
    participant Factory
    participant Pattern
    participant LLM

    User->>Soliplex: Send message
    Soliplex->>Factory: run_stream_events()
    Factory->>Factory: Extract prompt from history

    Factory->>Pattern: Call pattern function
    Pattern->>LLM: Agent.run()
    LLM-->>Pattern: Response
    Pattern-->>Factory: Result

    Factory->>Soliplex: Yield PartStartEvent
    Factory->>Soliplex: Yield ThinkingPartDelta events
    Factory->>Soliplex: Yield PartEndEvent
    Factory->>Soliplex: Yield AgentRunResultEvent

    Soliplex->>User: Stream response
```

## AG-UI Integration

AG-UI protocol enables rich UI updates during agent execution:

```mermaid
flowchart LR
    subgraph "Factory Agent"
        Code[Agent Code]
        Emit[emitter.update_activity]
    end

    subgraph "AG-UI Events"
        ACT[ACTIVITY_UPDATE]
        STATE[STATE_SNAPSHOT]
        DELTA[STATE_DELTA]
    end

    subgraph "UI"
        Panel[Activity Panel]
        Viz[Visualizations]
    end

    Code --> Emit
    Emit --> ACT
    ACT --> Panel
    Panel --> Viz
```

!!! warning "State Snapshot Timing"
    Due to a race condition in `multiplex_streams`, STATE_SNAPSHOT events may be rejected if RUN_FINISHED arrives first. **Use activities exclusively** for factory agents.

## Multi-Agent Patterns

### Debate Pattern (Parallelization)

```mermaid
sequenceDiagram
    participant Factory
    participant PRO as PRO Agent
    participant CON as CON Agent
    participant Judge

    loop Each Round
        Factory->>PRO: Generate argument
        PRO-->>Factory: Pro argument

        Factory->>CON: Counter argument
        CON-->>Factory: Con argument

        par Score in parallel
            Factory->>Judge: Score PRO
            Factory->>Judge: Score CON
        end
        Judge-->>Factory: Scores
    end

    Factory->>Factory: Determine winner
```

### Shark Tank Pattern (Planning + Parallelization)

```mermaid
sequenceDiagram
    participant Factory
    participant Planner
    participant Shark1 as Marina Market
    participant Shark2 as Trevor Tech
    participant Shark3 as Morgan Money
    participant Shark4 as Rita Risk

    Factory->>Planner: Break down pitch
    Planner-->>Factory: Company analysis

    par Analyze in parallel
        Factory->>Shark1: Market analysis
        Factory->>Shark2: Tech analysis
        Factory->>Shark3: Financial analysis
        Factory->>Shark4: Risk analysis
    end

    Shark1-->>Factory: Market verdict
    Shark2-->>Factory: Tech verdict
    Shark3-->>Factory: Financial verdict
    Shark4-->>Factory: Risk verdict

    par Decide in parallel
        Factory->>Shark1: Investment decision
        Factory->>Shark2: Investment decision
        Factory->>Shark3: Investment decision
        Factory->>Shark4: Investment decision
    end

    Factory->>Factory: Aggregate results
```

### Code Review Pattern (Reflection)

```mermaid
flowchart TB
    subgraph "Reflection Loop"
        J[Junior Developer]
        S[Senior Reviewer]
        Check{Approved?}
    end

    Start([Task]) --> J
    J -->|Writes code| S
    S -->|Reviews| Check
    Check -->|No, max 3 rounds| J
    Check -->|Yes| Done([Final Code])
```

## Event Stream Architecture

```mermaid
flowchart LR
    subgraph "Factory Agent"
        Run[run_stream_events]
        Think[ThinkingPart]
        Text[TextPart]
    end

    subgraph "Event Types"
        PS[PartStartEvent]
        PD[PartDeltaEvent]
        PE[PartEndEvent]
        RR[AgentRunResultEvent]
    end

    subgraph "AGUIEmitter"
        Act[update_activity]
    end

    Run --> Think
    Run --> Text
    Think --> PS
    Think --> PD
    Think --> PE
    Text --> PS
    Text --> PE
    Run --> RR
    Run --> Act
```

## Data Flow

```mermaid
flowchart TB
    subgraph Input
        MSG[User Message]
        HIST[Message History]
    end

    subgraph Processing
        EXT[Extract Prompt]
        PAT[Run Pattern]
        FMT[Format Output]
    end

    subgraph Output
        EVT[Stream Events]
        ACT[Activity Updates]
        RES[Final Result]
    end

    MSG --> HIST
    HIST --> EXT
    EXT --> PAT
    PAT --> FMT
    FMT --> EVT
    PAT --> ACT
    FMT --> RES
```

## Configuration Hierarchy

```mermaid
flowchart TB
    subgraph Global
        INST[installation.yaml]
        ENV[.env]
    end

    subgraph Room
        RC[room_config.yaml]
        EC[extra_config]
    end

    subgraph Factory
        AC[agent_config]
        TC[tool_configs]
    end

    INST --> RC
    ENV --> INST
    RC --> AC
    EC --> AC
    AC --> TC
```
