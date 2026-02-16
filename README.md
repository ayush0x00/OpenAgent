# OpenAgent

Master–agent orchestration: a central master coordinates **Query agents** (send questions) and **Action agents** (expose tools). The master uses **GPT-4o-mini** to decide whether to answer directly or call a tool on an action agent.

## Setup

```bash
source ~/base/bin/activate  # or your venv
pip install -r requirements.txt
```

Create a `.env` in the project root (ignored by git) with:
```
OPENAI_API_KEY=sk-...
```

## Run

**Terminal 1 – Master (WebSocket on 8000)**

```bash
uvicorn master.app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 – Action agent (e.g. weather tool)**

```bash
python agents/action_weather.py
```

**Terminal 3 – Query agent (sends queries)**

```bash
python agents/query_demo.py
```

The query agent will send a few questions; the orchestrator will call the weather tool when relevant and return answers.

## Project layout

- `protocol/` – Wire protocol (Pydantic models: register, query, tool_call, tool_result, query_result, etc.)
- `openagent/` – Reusable agent client (WebSocket connect, register, query or handle tool_call)
- `master/` – FastAPI app, WebSocket endpoint, AgentRegistry, Orchestrator (GPT-4o-mini)
- `agents/` – Example query and action (weather) agents

## Message contract

- **Agent → Master**: `register`, `query`, `tool_result`, `ping`
- **Master → Agent**: `registered`, `query_result`, `tool_call`, `pong`, `error`

All JSON with `type`, `id`; see `protocol/messages.py` for full field spec.
