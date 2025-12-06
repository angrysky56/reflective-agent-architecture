# RAA Cognitive Interface (Dashboard)

The RAA Dashboard is a **Streamlit-based MCP Host** that provides a visual interface for the Reflective Agent Architecture.

## Architecture

Unlike traditional dashboards that import backend logic directly, this dashboard operates as an **MCP Client**.

- **Frontend (`app.py`)**: A Streamlit application that acts as the "Host". It initializes an LLM (via `LLMFactory`) and connects to the RAA Server.
- **Backend (`src/server.py`)**: The RAA Core running as a subprocess. It exposes tools via the Model Context Protocol (MCP).
- **Communication**: The Dashboard spawns the Server using `uv run src/server.py` and communicates over Stdio.

## Prerequisites

- Python 3.10+
- `uv` package manager
- Project dependencies installed (`uv sync`)

## Configuration

The dashboard relies on the project's root `.env` file. Ensure the following are set:

```ini
# LLM Configuration
LLM_PROVIDER=openrouter
LLM_MODEL=deepcogito/cogito-v2-preview-llama-405b
OPENROUTER_API_KEY=sk-or-v1-...

# RAA Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=...

# MCP Configuration
The dashboard uses `src/dashboard/internal_bridge_config.json` to launch the RAA Server.
Default: `uv run -q src/server.py`
```

## Running the Dashboard

From the project root (`/reflective-agent-architecture`):

```bash
streamlit run src/dashboard/app.py
```

## Features

1.  **Chat Interface**: Interact with the RAA using a ReAct loop. The generic LLM (e.g., Llama 3) can call RAA tools like `mcp_check_cognitive_state` or `mcp_consult_compass`.
2.  **Control Plane**: Configure the LLM Provider and Model dynamically in the sidebar.
3.  **Visualizations**:
    - **Energy Landscape**: Real-time plot of System Entropy vs. Free Energy.
    - **Cognitive State**: Current operating mode (e.g., "Deep, "Flow", "Stuck").
    - **Thought Trace**: A log of recent internal thoughts and swarming activities.

## Troubleshooting

- **Connection Error**: If the dashboard fails to list tools, ensure `src/server.py` can run successfully:
  ```bash
  uv run src/server.py
  ```
  If this command fails in the terminal, the dashboard will not work.
- **Model Issues**: If the Chat fails, verify your `LLM_MODEL` string in `.env` or the Sidebar settings.
