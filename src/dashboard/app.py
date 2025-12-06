import asyncio
import json
import os
import re
import sys
import time

import graphviz
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dashboard.mcp_client_wrapper import get_client
from src.llm.factory import LLMFactory

# ==============================================================================
# Configuration & Setup
# ==============================================================================

st.set_page_config(
    page_title="RAA Cognitive Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of {"role": str, "type": str, "content": str, "meta": dict}

if "tools_cache" not in st.session_state:
    st.session_state.tools_cache = []

if "llm_config" not in st.session_state:
    st.session_state.llm_config = {
        "provider": os.getenv("DASHBOARD_LLM_PROVIDER", "openrouter"),
        "model": os.getenv("DASHBOARD_LLM_MODEL", "deepcogito/cogito-v2-preview-llama-405b"),
        "api_key": os.getenv("OPENROUTER_API_KEY", "")
    }

if "internals" not in st.session_state:
    st.session_state.internals = {
        "energy": 0.0,
        "entropy": 0.0,
        "phase": "Unknown",
        "load": "0%",
        "goals": 0,
        "active_goal_list": []
    }

# ==============================================================================
# Helper Functions
# ==============================================================================

def convert_mcp_to_openai_tools(mcp_tools: list) -> list:
    """Convert MCP tool definitions to OpenAI function format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {})
            }
        })
    return openai_tools

def add_message(role: str, content: str, msg_type: str = "text", meta: dict = None):
    """Add a message to the persistent session state."""
    st.session_state.messages.append({
        "role": role,
        "type": msg_type,
        "content": content,
        "meta": meta or {}
    })

def render_message(msg):
    """Render a single message based on its type."""
    role = msg["role"]
    content = msg["content"]
    m_type = msg.get("type", "text")
    meta = msg.get("meta", {})

    if role == "user":
        st.markdown(f'<div class="user-message">👤 <b>Signal</b><br>{content}</div>', unsafe_allow_html=True)

    elif role == "assistant":
        if m_type == "text":
            st.markdown(f'<div class="assistant-message">🤖 <b>Response</b><br>{content}</div>', unsafe_allow_html=True)

        elif m_type == "tool_call":
            tool_name = meta.get("tool")
            args = meta.get("args")
            with st.expander(f"🛠️ Thinking: {tool_name}", expanded=False):
                st.markdown(f"**Tool**: `{tool_name}`")
                st.markdown("**Arguments**:")
                st.json(args)

        elif m_type == "tool_result":
            tool_name = meta.get("tool")
            with st.container():
                st.markdown(f"✅ **Observation** (`{tool_name}`):")
                if len(content) < 500:
                   st.code(content, language="json" if content.strip().startswith("{") else None)
                else:
                    with st.expander("View Output"):
                        st.code(content)

async def fetch_metrics_async():
    """Async worker to fetch metrics from MCP."""
    client = get_client()
    try:
        # 1. Check Cognitive State
        state_raw = await client.call_tool("mcp_check_cognitive_state", {})
        try:
            cog_state = json.loads(state_raw)
        except json.JSONDecodeError:
            cog_state = {}

        # 2. Get Active Goals
        goals_raw = await client.call_tool("mcp_get_active_goals", {})
        try:
            goals_data = json.loads(goals_raw)
        except json.JSONDecodeError:
            goals_data = {}

        return cog_state, goals_data
    except Exception:
        return {}, {}

def poll_internals():
    """Poll RAA internals and update session state."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cog_state, goals_data = loop.run_until_complete(fetch_metrics_async())
        loop.close()

        if cog_state:
            st.session_state.internals["energy"] = cog_state.get("energy", 0.0)
            st.session_state.internals["phase"] = cog_state.get("state", "Unknown")
            # Heuristic map for entropy/load based on stability if available?
            st.session_state.internals["entropy"] = 0.5 if cog_state.get("stability") == "Unstable" else 0.1

        if goals_data:
            st.session_state.internals["goals"] = goals_data.get("count", 0)
            st.session_state.internals["active_goal_list"] = goals_data.get("goals", {})

    except Exception as e:
        print(f"Polling failed: {e}")

# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    st.title("Control Plane")
    st.markdown('<div class="status-badge status-online">System Online</div>', unsafe_allow_html=True)

    st.divider()

    with st.expander("LLM Configuration"):
        provider = st.selectbox("Provider", ["openrouter", "openai", "anthropic"],
                              index=["openrouter", "openai", "anthropic"].index(st.session_state.llm_config["provider"]) if st.session_state.llm_config["provider"] in ["openrouter", "openai", "anthropic"] else 0)
        model = st.text_input("Model", value=st.session_state.llm_config["model"])
        api_key = st.text_input("API Key", value=st.session_state.llm_config["api_key"], type="password")

        if st.button("Update Configuration"):
            st.session_state.llm_config = {"provider": provider, "model": model, "api_key": api_key}
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_MODEL"] = model
            if api_key: os.environ["OPENROUTER_API_KEY"] = api_key
            st.toast("Configuration updated!")
            st.rerun()

    st.subheader("Connection Status")
    status_box = st.empty()
    status_box.info("Connecting...")

    # Initialize Client & Fetch Tools
    client = get_client()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if not st.session_state.tools_cache:
            tools = loop.run_until_complete(client.list_tools())
            st.session_state.tools_cache = tools

        status_box.success(f"Connected: {len(st.session_state.tools_cache)} Tools Active")

        with st.expander("Tool Registry", expanded=False):
            for t in st.session_state.tools_cache:
                st.markdown(f"**{t['name']}**")
                st.caption(t['description'])

    except Exception as e:
        status_box.error(f"Connection Failed: {e}")
        st.error(e)

    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Refresh System State"):
        poll_internals()
        st.rerun()

# ==============================================================================
# Main Interface
# ==============================================================================

st.title("Reflective Agent Architecture")

# Heads Up Display (HUD)
hud_cols = st.columns([1, 1, 1, 1])
internals = st.session_state.internals

with hud_cols[0]:
    st.metric("Cognitive Load", internals.get("load", "N/A"))
with hud_cols[1]:
    st.metric("Entropy", f"{internals.get('entropy', 0.0):.2f}")
with hud_cols[2]:
    st.metric("Energy", f"{internals.get('energy', 0.0):.2f}")
with hud_cols[3]:
    st.metric("System Phase", internals.get("phase", "Unknown"))

st.divider()

# Initialize Tab State
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        f"You are the Reflective Agent. You connect to a backend via MCP.\n"
        f"PROTOCOL:\n"
        f"1. To use a tool, output valid JSON inside a code block:\n"
        f"```json\n"
        f"{{\"tool\": \"tool_name\", \"args\": {{...}}}}\n"
        f"```\n"
        f"2. You will receive the tool output.\n"
        f"3. Synthesize the final answer.\n"
    )

# Compact Tabs
tab_chat, tab_topo, tab_internal, tab_config = st.tabs([
    "💬 Signal",
    "🕸️ Topology",
    "⚙️ Internals",
    "🛠️ Config"
])

# ==============================================================================
# TAB 1: Cognitive Signal (Chat)
# ==============================================================================
with tab_chat:
    # Render History
    for msg in st.session_state.messages:
        render_message(msg)

    # Chat Input
    if prompt := st.chat_input("Inject cognitive signal..."):
        # 1. Add User Message
        add_message("user", prompt)
        st.rerun()

    # Execution Loop (Running the Agent)
    should_run = False
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        role = last_msg["role"]
        m_type = last_msg.get("type", "text")

        if role == "user":
            should_run = True
        elif role == "assistant" and m_type == "tool_result":
            should_run = True

    if should_run:
        with st.chat_message("assistant"):
            with st.spinner("Cognitive processing..."):
                # Update System Prompt with Tools
                tools_desc = json.dumps(st.session_state.tools_cache, indent=2)
                current_system_prompt = f"{st.session_state.system_prompt}\n\nTOOLS AVAILABLE:\n{tools_desc}"

                try:
                    llm = LLMFactory.create_provider(
                        st.session_state.llm_config["provider"],
                        st.session_state.llm_config["model"]
                    )

                    # Reconstruct history
                    history_str = ""
                    for m in st.session_state.messages:
                        content = m["content"]
                        if m["type"] == "tool_result":
                            content = f"Tool Output: {content}"
                        history_str += f"{m['role'].upper()}: {content}\n"

                    openai_tools = convert_mcp_to_openai_tools(st.session_state.tools_cache)

                    raw_response = llm.generate(
                        system_prompt=current_system_prompt,
                        user_prompt=history_str,
                        tools=openai_tools
                    )

                    # Check for Tool Call
                    tool_match = re.search(r"```json\s*({.*?})\s*```", raw_response, re.DOTALL)

                    if tool_match:
                        try:
                            # 1. Extract Tool Details
                            tool_call = json.loads(tool_match.group(1))
                            t_name = tool_call.get("tool")
                            t_args = tool_call.get("args", {})

                            # 2. Extract Pre-computation Thought (if any)
                            # The tool block matches the regex. Everything before it is thought.
                            thought_content = raw_response[:tool_match.start()].strip()

                            if thought_content:
                                add_message("assistant", thought_content)

                            # 3. Add Tool Call Message
                            add_message("assistant", "Executing Tool...", "tool_call", {"tool": t_name, "args": t_args})
                            st.rerun()

                        except json.JSONDecodeError:
                            add_message("assistant", raw_response)
                            st.rerun()
                    else:
                        add_message("assistant", raw_response)
                        st.rerun()

                except Exception as e:
                    st.error(f"Cognitive Error: {e}")

    # Post-Execution Tool Logic
    if st.session_state.messages and st.session_state.messages[-1]["type"] == "tool_call":
        last_msg = st.session_state.messages[-1]
        t_name = last_msg["meta"]["tool"]
        t_args = last_msg["meta"]["args"]

        with st.spinner(f"Executing {t_name}..."):
            try:
                client = get_client()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(client.call_tool(t_name, t_args))
                add_message("assistant", result, "tool_result", {"tool": t_name})
                st.rerun()
            except Exception as e:
                add_message("assistant", f"Error: {str(e)}", "tool_result", {"tool": t_name})
                st.rerun()

# ==============================================================================
# TAB 2: Neuro-Topology
# ==============================================================================
with tab_topo:
    st.subheader("Manifold Visualization")
    if st.button("Refresh Topology"):
        st.info("Fetching graph data... (Placeholder)")
        # In real impl, calling inspect_graph via hidden tool call

    graph = graphviz.Digraph()
    graph.edge('State', 'Agent')
    graph.edge('Agent', 'Action')
    graph.edge('Action', 'State')
    st.graphviz_chart(graph)

# ==============================================================================
# TAB 3: Director Internals
# ==============================================================================
with tab_internal:
    st.subheader("Metabolic Ledger")
    col1, col2, col3 = st.columns(3)
    internals = st.session_state.internals

    col1.metric("Energy", f"{internals.get('energy', 0.0):.2f}")
    col2.metric("Entropy", f"{internals.get('entropy', 0.0):.2f}")
    col3.metric("State", internals.get("phase", "Unknown"))

    st.markdown("#### Thermodynamic Dynamics")
    # Placeholder Data (Real data requires history tracking)
    import numpy as np
    import pandas as pd
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Energy', 'Entropy', 'Complexity'])
    st.line_chart(chart_data)

    st.divider()
    st.subheader("Active Goals (Pointer)")

    active_goals = internals.get("active_goal_list", {})
    if active_goals:
        for gid, gdata in active_goals.items():
             with st.container():
                st.markdown(f"**🎯 {gid}**")
                st.caption(gdata.get('description', 'No description'))
                st.progress(gdata.get('utility', 0.5)) # fallback utility
    else:
        st.info("No active goals found.")

# ==============================================================================
# TAB 4: System Config
# ==============================================================================
with tab_config:
    st.subheader("System Instructions")
    new_prompt = st.text_area("System Prompt",
                             value=st.session_state.system_prompt,
                             height=400)

    if st.button("Save System Prompt"):
        st.session_state.system_prompt = new_prompt
        st.toast("System Prompt Updated")

