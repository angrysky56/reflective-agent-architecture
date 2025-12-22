import asyncio
import json
import os
import re
import sys
from datetime import datetime

import graphviz
import pandas as pd

# import plotly.io as pio
from dotenv import load_dotenv

load_dotenv()


import streamlit as st  # trunk-ignore(ruff/E402)  # noqa: E402

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dashboard.mcp_client_wrapper import get_client  # trunk-ignore(ruff/E402)  # noqa: E402
from src.llm.factory import LLMFactory  # trunk-ignore(ruff/E402)  # noqa: E402

# pio.renderers.default = "browser"

# ==============================================================================
# Constants
# ==============================================================================

RAA_CORE_TOOLS = {
    "deconstruct",
    "hypothesize",
    "synthesize",
    "constrain",
    "revise",
    "set_goal",
    "set_intentionality",
    "check_cognitive_state",
    "diagnose_pointer",
    "explore_for_utility",
    "inspect_graph",
    "inspect_knowledge_graph",
    "recall_work",
    "run_sleep_cycle",
    "consult_compass",
    "consult_curiosity",
    "evolve_formula",
    "resolve_meta_paradox",
    "compress_to_tool",
    "teach_cognitive_state",
    "get_active_goals",
    "propose_goal",
    "diagnose_antifragility",
    "get_known_archetypes",
    "visualize_thought",
    "compute_grok_depth",
    "orthogonal_dimensions_analyzer",
    "create_advisor",
    "delete_advisor",
    "list_advisors",
}

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
    st.session_state.messages = (
        []
    )  # List of {"role": str, "type": str, "content": str, "meta": dict}

if "tools_cache" not in st.session_state:
    st.session_state.tools_cache = []

if "llm_config" not in st.session_state:
    st.session_state.llm_config = {
        "provider": os.getenv("DASHBOARD_LLM_PROVIDER", "openrouter"),
        "model": os.getenv("DASHBOARD_LLM_MODEL", "deepcogito/cogito-v2-preview-llama-405b"),
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    }

if "internals" not in st.session_state:
    st.session_state.internals = {
        "energy": 0.0,
        "entropy": 0.0,
        "phase": "Unknown",
        "load": "0%",
        "goals": 0,
        "active_goal_list": [],
    }

if "energy_history" not in st.session_state:
    st.session_state.energy_history = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "metrics_polled" not in st.session_state:
    st.session_state.metrics_polled = False

if "agent_prompt" not in st.session_state:
    # Load RAA_AGENT.md as agent prompt
    agent_md_path = os.path.join(os.path.dirname(__file__), "..", "..", "RAA_AGENT.md")
    if os.path.exists(agent_md_path):
        with open(agent_md_path, "r") as f:
            st.session_state.agent_prompt = f.read()
    else:
        st.session_state.agent_prompt = "# RAA Agent\nNo agent prompt file found."

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if "compass_status" not in st.session_state:
    st.session_state.compass_status = {"active": False, "stage": None, "task": None}

if "active_agents" not in st.session_state:
    st.session_state.active_agents = []

# Chat History Directory
CHAT_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def save_conversation(name: str | None = None) -> None:
    """Save current conversation to JSON file."""
    name = name or st.session_state.current_conversation
    path = os.path.join(CHAT_HISTORY_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "name": name,
                "messages": st.session_state.messages,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )


def load_conversation(name: str) -> bool:
    """Load conversation from JSON file."""
    path = os.path.join(CHAT_HISTORY_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            st.session_state.messages = data.get("messages", [])
            st.session_state.current_conversation = name
            return True
    return False


def list_conversations() -> list:
    """List all saved conversations."""
    if not os.path.exists(CHAT_HISTORY_DIR):
        return []
    convos = []
    for f in os.listdir(CHAT_HISTORY_DIR):
        if f.endswith(".json"):
            path = os.path.join(CHAT_HISTORY_DIR, f)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
                    convos.append(
                        {
                            "name": f.replace(".json", ""),
                            "timestamp": data.get("timestamp", ""),
                            "messages": len(data.get("messages", [])),
                        }
                    )
            except (OSError, json.JSONDecodeError):
                # Skip invalid or inaccessible conversation files
                pass
    return sorted(convos, key=lambda x: x.get("timestamp", ""), reverse=True)


# ==============================================================================
# Helper Functions
# ==============================================================================


def convert_mcp_to_openai_tools(mcp_tools: list) -> list:
    """Convert MCP tool definitions to OpenAI function format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}),
                },
            }
        )
    return openai_tools


def add_message(role: str, content: str, msg_type: str = "text", meta: dict | None = None) -> None:
    """Add a message to the persistent session state."""
    st.session_state.messages.append(
        {"role": role, "type": msg_type, "content": content, "meta": meta or {}}
    )


def render_message(msg: dict) -> None:
    """Render a single message based on its type."""
    role = msg["role"]
    content = msg["content"]
    m_type = msg.get("type", "text")
    meta = msg.get("meta", {})

    if role == "user":
        st.markdown(
            f'<div class="user-message">👤 <b>Signal</b><br>{content}</div>', unsafe_allow_html=True
        )
        # Render attached file if present
        if meta.get("file_name"):
            st.caption(f"📎 Attached: {meta['file_name']}")

    elif role == "assistant":
        if m_type == "text":
            st.markdown(
                f'<div class="assistant-message">🤖 <b>Response</b><br>{content}</div>',
                unsafe_allow_html=True,
            )

        elif m_type == "image":
            # Render image output
            st.image(content, caption=meta.get("caption", "Generated Image"))

        elif m_type == "tool_call":
            tool_name = meta.get("tool")
            args = meta.get("args")
            # Compact tool call notification (not in expander)
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-left: 3px solid #e94560; padding: 10px; border-radius: 5px; margin: 5px 0;">🛠️ <b>Calling Tool:</b> <code>{tool_name}</code></div>',
                unsafe_allow_html=True,
            )
            with st.expander("View Arguments", expanded=False):
                st.json(args)

        elif m_type == "tool_result":
            tool_name = meta.get("tool")
            # Clear tool result display
            st.markdown(
                f'<div style="background: linear-gradient(135deg, #0f3443 0%, #34e89e20 100%); border-left: 3px solid #34e89e; padding: 10px; border-radius: 5px; margin: 5px 0;">✅ <b>Result from:</b> <code>{tool_name}</code></div>',
                unsafe_allow_html=True,
            )
            if len(content) < 800:
                st.code(content, language="json" if content.strip().startswith("{") else None)
            else:
                with st.expander("View Full Output", expanded=False):
                    st.code(content)


async def fetch_metrics_async() -> tuple[dict, dict]:
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


def poll_internals() -> None:
    """Poll RAA internals and update session state."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cog_state, goals_data = loop.run_until_complete(fetch_metrics_async())
        loop.close()

        if cog_state:
            energy = cog_state.get("energy", 0.0)
            entropy = 0.5 if cog_state.get("stability") == "Unstable" else 0.1

            st.session_state.internals["energy"] = energy
            st.session_state.internals["phase"] = cog_state.get("state", "Unknown")
            st.session_state.internals["entropy"] = entropy

            # Track history for charts (keep last 50 points)
            st.session_state.energy_history.append(
                {"time": datetime.now().strftime("%H:%M:%S"), "energy": energy, "entropy": entropy}
            )
            if len(st.session_state.energy_history) > 50:
                st.session_state.energy_history = st.session_state.energy_history[-50:]

        if goals_data:
            st.session_state.internals["goals"] = goals_data.get("count", 0)
            st.session_state.internals["active_goal_list"] = goals_data.get("goals", {})

        st.session_state.metrics_polled = True

    except Exception as e:
        print(f"Polling failed: {e}")


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    st.title("⚡ Control Plane")
    st.markdown(
        '<div class="status-badge status-online">System Online</div>', unsafe_allow_html=True
    )

    st.divider()

    # Auto-poll metrics on first load
    if not st.session_state.metrics_polled:
        poll_internals()

    # Chat History Section
    st.subheader("📜 Conversations")

    # New conversation button
    if st.button("➕ New Chat", use_container_width=True):
        save_conversation()  # Save current before creating new
        st.session_state.messages = []
        st.session_state.current_conversation = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.rerun()

    # List past conversations
    convos = list_conversations()
    if convos:
        with st.expander(f"📂 History ({len(convos)})", expanded=False):
            for c in convos[:10]:  # Show last 10
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"💬 {c['name'][:15]}...", key=f"load_{c['name']}", use_container_width=True
                    ):
                        save_conversation()  # Save current first
                        load_conversation(c["name"])
                        st.rerun()
                with col2:
                    st.caption(f"{c['messages']}msg")

    st.divider()

    # COMPASS Status (if active)
    if st.session_state.compass_status.get("active"):
        st.subheader("🧭 COMPASS")
        st.info(f"Stage: {st.session_state.compass_status.get('stage', 'Unknown')}")
        st.caption(st.session_state.compass_status.get("task", "")[:50])
        st.divider()

    # Connection & Tools
    st.subheader("🔌 Connection")
    status_box = st.empty()
    status_box.info("Connecting...")

    client = get_client()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if not st.session_state.tools_cache:
            tools = loop.run_until_complete(client.list_tools())
            st.session_state.tools_cache = tools

        status_box.success(f"✓ {len(st.session_state.tools_cache)} Tools")

        # DEBUG: Show what's actually in the cache
        # st.write([t['name'] for t in st.session_state.tools_cache])

        # RAA Core tools (our defined tools) vs external MCP tools
        # RAA core tools are the cognitive primitives
        core_tools = [t for t in st.session_state.tools_cache if t["name"] in RAA_CORE_TOOLS]
        mcp_tools = [t for t in st.session_state.tools_cache if t["name"] not in RAA_CORE_TOOLS]

        with st.expander(f"🧠 RAA Core ({len(core_tools)})", expanded=False):
            for t in core_tools:
                st.markdown(f"**{t['name']}**")

        with st.expander(f"🔌 MCP/External ({len(mcp_tools)})", expanded=False):
            for t in mcp_tools:
                st.markdown(f"**{t['name']}**")

    except Exception as e:
        status_box.error(f"Failed: {e}")

    st.divider()

    # LLM Config (collapsed)
    with st.expander("⚙️ LLM Config"):
        provider = st.selectbox(
            "Provider",
            ["openrouter", "openai", "anthropic"],
            index=(
                ["openrouter", "openai", "anthropic"].index(st.session_state.llm_config["provider"])
                if st.session_state.llm_config["provider"] in ["openrouter", "openai", "anthropic"]
                else 0
            ),
        )
        model = st.text_input("Model", value=st.session_state.llm_config["model"])
        api_key = st.text_input(
            "API Key", value=st.session_state.llm_config["api_key"], type="password"
        )

        if st.button("Update Config"):
            st.session_state.llm_config = {"provider": provider, "model": model, "api_key": api_key}
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_MODEL"] = model
            if api_key:
                os.environ["OPENROUTER_API_KEY"] = api_key
            st.toast("Config updated!")
            st.rerun()


# ==============================================================================
# Main Interface
# ==============================================================================

# Initialize System Prompt
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are the Reflective Agent. You connect to a backend via MCP.\n"
        "PROTOCOL:\n"
        "1. To use a tool, output valid JSON inside a code block:\n"
        "```json\n"
        '{{"tool": "tool_name", "args": {{...}}}}\n'
        "```\n"
        "2. You will receive the tool output.\n"
        "3. Synthesize the final answer.\n"
    )

# HEADER FIRST (at very top)
internals = st.session_state.internals
st.markdown(
    f"""
<div class="hud-header">
    <span class="hud-title">🧠 RAA</span>
    <span class="hud-metric">⚡ {internals.get("energy", 0.0):.1f}J</span>
    <span class="hud-metric">📊 {internals.get("entropy", 0.0):.2f}</span>
    <span class="hud-metric">🎯 {internals.get("goals", 0)}</span>
    <span class="hud-metric">🌡️ {internals.get("temperature", 0.7):.1f}</span>
    <span class="hud-phase">{internals.get("phase", "Unknown")}</span>
</div>
""",
    unsafe_allow_html=True,
)

# TABS BELOW HEADER
tab_chat, tab_topo, tab_internal, tab_config = st.tabs(
    ["💬 Signal", "🕸️ Topology", "⚙️ Internals", "🛠️ Config"]
)

# ==============================================================================
# TAB 1: Cognitive Signal (Chat)
# ==============================================================================
with tab_chat:
    # Main chat area with fixed bottom input
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    # Chat history container
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            render_message(msg)

    st.markdown("</div>", unsafe_allow_html=True)

    # Fixed bottom input area
    st.markdown('<div class="fixed-input-area">', unsafe_allow_html=True)

    # Attach button and file indicator
    attach_col, spacer = st.columns([1, 10])
    with attach_col:
        if st.button("📎", key="attach_btn", help="Attach file", use_container_width=True):
            st.session_state.show_uploader = not st.session_state.show_uploader

    # Show attached file indicator
    if st.session_state.uploaded_file:
        st.caption(f"📎 {st.session_state.uploaded_file.name}")

    # File uploader (shown when toggled)
    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader(
            "Attach file",
            type=["png", "jpg", "jpeg", "gif", "csv", "txt", "json", "md"],
            key="file_uploader",
            label_visibility="collapsed",
        )
        if uploaded_file and uploaded_file != st.session_state.uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.toast(f"File attached: {uploaded_file.name}")
            st.session_state.show_uploader = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Inject cognitive signal..."):
        # Include file context if attached
        file_context = ""
        file_meta = {}
        if st.session_state.uploaded_file:
            uf = st.session_state.uploaded_file
            file_meta = {"file_name": uf.name, "file_size": uf.size}
            # Read text files into context
            if uf.name.endswith((".txt", ".md", ".json", ".csv")):
                try:
                    file_content = uf.read().decode("utf-8")
                    file_context = (
                        f"\n\n[File Contents of {uf.name}]:\n{file_content[:2000]}..."
                        if len(file_content) > 2000
                        else f"\n\n[File Contents of {uf.name}]:\n{file_content}"
                    )
                except Exception:
                    file_context = f"\n\n[Binary file attached: {uf.name}]"
            else:
                file_context = f"\n\n[Image file attached: {uf.name}]"
            st.session_state.uploaded_file = None  # Clear after use

        # Add User Message with context
        add_message("user", prompt + file_context, meta=file_meta)
        save_conversation()  # Auto-save after each message
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
                # Combine System Prompt (tool protocol) + Agent Prompt (cognitive behavior)
                # Filter tools to only RAA Core for the agent context to prevent flooding
                agent_tools = [
                    t for t in st.session_state.tools_cache if t["name"] in RAA_CORE_TOOLS
                ]

                tools_desc = json.dumps(agent_tools, indent=2)
                full_system_prompt = (
                    f"{st.session_state.system_prompt}\n\n"
                    f"TOOLS AVAILABLE:\n{tools_desc}\n\n"
                    f"--- AGENT PROTOCOL ---\n{st.session_state.agent_prompt[:8000]}"
                )

                try:
                    llm = LLMFactory.create_provider(
                        st.session_state.llm_config["provider"],
                        st.session_state.llm_config["model"],
                    )

                    # Reconstruct history
                    history_str = ""
                    for m in st.session_state.messages:
                        content = m["content"]
                        if m["type"] == "tool_result":
                            content = f"Tool Output: {content}"
                        history_str += f"{m['role'].upper()}: {content}\n"

                    openai_tools = convert_mcp_to_openai_tools(agent_tools)

                    raw_response = llm.generate(
                        system_prompt=full_system_prompt,
                        user_prompt=history_str,
                        tools=openai_tools,
                    )

                    # Robust JSON extraction: find ```json ... ``` and parse
                    tool_call = None
                    tool_match = None
                    json_block_match = re.search(r"```json\s*(.*?)```", raw_response, re.DOTALL)
                    if json_block_match:
                        json_text = json_block_match.group(1).strip()
                        try:
                            tool_call = json.loads(json_text)
                            tool_match = json_block_match
                        except json.JSONDecodeError:
                            tool_call = None

                    if tool_call and "tool" in tool_call and tool_match:
                        t_name = tool_call.get("tool")
                        t_args = tool_call.get("args", {})

                        # Extract Pre-computation Thought (if any)
                        thought_content = raw_response[: tool_match.start()].strip()

                        if thought_content:
                            add_message("assistant", thought_content)

                        # Add Tool Call Message
                        add_message(
                            "assistant",
                            "Executing Tool...",
                            "tool_call",
                            {"tool": t_name, "args": t_args},
                        )
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
    st.subheader("🕸️ Knowledge Graph")

    # Store topology in session state
    if "topology_data" not in st.session_state:
        st.session_state.topology_data = None

    col1, col2 = st.columns([3, 1])
    with col2:
        node_label = st.selectbox(
            "Node Type", ["Thought", "Goal", "Tool", "Pattern"], key="topo_label"
        )
        node_limit = st.slider("Max Nodes", 5, 50, 15, key="topo_limit")

    if st.button("🔄 Fetch Graph", key="topo_refresh"):
        with st.spinner("Fetching graph data..."):
            try:
                client = get_client()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    client.call_tool(
                        "mcp_inspect_graph",
                        {"mode": "nodes", "label": node_label, "limit": node_limit},
                    )
                )
                loop.close()
                st.session_state.topology_data = (
                    json.loads(result) if isinstance(result, str) else result
                )
                st.toast("Graph refreshed!")
            except Exception as e:
                st.error(f"Failed to fetch: {e}")

    # Render graph
    graph = graphviz.Digraph()
    graph.attr(bgcolor="transparent", rankdir="TB")
    graph.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#1a1a2e",
        fontcolor="white",
        fontname="Outfit",
    )
    graph.attr("edge", color="#6c5ce7", penwidth="1.5")

    if st.session_state.topology_data and "nodes" in st.session_state.topology_data:
        nodes = st.session_state.topology_data["nodes"]
        for node in nodes[:node_limit]:
            node_id = node.get("id", str(hash(str(node))))
            node_name = node.get("name", node.get("content", "Node"))[:30]
            graph.node(str(node_id), node_name)

        # Add some edges if relationships exist
        if "relationships" in st.session_state.topology_data:
            for rel in st.session_state.topology_data["relationships"]:
                graph.edge(str(rel.get("from")), str(rel.get("to")), label=rel.get("type", ""))
    else:
        # Default visualization
        graph.node("Core", "RAA Core")
        graph.node("Goals", "Goal Controller")
        graph.node("Memory", "Memory Manifold")
        graph.node("Tools", "Tool Registry")
        graph.edge("Core", "Goals")
        graph.edge("Core", "Memory")
        graph.edge("Core", "Tools")
        graph.edge("Memory", "Goals", style="dashed")

    st.graphviz_chart(graph, use_container_width=True)

# ==============================================================================
# TAB 3: Director Internals
# ==============================================================================
with tab_internal:
    st.subheader("⚡ Metabolic Ledger")
    col1, col2, col3 = st.columns(3)
    internals = st.session_state.internals

    col1.metric("Energy", f"{internals.get('energy', 0.0):.2f}J")
    col2.metric("Entropy", f"{internals.get('entropy', 0.0):.2f}")
    col3.metric("Phase", internals.get("phase", "Unknown"))

    st.markdown("#### 📈 Thermodynamic History")

    # Use real history data if available
    import pandas as pd

    if st.session_state.energy_history:
        df = pd.DataFrame(st.session_state.energy_history)
        # Enforce numeric types
        df["energy"] = pd.to_numeric(df["energy"], errors="coerce").fillna(0.0)
        df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce").fillna(0.0)
        st.line_chart(df.set_index("time")[["energy", "entropy"]])
    else:
        st.info("No history data yet. Metrics will appear after system activity.")

    st.divider()
    st.subheader("🎯 Active Goals")

    active_goals = internals.get("active_goal_list", {})
    if active_goals:
        for gid, gdata in active_goals.items():
            with st.container():
                st.markdown(f"**{gid}**")
                st.caption(gdata.get("description", "No description"))
                st.progress(gdata.get("utility", 0.5))
    else:
        st.info("No active goals. Use `set_goal` to create one.")

# ==============================================================================
# TAB 4: System Config
# ==============================================================================
with tab_config:
    # Dual Prompt System
    st.subheader("🛠️ Tool Protocol (System Prompt)")
    st.caption("Controls how the LLM calls tools. Keep this minimal.")
    new_system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=150,
        key="system_prompt_editor",
    )

    if st.button("💾 Save System Prompt"):
        st.session_state.system_prompt = new_system_prompt
        st.toast("System Prompt Updated")

    st.divider()

    st.subheader("🧠 Agent Instructions (from RAA_AGENT.md)")
    st.caption("Rich cognitive protocol loaded from RAA_AGENT.md. Guides agent behavior.")

    with st.expander("View/Edit Agent Prompt", expanded=False):
        new_agent_prompt = st.text_area(
            "Agent Prompt",
            value=st.session_state.agent_prompt,
            height=400,
            key="agent_prompt_editor",
        )

        if st.button("💾 Save Agent Prompt"):
            st.session_state.agent_prompt = new_agent_prompt
            st.toast("Agent Prompt Updated")

    st.divider()
    st.caption(
        f"Agent Prompt: {len(st.session_state.agent_prompt)} chars | System Prompt: {len(st.session_state.system_prompt)} chars"
    )
