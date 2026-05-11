import streamlit as st
import sqlite3
import os
import uuid
from dotenv import load_dotenv
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import initial_state

# Load environment variables from .env
load_dotenv()

# Page config
st.set_page_config(page_title="Advanced LangGraph Agent", layout="wide")

st.title("🤖 Advanced LangGraph Agentic Support")
st.markdown("---")

# Persistent state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    thread_id = st.text_input("Thread ID", value=st.session_state.thread_id)
    st.session_state.thread_id = thread_id
    
    if st.button("New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("System Info")
    st.info(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    st.info(f"Tracing: {os.getenv('LANGSMITH_TRACING', 'false')}")

# Build graph
checkpointer = build_checkpointer()
graph = build_graph(checkpointer=checkpointer)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Main Chat Interface
chat_container = st.container()

def display_history():
    with chat_container:
        for entry in st.session_state.history:
            if entry["role"] == "user":
                st.chat_message("user").write(entry["content"])
            elif entry["role"] == "assistant":
                st.chat_message("assistant").write(entry["content"])
            elif entry["role"] == "system":
                st.status(entry["content"], expanded=False)

display_history()

# Check for interrupts (HITL)
current_state = graph.get_state(config)
if current_state.next:
    if "approval" in current_state.next:
        st.warning("⚠️ Approval Required for Risky Action!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve"):
                graph.update_state(config, {"approval": {"approved": True, "comment": "Approved via UI"}}, as_node="approval")
                st.rerun()
        with col2:
            if st.button("Reject"):
                graph.update_state(config, {"approval": {"approved": False, "comment": "Rejected via UI"}}, as_node="approval")
                st.rerun()

# User Input
if prompt := st.chat_input("Ask something (e.g. 'Refund order 123' or 'Check status')"):
    st.session_state.history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        status_box = st.empty()
        # Initialize state for new thread if needed
        if not current_state.values:
            inputs = initial_state()
            inputs["query"] = prompt
        else:
            inputs = None # Resume from previous state
            
        # Streaming logic
        full_response = ""
        
        # We use stream() to get events
        for event in graph.stream(inputs if inputs else {"query": prompt}, config, stream_mode="updates"):
            for node_name, values in event.items():
                with st.status(f"Executing node: **{node_name}**", expanded=False) as status:
                    if "final_answer" in values:
                        full_response = values["final_answer"]
                    status.update(label=f"Node **{node_name}** completed", state="complete")
        
        if full_response:
            st.markdown(full_response)
            st.session_state.history.append({"role": "assistant", "content": full_response})
        else:
            # Check if we are stopped at an interrupt
            current_state = graph.get_state(config)
            if current_state.next:
                st.info("System waiting for approval...")
                st.rerun()
            else:
                st.error("Workflow finished without an answer. Please check state.")

# Time Travel Feature
st.markdown("---")
st.subheader("🕰️ Time Travel (State History)")
history_list = list(graph.get_state_history(config))
if history_list:
    selected_idx = st.select_slider("Select past state version", options=range(len(history_list)), format_func=lambda i: f"Step {len(history_list)-i}")
    past_state = history_list[selected_idx]
    
    with st.expander("View State Snapshot"):
        st.json(past_state.values)
        if st.button("Restore to this state"):
            # Restore logic: Overwrite current state with past state values
            # This creates a new checkpoint that matches the past state
            graph.update_state(config, past_state.values)
            st.success(f"🕰️ Restored to Step {len(history_list)-selected_idx}! Next interaction will start from this point.")
            st.rerun()
