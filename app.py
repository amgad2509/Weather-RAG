import torch
import types

torch.classes.__path__ = types.SimpleNamespace(_path=[])

import streamlit as st
from datetime import datetime
import json
import re

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agent import WeatherActivityClothingAgent


# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(
    page_title="Weather & Clothing Assistant",
    page_icon="⛅",
    layout="wide",
)

st.title("👕🌦️ Weather & Clothing Chat Assistant")
st.markdown("Ask me about what to wear, weather conditions, or anything else!")

st.markdown(
    """
<style>
.element-container .markdown-text-container {
    font-size: 16px !important;
    line-height: 1.6 !important;
}
.chat-bubble {
    padding: 10px;
    border-radius: 10px;
    max-width: 80%;
    margin-bottom: 5px;
    color: #222;
    font-weight: 500;
}
.user-bubble {
    background-color: #DCF8C6;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #F1F0F0;
    align-self: flex-start;
}
.timestamp {
    font-size: 12px;
    color: #666;
    float: right;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# REASONING PARSER
# -------------------------
_REASONING_RE = re.compile(r"(?is)<reasoning>\s*(.*?)\s*</reasoning>")

def split_reasoning(content: str):
    """
    Returns (reasoning_text_or_None, answer_text)
    """
    if not content:
        return None, ""
    m = _REASONING_RE.search(content)
    if not m:
        return None, content.strip()
    reasoning = m.group(1).strip()
    answer = _REASONING_RE.sub("", content, count=1).strip()
    return reasoning, answer


# -------------------------
# TERMINAL LOGGING ONLY
# -------------------------
def tlog(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def msg_key(m):
    t = type(m).__name__
    c = getattr(m, "content", "")
    tcid = getattr(m, "tool_call_id", "")
    return (t, c, tcid)


def merge_messages(current, incoming):
    """
    Robust merge:
    - If incoming looks like full state (prefix matches current), replace.
    - Else treat as delta and append.
    """
    if not incoming:
        return current

    # likely full state
    if current and len(incoming) >= len(current):
        try:
            prefix_ok = True
            for i in range(min(len(current), len(incoming))):
                if msg_key(incoming[i]) != msg_key(current[i]):
                    prefix_ok = False
                    break
            if prefix_ok:
                return incoming
        except Exception:
            pass

    # delta append
    merged = list(current)
    for m in incoming:
        if merged and msg_key(merged[-1]) == msg_key(m):
            continue
        merged.append(m)
    return merged


def _safe_preview(x, n=160):
    if x is None:
        return ""
    if not isinstance(x, str):
        try:
            x = json.dumps(x, ensure_ascii=False)
        except Exception:
            x = str(x)
    x = x.strip().replace("\n", " ")
    return x[:n] + ("..." if len(x) > n else "")


def iter_tool_calls_from_ai(ai_msg):
    """
    Extract tool calls from AIMessage across different LangChain formats.
    Returns tuples: (call_id, tool_name, tool_args)
    """
    calls = getattr(ai_msg, "tool_calls", None)
    if not calls:
        calls = getattr(ai_msg, "additional_kwargs", {}).get("tool_calls")

    if not calls:
        return []

    out = []
    for c in calls:
        call_id, name, args = None, None, None

        if isinstance(c, dict):
            # LangChain style: {"name": "...", "args": {...}, "id": "..."}
            name = c.get("name")
            args = c.get("args")
            call_id = c.get("id")

            # OpenAI-like style: {"id": "...", "function": {"name": "...", "arguments": "..."}}
            if not name and "function" in c:
                name = (c.get("function") or {}).get("name")
                args = (c.get("function") or {}).get("arguments")

        else:
            # object style
            name = getattr(c, "name", None)
            args = getattr(c, "args", None)
            call_id = getattr(c, "id", None)

        if name:
            out.append((call_id, name, args))
    return out


# -------------------------
# SESSION STATE INIT
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Map tool_call_id -> tool_name (so ToolMessage can be labeled)
if "tool_call_map" not in st.session_state:
    st.session_state.tool_call_map = {}

# Prevent duplicate tool logs across steps/reruns
if "logged_tool_calls" not in st.session_state:
    st.session_state.logged_tool_calls = set()

if "logged_tool_results" not in st.session_state:
    st.session_state.logged_tool_results = set()


# -------------------------
# LOAD AGENT (cached)
# -------------------------
@st.cache_resource
def get_agent():
    return WeatherActivityClothingAgent()


with st.spinner("Loading smart weather assistant..."):
    if "agent" not in st.session_state:
        st.session_state.agent = get_agent()
        tlog("Agent initialized and cached.")
    else:
        tlog("Agent retrieved from cache.")


# -------------------------
# TOP ACTIONS
# -------------------------
col1, col2 = st.columns([1, 6])
with col1:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.tool_call_map = {}
        st.session_state.logged_tool_calls = set()
        st.session_state.logged_tool_results = set()
        tlog("Chat cleared by user.")
        st.rerun()


# -------------------------
# CHAT INPUT
# -------------------------
user_input = st.chat_input("Type your message here...")


# -------------------------
# RUN AGENT + TERMINAL LOGS (WITH TOOL NAMES)
# -------------------------
if user_input:
    tlog(f"User: {_safe_preview(user_input, 200)}")

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    try:
        tlog("Starting graph execution (stream).")

        for step_idx, step in enumerate(
            st.session_state.agent.app.stream({"messages": st.session_state.chat_history})
        ):
            for node_name, payload in step.items():
                if not (isinstance(payload, dict) and "messages" in payload):
                    continue

                incoming = payload["messages"]

                # Log tool calls / tool results from incoming (deduped)
                for m in incoming:
                    # Tool calls requested by AI
                    if isinstance(m, AIMessage):
                        for call_id, tool_name, tool_args in iter_tool_calls_from_ai(m):
                            # call_id may be None sometimes; dedupe by (tool_name + args) fallback
                            dedupe_id = call_id or f"{tool_name}:{_safe_preview(tool_args, 120)}"
                            if dedupe_id in st.session_state.logged_tool_calls:
                                continue

                            st.session_state.logged_tool_calls.add(dedupe_id)
                            if call_id:
                                st.session_state.tool_call_map[call_id] = tool_name

                            tlog(
                                f"TOOL_CALL -> {tool_name}"
                                + (f" | id={call_id}" if call_id else "")
                                + (f" | args={_safe_preview(tool_args, 220)}" if tool_args is not None else "")
                            )

                    # Tool results returned
                    if isinstance(m, ToolMessage):
                        tcid = getattr(m, "tool_call_id", None)
                        dedupe_id = tcid or f"tool_result:{_safe_preview(m.content, 120)}"
                        if dedupe_id in st.session_state.logged_tool_results:
                            continue

                        st.session_state.logged_tool_results.add(dedupe_id)

                        tool_name = getattr(m, "name", None) or (
                            st.session_state.tool_call_map.get(tcid, "unknown_tool")
                            if tcid
                            else "unknown_tool"
                        )

                        tlog(
                            f"TOOL_RESULT <- {tool_name}"
                            + (f" | id={tcid}" if tcid else "")
                            + f" | output={_safe_preview(m.content, 260)}"
                        )

                # Merge messages into chat_history (keep user queries)
                before = len(st.session_state.chat_history)
                st.session_state.chat_history = merge_messages(st.session_state.chat_history, incoming)
                after = len(st.session_state.chat_history)
                added = after - before

                last = st.session_state.chat_history[-1] if st.session_state.chat_history else None
                last_type = type(last).__name__ if last else "None"
                last_preview = _safe_preview(getattr(last, "content", ""), 160) if last else ""

                tlog(
                    f"Step {step_idx} | node='{node_name}' | incoming={len(incoming)} | added={added} | last={last_type}"
                    + (f" | last_preview='{last_preview}'" if last_preview else "")
                )

        tlog(f"Graph completed. Total messages now: {len(st.session_state.chat_history)}")

    except Exception as e:
        tlog(f"ERROR during graph execution: {repr(e)}")
        st.session_state.chat_history.append(
            AIMessage(content=f"Sorry, an error occurred while running the agent:\n{e}")
        )

    st.rerun()


# -------------------------
# RENDER CHAT (show Human + AI, hide ToolMessage)
# -------------------------
assistant_i = 0

for msg in st.session_state.chat_history:
    timestamp = datetime.now().strftime("%H:%M")

    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(
                f"""
<div class="chat-bubble user-bubble">
  {msg.content}
  <div class="timestamp">{timestamp}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            reasoning_text, answer_text = split_reasoning(msg.content)

            # Collapsible reasoning (ChatGPT-like): label must be unique (Streamlit uses label as key)
            if reasoning_text:
                unique_label = "Reasoning" + ("\u200b" * assistant_i)  # invisible uniqueness
                with st.expander(unique_label, expanded=False):
                    st.markdown(reasoning_text)

            # Show ONLY final answer in bubble
            content_to_show = answer_text if answer_text else (msg.content or "")

            st.markdown(
                f"""
<div class="chat-bubble bot-bubble">
  {content_to_show}
  <div class="timestamp">{timestamp}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        assistant_i += 1

    elif isinstance(msg, ToolMessage):
        # Keep tool messages for context, but don't show them
        continue
