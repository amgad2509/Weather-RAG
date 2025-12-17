import torch
import types

torch.classes.__path__ = types.SimpleNamespace(_path=[])

import streamlit as st
from datetime import datetime
import json
import re
import asyncio
import html

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
    white-space: normal;
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
    if not content:
        return None, ""
    m = _REASONING_RE.search(content)
    if not m:
        return None, content.strip()
    reasoning = m.group(1).strip()
    answer = _REASONING_RE.sub("", content, count=1).strip()
    return reasoning, answer

def strip_reasoning_during_stream(raw: str):
    """
    While streaming:
    - If <reasoning> started but not closed => show nothing (avoid showing tags)
    - If closed => show answer only
    - If no reasoning tags => show raw
    """
    if not raw:
        return ""
    lower = raw.lower()
    if "<reasoning>" in lower and "</reasoning>" not in lower:
        # reasoning is currently streaming, hide it
        before = lower.split("<reasoning>", 1)[0]
        return before.strip()  # usually empty
    # if closed, remove it
    _, answer = split_reasoning(raw)
    return answer


# -------------------------
# TERMINAL LOGGING ONLY
# -------------------------
def tlog(message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {message}", flush=True)

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

def render_bubble(content: str, who: str, timestamp: str):
    safe = html.escape(content).replace("\n", "<br>")
    bubble_class = "user-bubble" if who == "user" else "bot-bubble"
    return f"""
<div class="chat-bubble {bubble_class}">
  {safe}
  <div class="timestamp">{timestamp}</div>
</div>
"""

def is_tool_only_ai(msg: AIMessage) -> bool:
    """
    Skip AIMessage that has no content and only exists to request tool calls
    (these were causing the empty white lines).
    """
    c = (msg.content or "").strip()
    tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "additional_kwargs", {}).get("tool_calls")
    return (not c) and bool(tool_calls)


# -------------------------
# SESSION STATE INIT
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
        tlog("Chat cleared by user.")
        st.rerun()


# -------------------------
# RENDER EXISTING CHAT FIRST
# -------------------------
assistant_i = 0

for msg in st.session_state.chat_history:
    timestamp = datetime.now().strftime("%H:%M")

    if isinstance(msg, ToolMessage):
        continue

    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(render_bubble(msg.content, "user", timestamp), unsafe_allow_html=True)

    elif isinstance(msg, AIMessage):
        # remove empty tool-only AI messages (fixes the blank lines)
        if is_tool_only_ai(msg):
            continue

        reasoning_text, answer_text = split_reasoning(msg.content or "")
        content_to_show = answer_text.strip() if answer_text else (msg.content or "").strip()

        # If nothing to show, skip (avoids another blank bar)
        if not content_to_show and not reasoning_text:
            continue

        with st.chat_message("assistant", avatar="🤖"):
            if reasoning_text:
                unique_label = "Reasoning" + ("\u200b" * assistant_i)
                with st.expander(unique_label, expanded=False):
                    st.markdown(reasoning_text)

            if content_to_show:
                st.markdown(render_bubble(content_to_show, "assistant", timestamp), unsafe_allow_html=True)

        assistant_i += 1


# -------------------------
# CHAT INPUT
# -------------------------
user_input = st.chat_input("Type your message here...")


def _run_coro(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        # In case Streamlit already has a running loop in some environments
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        raise


async def run_graph_token_stream(new_history, placeholder_md):
    """
    Runs the graph via astream_events (token-by-token) and updates the placeholder.
    Also captures final messages from the root chain end event when available.
    """
    agent = st.session_state.agent

    # buffers
    current_text = ""
    last_nonempty_text = ""
    final_messages = None

    # initial blank bubble (optional)
    ts = datetime.now().strftime("%H:%M")
    placeholder_md.markdown(render_bubble("", "assistant", ts), unsafe_allow_html=True)

    if hasattr(agent.app, "astream_events"):
        async for ev in agent.app.astream_events({"messages": new_history}, version="v2"):
            et = ev.get("event", "")
            data = ev.get("data", {}) or {}

            if et == "on_chat_model_start":
                current_text = ""

            elif et == "on_chat_model_stream":
                chunk = data.get("chunk")
                piece = ""
                try:
                    piece = getattr(chunk, "content", "") or ""
                    if isinstance(piece, list):
                        # some models stream structured content; best-effort stringify
                        piece = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in piece])
                    piece = str(piece)
                except Exception:
                    piece = ""

                if piece:
                    current_text += piece
                    shown = strip_reasoning_during_stream(current_text)
                    placeholder_md.markdown(render_bubble(shown, "assistant", ts), unsafe_allow_html=True)

            elif et == "on_chat_model_end":
                if current_text.strip():
                    last_nonempty_text = current_text

            elif et == "on_tool_start":
                name = ev.get("name", "tool")
                tool_in = data.get("input")
                tlog(f"TOOL_CALL -> {name} | args={_safe_preview(tool_in, 220)}")

            elif et == "on_tool_end":
                name = ev.get("name", "tool")
                tool_out = data.get("output")
                tlog(f"TOOL_RESULT <- {name} | output={_safe_preview(tool_out, 260)}")

            elif et == "on_chain_end":
                out = data.get("output") or data.get("outputs")
                if isinstance(out, dict) and "messages" in out and isinstance(out["messages"], list):
                    final_messages = out["messages"]

        return final_messages, last_nonempty_text

    # Fallback: no events support (no token streaming)
    tlog("WARNING: astream_events not available. Falling back to non-token streaming.")
    if hasattr(agent.app, "ainvoke"):
        out = await agent.app.ainvoke({"messages": new_history})
        if isinstance(out, dict) and "messages" in out:
            return out["messages"], ""
    return None, ""


# -------------------------
# HANDLE NEW INPUT (token streaming)
# -------------------------
if user_input:
    tlog(f"User: {_safe_preview(user_input, 200)}")

    # Append user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Render user bubble immediately
    timestamp = datetime.now().strftime("%H:%M")
    with st.chat_message("user", avatar="👤"):
        st.markdown(render_bubble(user_input, "user", timestamp), unsafe_allow_html=True)

    # Assistant streaming placeholder
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()

        try:
            tlog("Starting graph execution (astream_events for token streaming).")
            final_msgs, last_text = _run_coro(
                run_graph_token_stream(st.session_state.chat_history, placeholder)
            )

            if final_msgs is not None:
                # replace/merge with final state messages
                # safest: just set to final (it should include full state)
                st.session_state.chat_history = final_msgs
            else:
                # fallback: at least keep the streamed final text
                if last_text.strip():
                    st.session_state.chat_history.append(AIMessage(content=last_text))

        except Exception as e:
            tlog(f"ERROR during graph execution: {repr(e)}")
            st.session_state.chat_history.append(
                AIMessage(content=f"Sorry, an error occurred while running the agent:\n{e}")
            )

    # Rerun to render the reasoning expander cleanly and persist state
    st.rerun()