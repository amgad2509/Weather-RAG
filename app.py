import torch
import types

torch.classes.__path__ = types.SimpleNamespace(_path=[])

import os
import requests
import streamlit as st
from datetime import datetime
import json
import re
import html

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# -------------------------
# API CONFIG (STREAMING + QA ENDPOINTS)
# -------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
CHAT_STREAM_URL = f"{API_BASE}/api/v1/chat/stream"
CHAT_QA_URL = f"{API_BASE}/api/v1/chat/qa"

# Avatars (keep consistent everywhere)
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"


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
    - If <reasoning> started but not closed => hide it
    - If closed => show answer only
    - If no reasoning tags => show raw
    """
    if not raw:
        return ""
    lower = raw.lower()
    if "<reasoning>" in lower and "</reasoning>" not in lower:
        before = lower.split("<reasoning>", 1)[0]
        return before.strip()
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
    c = (msg.content or "").strip()
    tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "additional_kwargs", {}).get("tool_calls")
    return (not c) and bool(tool_calls)


def render_sources(sources) -> str:
    if not sources:
        return ""
    lines = []
    for s in sources:
        url = (s.get("url") if isinstance(s, dict) else None) or ""
        name = (s.get("name") if isinstance(s, dict) else None) or url
        if url:
            lines.append(f"- [{name}]({url})")
    return "\n".join(lines)


def call_qa_api(message: str):
    """
    Calls FastAPI /api/v1/chat/qa
    Expects response: {"answer": "...", "sources": [...]}
    """
    payload = {"message": message}

    try:
        r = requests.post(CHAT_QA_URL, json=payload, timeout=(10, 180))
    except requests.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")

    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"detail": r.text}
        raise RuntimeError(f"HTTP {r.status_code}: {err}")

    try:
        data = r.json()
    except Exception:
        raise RuntimeError("API returned non-JSON response.")

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response: {data}")

    answer = (data.get("answer") if isinstance(data, dict) else "") or ""
    sources = data.get("sources") or []
    return str(answer), sources


# -------------------------
# SESSION STATE INIT
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------
# TOP ACTIONS
# -------------------------
col1, col2 = st.columns([1, 6])
with col1:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        tlog("Chat cleared by user.")
        st.rerun()

with col2:
    st.caption(f"Streaming API: {CHAT_STREAM_URL} | QA API: {CHAT_QA_URL}")


# -------------------------
# RENDER EXISTING CHAT FIRST
# -------------------------
assistant_i = 0

for msg in st.session_state.chat_history:
    timestamp = datetime.now().strftime("%H:%M")

    if isinstance(msg, ToolMessage):
        continue

    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(render_bubble(msg.content, "user", timestamp), unsafe_allow_html=True)

    elif isinstance(msg, AIMessage):
        if is_tool_only_ai(msg):
            continue

        reasoning_text, answer_text = split_reasoning(msg.content or "")
        content_to_show = answer_text.strip() if answer_text else (msg.content or "").strip()

        if not content_to_show and not reasoning_text:
            continue

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            if reasoning_text:
                unique_label = "Reasoning" + ("\u200b" * assistant_i)
                with st.expander(unique_label, expanded=False):
                    st.markdown(reasoning_text)

            if content_to_show:
                st.markdown(render_bubble(content_to_show, "assistant", timestamp), unsafe_allow_html=True)

            sources_md = render_sources((getattr(msg, "additional_kwargs", {}) or {}).get("sources"))
            if sources_md:
                st.markdown("**Sources**")
                st.markdown(sources_md)

        assistant_i += 1


# -------------------------
# CHAT INPUT
# -------------------------
user_input = st.chat_input("Type your message here...")


def stream_from_api(message: str, placeholder_md):
    """
    Connect to FastAPI SSE stream and update UI token-by-token.
    Returns (full raw text, sources) where sources come from final SSE event.
    """
    ts = datetime.now().strftime("%H:%M")
    full_text = ""
    sources = []

    # initial empty assistant bubble
    placeholder_md.markdown(render_bubble("", "assistant", ts), unsafe_allow_html=True)

    payload = {"message": message}

    try:
        with requests.post(
            CHAT_STREAM_URL,
            json=payload,
            stream=True,
            timeout=(10, 600),  # connect/read timeouts
            headers={"Accept": "text/event-stream"},
        ) as r:
            if r.status_code != 200:
                try:
                    err = r.json()
                except Exception:
                    err = {"detail": r.text}
                raise RuntimeError(f"HTTP {r.status_code}: {err}")

            for raw_line in r.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line.strip()
                if not line:
                    continue

                # SSE payload line
                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue

                try:
                    ev = json.loads(data_str)
                except Exception:
                    continue

                etype = ev.get("type")

                if etype == "status":
                    # optional: ignore or log
                    continue

                if etype == "delta":
                    piece = ev.get("value", "") or ""
                    if piece:
                        full_text += piece
                        shown = strip_reasoning_during_stream(full_text)
                        placeholder_md.markdown(render_bubble(shown, "assistant", ts), unsafe_allow_html=True)

                elif etype == "error":
                    raise RuntimeError(ev.get("message", "Unknown streaming error"))

                elif etype == "done":
                    sources = ev.get("sources") or []
                    break

        # final render (ensure reasoning stripped for visible bubble)
        shown = strip_reasoning_during_stream(full_text)
        placeholder_md.markdown(render_bubble(shown, "assistant", ts), unsafe_allow_html=True)

        return full_text, sources

    except Exception as e:
        raise e


# -------------------------
# HANDLE NEW INPUT (STREAM ANSWER, THEN FETCH SOURCES FROM QA)
# -------------------------
if user_input:
    tlog(f"User: {_safe_preview(user_input, 200)}")

    # store user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # render user bubble
    timestamp = datetime.now().strftime("%H:%M")
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(render_bubble(user_input, "user", timestamp), unsafe_allow_html=True)

    # stream assistant response, then fetch sources via QA endpoint
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        try:
            tlog(f"Streaming from API: {CHAT_STREAM_URL}")
            full_answer, _ = stream_from_api(user_input, answer_placeholder)

            shown = strip_reasoning_during_stream(full_answer)
            ts = datetime.now().strftime("%H:%M")
            answer_placeholder.markdown(render_bubble(shown, "assistant", ts), unsafe_allow_html=True)

            sources = []
            try:
                tlog(f"Calling QA API for sources: {CHAT_QA_URL}")
                _, sources = call_qa_api(user_input)
            except Exception as e:
                tlog(f"ERROR during QA call (sources fetch): {repr(e)}")

            sources_md = render_sources(sources)
            if sources_md:
                sources_placeholder.markdown("**Sources**\n" + sources_md)

            if full_answer.strip():
                st.session_state.chat_history.append(AIMessage(content=full_answer, additional_kwargs={"sources": sources}))
            else:
                st.session_state.chat_history.append(AIMessage(content="(Empty answer)", additional_kwargs={"sources": sources}))

        except Exception as e:
            tlog(f"ERROR during streaming: {repr(e)}")
            st.session_state.chat_history.append(
                AIMessage(content=f"Sorry, an error occurred while streaming the answer:\n{e}")
            )

    st.rerun()
