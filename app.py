# import torch
# import types

# torch.classes.__path__ = types.SimpleNamespace(_path=[])

# import os
# import requests
# import streamlit as st
# from datetime import datetime
# import json
# import re
# import html

# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# # -------------------------
# # API CONFIG
# # -------------------------
# API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
# CHAT_URL = f"{API_BASE}/api/v1/chat"


# # -------------------------
# # UI CONFIG
# # -------------------------
# st.set_page_config(
#     page_title="Weather & Clothing Assistant",
#     page_icon="⛅",
#     layout="wide",
# )

# st.title("👕🌦️ Weather & Clothing Chat Assistant")
# st.markdown("Ask me about what to wear, weather conditions, or anything else!")

# st.markdown(
#     """
# <style>
# .element-container .markdown-text-container {
#     font-size: 16px !important;
#     line-height: 1.6 !important;
# }
# .chat-bubble {
#     padding: 10px;
#     border-radius: 10px;
#     max-width: 80%;
#     margin-bottom: 5px;
#     color: #222;
#     font-weight: 500;
#     white-space: normal;
# }
# .user-bubble {
#     background-color: #DCF8C6;
#     align-self: flex-end;
# }
# .bot-bubble {
#     background-color: #F1F0F0;
#     align-self: flex-start;
# }
# .timestamp {
#     font-size: 12px;
#     color: #666;
#     float: right;
# }
# </style>
# """,
#     unsafe_allow_html=True,
# )


# # -------------------------
# # REASONING PARSER
# # -------------------------
# _REASONING_RE = re.compile(r"(?is)<reasoning>\s*(.*?)\s*</reasoning>")

# def split_reasoning(content: str):
#     if not content:
#         return None, ""
#     m = _REASONING_RE.search(content)
#     if not m:
#         return None, content.strip()
#     reasoning = m.group(1).strip()
#     answer = _REASONING_RE.sub("", content, count=1).strip()
#     return reasoning, answer

# def strip_reasoning_during_stream(raw: str):
#     """
#     Kept for compatibility; not used in non-streaming mode.
#     """
#     if not raw:
#         return ""
#     lower = raw.lower()
#     if "<reasoning>" in lower and "</reasoning>" not in lower:
#         before = lower.split("<reasoning>", 1)[0]
#         return before.strip()
#     _, answer = split_reasoning(raw)
#     return answer


# # -------------------------
# # TERMINAL LOGGING ONLY
# # -------------------------
# def tlog(message: str) -> None:
#     ts = datetime.now().strftime("%H:%M:%S")
#     print(f"[{ts}] {message}", flush=True)

# def _safe_preview(x, n=160):
#     if x is None:
#         return ""
#     if not isinstance(x, str):
#         try:
#             x = json.dumps(x, ensure_ascii=False)
#         except Exception:
#             x = str(x)
#     x = x.strip().replace("\n", " ")
#     return x[:n] + ("..." if len(x) > n else "")

# def render_bubble(content: str, who: str, timestamp: str):
#     safe = html.escape(content).replace("\n", "<br>")
#     bubble_class = "user-bubble" if who == "user" else "bot-bubble"
#     return f"""
# <div class="chat-bubble {bubble_class}">
#   {safe}
#   <div class="timestamp">{timestamp}</div>
# </div>
# """

# def is_tool_only_ai(msg: AIMessage) -> bool:
#     """
#     Still kept; in API mode there are no tool-only AI messages usually.
#     """
#     c = (msg.content or "").strip()
#     tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "additional_kwargs", {}).get("tool_calls")
#     return (not c) and bool(tool_calls)


# # -------------------------
# # SESSION STATE INIT
# # -------------------------
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


# # -------------------------
# # TOP ACTIONS
# # -------------------------
# col1, col2 = st.columns([1, 6])
# with col1:
#     if st.button("Clear Chat"):
#         st.session_state.chat_history = []
#         tlog("Chat cleared by user.")
#         st.rerun()

# with col2:
#     st.caption(f"API: {CHAT_URL}")


# # -------------------------
# # RENDER EXISTING CHAT FIRST
# # -------------------------
# assistant_i = 0

# for msg in st.session_state.chat_history:
#     timestamp = datetime.now().strftime("%H:%M")

#     if isinstance(msg, ToolMessage):
#         continue

#     if isinstance(msg, HumanMessage):
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(render_bubble(msg.content, "user", timestamp), unsafe_allow_html=True)

#     elif isinstance(msg, AIMessage):
#         if is_tool_only_ai(msg):
#             continue

#         reasoning_text, answer_text = split_reasoning(msg.content or "")
#         content_to_show = answer_text.strip() if answer_text else (msg.content or "").strip()

#         if not content_to_show and not reasoning_text:
#             continue

#         with st.chat_message("assistant", avatar="🤖"):
#             if reasoning_text:
#                 unique_label = "Reasoning" + ("\u200b" * assistant_i)
#                 with st.expander(unique_label, expanded=False):
#                     st.markdown(reasoning_text)

#             if content_to_show:
#                 st.markdown(render_bubble(content_to_show, "assistant", timestamp), unsafe_allow_html=True)

#         assistant_i += 1


# # -------------------------
# # CHAT INPUT
# # -------------------------
# user_input = st.chat_input("Type your message here...")


# def call_chat_api(message: str) -> str:
#     """
#     Calls FastAPI /api/v1/chat
#     Expects response: {"answer": "..."}
#     """
#     payload = {"message": message}

#     try:
#         r = requests.post(CHAT_URL, json=payload, timeout=(10, 180))
#     except requests.RequestException as e:
#         raise RuntimeError(f"API request failed: {e}")

#     if r.status_code != 200:
#         try:
#             err = r.json()
#         except Exception:
#             err = {"detail": r.text}
#         raise RuntimeError(f"HTTP {r.status_code}: {err}")

#     try:
#         data = r.json()
#     except Exception:
#         raise RuntimeError("API returned non-JSON response.")

#     ans = (data.get("answer") if isinstance(data, dict) else "") or ""
#     return str(ans)


# # -------------------------
# # HANDLE NEW INPUT (API call)
# # -------------------------
# if user_input:
#     tlog(f"User: {_safe_preview(user_input, 200)}")

#     # Append user message to history
#     st.session_state.chat_history.append(HumanMessage(content=user_input))

#     # Render user bubble immediately
#     timestamp = datetime.now().strftime("%H:%M")
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(render_bubble(user_input, "user", timestamp), unsafe_allow_html=True)

#     # Call API and render assistant answer
#     with st.chat_message("assistant", avatar="🤖"):
#         placeholder = st.empty()
#         ts = datetime.now().strftime("%H:%M")
#         placeholder.markdown(render_bubble("Thinking...", "assistant", ts), unsafe_allow_html=True)

#         try:
#             tlog(f"Calling API: {CHAT_URL}")
#             answer = call_chat_api(user_input)

#             if answer.strip():
#                 st.session_state.chat_history.append(AIMessage(content=answer))
#             else:
#                 st.session_state.chat_history.append(AIMessage(content="(Empty answer)"))

#         except Exception as e:
#             tlog(f"ERROR during API call: {repr(e)}")
#             st.session_state.chat_history.append(
#                 AIMessage(content=f"Sorry, an error occurred while calling the API:\n{e}")
#             )

#     # Rerun to render the reasoning expander cleanly and persist state
#     st.rerun()

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
# API CONFIG (STREAMING ENDPOINT)
# -------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
CHAT_STREAM_URL = f"{API_BASE}/api/v1/chat/stream"


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
    st.caption(f"Streaming API: {CHAT_STREAM_URL}")


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
        if is_tool_only_ai(msg):
            continue

        reasoning_text, answer_text = split_reasoning(msg.content or "")
        content_to_show = answer_text.strip() if answer_text else (msg.content or "").strip()

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


def stream_from_api(message: str, placeholder_md) -> str:
    """
    Connect to FastAPI SSE stream and update UI token-by-token.
    Returns full raw text (may include <reasoning>...</reasoning>).
    """
    ts = datetime.now().strftime("%H:%M")
    full_text = ""

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
                    break

        # final render (ensure reasoning stripped for visible bubble)
        shown = strip_reasoning_during_stream(full_text)
        placeholder_md.markdown(render_bubble(shown, "assistant", ts), unsafe_allow_html=True)

        return full_text

    except Exception as e:
        raise e


# -------------------------
# HANDLE NEW INPUT (TRUE STREAMING FROM API)
# -------------------------
if user_input:
    tlog(f"User: {_safe_preview(user_input, 200)}")

    # store user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # render user bubble
    timestamp = datetime.now().strftime("%H:%M")
    with st.chat_message("user", avatar="👤"):
        st.markdown(render_bubble(user_input, "user", timestamp), unsafe_allow_html=True)

    # stream assistant response
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()

        try:
            tlog(f"Streaming from API: {CHAT_STREAM_URL}")
            full_answer = stream_from_api(user_input, placeholder)

            # store full raw answer (so rerun shows reasoning expander correctly)
            if full_answer.strip():
                st.session_state.chat_history.append(AIMessage(content=full_answer))
            else:
                st.session_state.chat_history.append(AIMessage(content="(Empty answer)"))

        except Exception as e:
            tlog(f"ERROR during API streaming: {repr(e)}")
            st.session_state.chat_history.append(
                AIMessage(content=f"Sorry, an error occurred while calling the API stream:\n{e}")
            )

    st.rerun()

