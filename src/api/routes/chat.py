# src/api/routes/chat.py
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Set

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.api.schemas import AgentRequest, AgentResponse, QAResponse
from src.api.dependencies import get_agent
from src.utils.telemetry import emit, Stopwatch, _truncate  # noqa
from src.api.tracing_logger import (
    emit_trace,
    parse_sources_from_internet_output,
    parse_sources_from_retriever_output,
    sse,
)

logger = logging.getLogger(__name__)
ai_agent_router = APIRouter(prefix="/chat", tags=["chat"])




# -------------------------
# Non-streaming (legacy/simple)
# -------------------------
@ai_agent_router.post("", response_model=AgentResponse)
def chat(req: AgentRequest, request: Request) -> AgentResponse:
    """
    Non-streaming endpoint: returns a single answer string in AgentResponse.
    """
    try:
        agent = get_agent(request)
        answer = agent.invoke(req.message)

        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(status_code=500, detail="Agent returned empty answer")

        return AgentResponse(answer=answer)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Required structured JSON output endpoint
# -------------------------
@ai_agent_router.post("/qa", response_model=QAResponse)
def chat_qa(req: AgentRequest, request: Request) -> QAResponse:
    """
    Returns the required final structured JSON output (non-streaming):

    {
      "answer": "string",
      "sources": [{"name":"string","url":"string"}],
      "latency_ms": {"total": 123, "by_step": {"retrieve": 45, "llm": 60}},
      "tokens": {"prompt": 0, "completion": 0}
    }

    NOTE (current implementation):
    - sources: [] (to be filled later when you wire citations/URLs)
    - tokens: 0 (to be filled later if usage is available)
    - latency by_step: retrieve=0, llm=total (until we split steps precisely)
    """
    agent = get_agent(request)
    sw_total = Stopwatch()

    trace_id = uuid.uuid4().hex
    client_host = getattr(getattr(request, "client", None), "host", None)
    user_agent = request.headers.get("user-agent")
    route_path = str(request.url.path)

    emit_trace(
        "request_received",
        trace_id=trace_id,
        route=route_path,
        client_host=client_host,
        user_agent=_truncate(user_agent, 180),
        input_chars=len(req.message or ""),
        message_preview=_truncate(req.message, 160),
    )

    try:
        answer, sources, metrics = agent.invoke_with_sources(req.message)

        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(status_code=500, detail="Agent returned empty answer")

        total_ms = metrics.get("total_ms", sw_total.ms())
        retrieve_ms = metrics.get("retrieve_ms", 0)
        llm_ms = metrics.get("llm_ms", total_ms if retrieve_ms == 0 else max(total_ms - retrieve_ms, 0))

        payload: Dict[str, Any] = {
            "answer": answer,
            "sources": sources or [],
            "latency_ms": {
                "total": total_ms,
                "by_step": {"retrieve": retrieve_ms, "llm": llm_ms},
            },
            "tokens": {"prompt": 0, "completion": 0},
        }

        emit_trace(
            "qa_done",
            trace_id=trace_id,
            status="ok",
            latency_ms=total_ms,
            sources_count=len(sources or []),
        )

        return QAResponse.model_validate(payload)

    except HTTPException:
        emit_trace(
            "qa_error",
            trace_id=trace_id,
            status="error",
            latency_ms=sw_total.ms(),
            error="http_exception",
        )
        raise
    except Exception as e:
        logger.exception("chat_qa failed")
        emit_trace(
            "qa_error",
            trace_id=trace_id,
            status="error",
            latency_ms=sw_total.ms(),
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# True SSE streaming endpoint
# -------------------------
@ai_agent_router.post("/stream")
async def chat_stream(req: AgentRequest, request: Request):
    """
    True token/event streaming via LangGraph astream_events (when available).
    SSE events:
      - {"type":"status","value":"started"}
      - {"type":"delta","value":"..."}  (token/chunk deltas)
      - {"type":"done"}
      - {"type":"error","message":"..."}
    """
    agent = get_agent(request)

    trace_id = uuid.uuid4().hex
    sw_total = Stopwatch()

    client_host = getattr(getattr(request, "client", None), "host", None)
    user_agent = request.headers.get("user-agent")
    route_path = str(request.url.path)

    emit_trace(
        "request_received",
        trace_id=trace_id,
        route=route_path,
        client_host=client_host,
        user_agent=_truncate(user_agent, 180),
        input_chars=len(req.message or ""),
        message_preview=_truncate(req.message, 160),
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        tool_calls_count = 0
        delta_chars = 0
        deltas_count = 0
        tool_timers: Dict[str, Stopwatch] = {}
        sources: List[Dict[str, str]] = []
        sources_seen: Set[str] = set()
        retriever_added = 0

        try:
            emit_trace("stream_started", trace_id=trace_id)
            yield sse({"type": "status", "value": "started"})

            if hasattr(agent.app, "astream_events"):
                async for ev in agent.app.astream_events(
                    {"messages": [HumanMessage(content=req.message)]},
                    version="v2",
                ):
                    if await request.is_disconnected():
                        emit_trace(
                            "client_disconnected",
                            trace_id=trace_id,
                            latency_ms=sw_total.ms(),
                            tool_calls_count=tool_calls_count,
                            deltas_count=deltas_count,
                            delta_chars=delta_chars,
                        )
                        return

                    et = ev.get("event", "")
                    data = ev.get("data", {}) or {}

                    if et == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        piece = ""
                        try:
                            piece = getattr(chunk, "content", "") or ""
                            if isinstance(piece, list):
                                piece = "".join(
                                    [p.get("text", "") if isinstance(p, dict) else str(p) for p in piece]
                                )
                            piece = str(piece)
                        except Exception:
                            piece = ""

                        if piece:
                            deltas_count += 1
                            delta_chars += len(piece)
                            yield sse({"type": "delta", "value": piece})

                    elif et == "on_tool_start":
                        tool_calls_count += 1
                        name = ev.get("name") or data.get("name") or "tool"
                        tool_in = data.get("input")

                        tool_timers[name] = Stopwatch()

                        emit_trace(
                            "tool_start",
                            trace_id=trace_id,
                            tool=name,
                            tool_call_index=tool_calls_count,
                            args_preview=_truncate(tool_in, 260),
                        )

                    elif et == "on_tool_end":
                        name = ev.get("name") or data.get("name") or "tool"
                        tool_out = data.get("output")

                        timer = tool_timers.pop(name, None)
                        tool_ms = timer.ms() if timer else None

                        if name == "internet_search":
                            normalized = tool_out
                            if isinstance(tool_out, list):
                                normalized = "".join(str(p) for p in tool_out)
                            if normalized:
                                for s in parse_sources_from_internet_output(str(normalized)):
                                    url = s.get("url")
                                    if url and url not in sources_seen:
                                        sources.append(s)
                                        sources_seen.add(url)
                        elif name == "retrieve_weather_activity_clothing_info":
                            if retriever_added < 2:
                                for s in parse_sources_from_retriever_output(tool_out, limit=2 - retriever_added):
                                    url = s.get("url")
                                    if url and url not in sources_seen and retriever_added < 2:
                                        sources.append(s)
                                        sources_seen.add(url)
                                        retriever_added += 1

                        emit_trace(
                            "tool_end",
                            trace_id=trace_id,
                            tool=name,
                            latency_ms=tool_ms,
                            output_preview=_truncate(tool_out, 320),
                        )

                yield sse({"type": "done", "sources": sources})
                emit_trace(
                    "stream_done",
                    trace_id=trace_id,
                    status="ok",
                    latency_ms=sw_total.ms(),
                    tool_calls_count=tool_calls_count,
                    deltas_count=deltas_count,
                    delta_chars=delta_chars,
                    sources_count=len(sources),
                )
                return

            logger.warning("astream_events not available; falling back to non-stream.")
            emit_trace("stream_fallback_no_astream_events", trace_id=trace_id)

            answer, sources, _ = agent.invoke_with_sources(req.message)
            if answer:
                yield sse({"type": "delta", "value": str(answer)})
                delta_chars += len(str(answer))
                deltas_count += 1

            yield sse({"type": "done", "sources": sources})
            emit_trace(
                "stream_done",
                trace_id=trace_id,
                status="ok",
                latency_ms=sw_total.ms(),
                tool_calls_count=tool_calls_count,
                deltas_count=deltas_count,
                delta_chars=delta_chars,
                sources_count=len(sources),
                mode="fallback",
            )

        except Exception as e:
            logger.exception("Chat stream failed")
            emit_trace(
                "stream_error",
                trace_id=trace_id,
                status="error",
                latency_ms=sw_total.ms(),
                tool_calls_count=tool_calls_count,
                deltas_count=deltas_count,
                delta_chars=delta_chars,
                error=str(e),
            )
            yield sse({"type": "error", "message": str(e)})
            yield sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
