# src/api/routes/chat.py

import re
import time
import asyncio
from functools import lru_cache
from typing import List

from fastapi import HTTPException

from langchain_core.messages import HumanMessage, AIMessage

from src.api.routes.base_route import router
from src.api.routes.module.schema import ChatRequest, ChatResponse, SourceItem, LatencyMs, Tokens

# IMPORTANT: عدّل الاستيراد ده حسب مكان الـ Agent عندك فعليًا
# لو عندك: src/agent/weather_agent.py وفيه WeatherActivityClothingAgent
from src.agent.weather_agent import WeatherActivityClothingAgent


@lru_cache(maxsize=1)
def get_agent() -> WeatherActivityClothingAgent:
    # singleton (عشان الـ init تقيل)
    return WeatherActivityClothingAgent()


_URL_RE = re.compile(r"(https?://[^\s)]+)")


def _build_langchain_messages(req: ChatRequest):
    msgs = []
    for h in req.history or []:
        if h.role == "user":
            msgs.append(HumanMessage(content=h.content))
        else:
            msgs.append(AIMessage(content=h.content))
    msgs.append(HumanMessage(content=req.message))
    return msgs


def _extract_sources_from_tool_outputs(messages) -> List[SourceItem]:
    # best-effort: استخراج أي URLs من ToolMessage outputs (أو أي message content)
    sources = []
    seen = set()
    for m in messages or []:
        content = getattr(m, "content", "") or ""
        for url in _URL_RE.findall(content):
            if url in seen:
                continue
            seen.add(url)
            sources.append(SourceItem(name="source", url=url))
    return sources


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    by_step = {}

    try:
        agent = get_agent()

        msgs = _build_langchain_messages(req)
        payload = {"messages": msgs}

        t1 = time.perf_counter()
        if hasattr(agent.app, "ainvoke"):
            out = await agent.app.ainvoke(payload)
        else:
            out = await asyncio.to_thread(agent.app.invoke, payload)
        by_step["agent_invoke"] = int((time.perf_counter() - t1) * 1000)

        messages = out.get("messages", []) if isinstance(out, dict) else []
        answer = ""
        for m in reversed(messages):
            if getattr(m, "type", "") == "ai":
                answer = getattr(m, "content", "") or ""
                break

        sources = _extract_sources_from_tool_outputs(messages)

        total_ms = int((time.perf_counter() - t0) * 1000)

        return ChatResponse(
            answer=answer,
            sources=sources,
            latency_ms=LatencyMs(total=total_ms, by_step=by_step),
            tokens=Tokens(prompt=0, completion=0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
