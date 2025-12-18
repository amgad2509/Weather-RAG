# # src/api/routes/chat.py
# from __future__ import annotations

# import logging
# from fastapi import APIRouter, HTTPException, Request

# from src.api.routes.module.schema import AgentRequest, AgentResponse
# from src.agent.weather_agent import WeatherActivityClothingAgent

# logger = logging.getLogger(__name__)
# ai_agent_router = APIRouter(prefix="/chat", tags=["chat"])


# def get_agent(request: Request):
#     agent = getattr(request.app.state, "weather_agent", None)
#     if agent is None:
#         request.app.state.weather_agent = WeatherActivityClothingAgent()
#         agent = request.app.state.weather_agent
#     return agent


# @ai_agent_router.post("", response_model=AgentResponse)
# def chat(req: AgentRequest, request: Request) -> AgentResponse:
#     try:
#         agent = get_agent(request)
#         answer = agent.invoke(req.message)

#         if not isinstance(answer, str) or not answer.strip():
#             raise HTTPException(status_code=500, detail="Agent returned empty answer")

#         return AgentResponse(answer=answer)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Chat endpoint failed")
#         raise HTTPException(status_code=500, detail=str(e))

# src/api/routes/chat.py
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage

from src.api.routes.module.schema import AgentRequest, AgentResponse
from src.agent.weather_agent import WeatherActivityClothingAgent

logger = logging.getLogger(__name__)
ai_agent_router = APIRouter(prefix="/chat", tags=["chat"])


def get_agent(request: Request) -> WeatherActivityClothingAgent:
    agent = getattr(request.app.state, "weather_agent", None)
    if agent is None:
        request.app.state.weather_agent = WeatherActivityClothingAgent()
        agent = request.app.state.weather_agent
    return agent


def _sse(data: dict) -> str:
    # Server-Sent Events (SSE): each message is "data: <json>\n\n"
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@ai_agent_router.post("", response_model=AgentResponse)
def chat(req: AgentRequest, request: Request) -> AgentResponse:
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

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"type": "status", "value": "started"})

            # Preferred: true token streaming from graph
            if hasattr(agent.app, "astream_events"):
                async for ev in agent.app.astream_events(
                    {"messages": [HumanMessage(content=req.message)]},
                    version="v2",
                ):
                    # Stop if client disconnected
                    if await request.is_disconnected():
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
                            yield _sse({"type": "delta", "value": piece})

                    elif et == "on_tool_start":
                        # log only (do not stream tool info to client)
                        name = ev.get("name", "tool")
                        tool_in = data.get("input")
                        logger.info("TOOL_CALL -> %s | args=%s", name, tool_in)

                    elif et == "on_tool_end":
                        name = ev.get("name", "tool")
                        tool_out = data.get("output")
                        logger.info("TOOL_RESULT <- %s | output=%s", name, tool_out)

                yield _sse({"type": "done"})
                return

            # Fallback: no events support (stream as one delta)
            logger.warning("astream_events not available; falling back to non-stream.")
            answer = agent.invoke(req.message)
            if answer:
                yield _sse({"type": "delta", "value": str(answer)})
            yield _sse({"type": "done"})

        except Exception as e:
            logger.exception("Chat stream failed")
            yield _sse({"type": "error", "message": str(e)})
            yield _sse({"type": "done"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
