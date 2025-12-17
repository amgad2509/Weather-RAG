# src/api/routes/module/schema.py

from typing import Dict, List, Literal
from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"] = Field(...)
    content: str = Field(...)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[ChatHistoryItem] = Field(default_factory=list)


class SourceItem(BaseModel):
    name: str
    url: str


class LatencyMs(BaseModel):
    total: int = 0
    by_step: Dict[str, int] = Field(default_factory=dict)


class Tokens(BaseModel):
    prompt: int = 0
    completion: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    latency_ms: LatencyMs = Field(default_factory=LatencyMs)
    tokens: Tokens = Field(default_factory=Tokens)
