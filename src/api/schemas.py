from __future__ import annotations

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")

    class Config:
        extra = "ignore"


class AgentResponse(BaseModel):
    answer: str


class Source(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)


class LatencyBreakdown(BaseModel):
    retrieve: int = Field(ge=0)
    llm: int = Field(ge=0)


class Latency(BaseModel):
    total: int = Field(ge=0)
    by_step: LatencyBreakdown


class Tokens(BaseModel):
    prompt: int = Field(ge=0)
    completion: int = Field(ge=0)


class QAResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: List[Source] = Field(default_factory=list)
    latency_ms: Latency
    tokens: Tokens

    @validator("sources", pre=True)
    def _coerce_sources(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, list):
            # ensure each entry is dict with name/url
            coerced = []
            for item in v:
                if isinstance(item, dict):
                    coerced.append(
                        {
                            "name": item.get("name") or item.get("title") or item.get("url") or "",
                            "url": item.get("url") or "",
                        }
                    )
                else:
                    coerced.append({"name": str(item), "url": str(item)})
            return coerced
        return v
