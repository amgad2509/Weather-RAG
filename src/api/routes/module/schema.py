# src/api/routes/module/schema.py
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")

    class Config:
        extra = "ignore"


class AgentResponse(BaseModel):
    answer: str
