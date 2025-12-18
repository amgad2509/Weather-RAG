from pydantic import BaseModel, Field


class InternetSearchInput(BaseModel):
    """Input schema for DuckDuckGo internet search tool."""
    query: str = Field(..., description="Search query text", min_length=1)
    max_related: int = Field(
        default=6,
        ge=0,
        le=20,
        description="Max number of related topics/links to include (0-20).",
    )
