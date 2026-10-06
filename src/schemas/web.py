"""Structured search planning and discovered page references."""

from pydantic import Field, HttpUrl

from src.schemas.common import Contract, NonBlank


class SearchPlan(Contract):
    queries: list[NonBlank] = Field(min_length=1)


class SearchHit(Contract):
    url: HttpUrl
    title: NonBlank
