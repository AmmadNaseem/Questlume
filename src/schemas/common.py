"""Shared schema conventions."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Contract(BaseModel):
    """Reject unknown fields and keep raw inputs out of printed errors."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
