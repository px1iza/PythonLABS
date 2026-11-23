from typing import Any

from fastapi import HTTPException, Request


def get_index(request: Request) -> Any:
    index = getattr(request.app.state, "index", None)

    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Search index is not loaded",
        )

    return index
