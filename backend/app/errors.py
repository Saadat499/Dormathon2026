"""One error type for everything the API refuses. Always rendered as {"detail": "message"}."""
from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail
