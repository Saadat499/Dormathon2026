"""Bounded reading of uploaded files."""
from __future__ import annotations

from fastapi import UploadFile

from .errors import ApiError


async def read_upload(f: UploadFile, limit: int) -> bytes:
    data = await f.read(limit + 1)
    if len(data) > limit:
        raise ApiError(413, f"The file is larger than {limit // (1024 * 1024)} MB.")
    return data
