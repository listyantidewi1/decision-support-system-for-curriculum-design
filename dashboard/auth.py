"""
Authentication helpers for dashboard pages.
"""

from typing import Optional

from fastapi import HTTPException, Request


def current_user(request: Request) -> Optional[dict]:
    return request.session.get("user")


def require_user(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return user


def require_role(request: Request, role: str) -> dict:
    user = require_user(request)
    if user.get("role") != role:
        raise HTTPException(status_code=403, detail="Forbidden")
    return user

