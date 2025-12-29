from __future__ import annotations

from fastapi import HTTPException, Request
from starlette import status


SESSION_USER_KEY = "wf_user"


def authenticate(users: dict[str, str], username: str, password: str) -> bool:
    expected = users.get(username)
    if expected is None:
        return False
    return password == expected


def get_current_user(request: Request) -> str | None:
    return request.session.get(SESSION_USER_KEY)


def require_user_for_api(request: Request) -> str:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user
