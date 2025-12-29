from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    images_dir: Path
    labels_file: Path | None
    sqlite_path: Path
    secret_key: str
    users: dict[str, str]


def _parse_users(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}

    users: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("WF_USERS must be comma-separated username:password entries")
        username, password = part.split(":", 1)
        username = username.strip()
        password = password.strip()
        if not username:
            raise ValueError("WF_USERS contains an empty username")
        users[username] = password

    return users


def load_settings() -> Settings:
    load_dotenv()
    repo_root = Path(__file__).resolve().parents[1]

    # Dataset root directory. Expected layout:
    #   <WF_IMAGES_DIR>/with_signal/*.png
    #   <WF_IMAGES_DIR>/without_signal/*.png
    images_dir = Path(os.getenv("WF_IMAGES_DIR", str(repo_root / "data"))).expanduser().resolve()

    labels_file_raw = os.getenv("WF_LABELS_FILE", "")
    labels_file = Path(labels_file_raw).expanduser().resolve() if labels_file_raw else None

    sqlite_path = Path(os.getenv("WF_DB_PATH", str(repo_root / "data" / "annotations.sqlite3"))).expanduser().resolve()

    secret_key = os.getenv("WF_SECRET_KEY", "")
    if not secret_key:
        # Rudimentary default; recommended to set WF_SECRET_KEY in production.
        secret_key = "dev-secret-change-me"

    users = _parse_users(os.getenv("WF_USERS"))

    return Settings(
        images_dir=images_dir,
        labels_file=labels_file,
        sqlite_path=sqlite_path,
        secret_key=secret_key,
        users=users,
    )
