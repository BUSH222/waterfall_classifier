from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Decision:
    filename: str
    decision: int
    ground_truth: int | None
    correct: int | None


class AnnotationDB:
    def __init__(self, sqlite_path: Path) -> None:
        self._sqlite_path = sqlite_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    decision INTEGER NOT NULL,
                    ground_truth INTEGER NULL,
                    correct INTEGER NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(username, filename)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_user ON decisions(username);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_user_correct ON decisions(username, correct);")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_decision(self, *, username: str, filename: str, decision: int, ground_truth: int | None) -> None:
        correct: int | None
        if ground_truth is None:
            correct = None
        else:
            correct = 1 if int(decision) == int(ground_truth) else 0

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO decisions (username, filename, decision, ground_truth, correct)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(username, filename)
                DO UPDATE SET
                    decision=excluded.decision,
                    ground_truth=excluded.ground_truth,
                    correct=excluded.correct,
                    created_at=datetime('now');
                """,
                (username, filename, int(decision), ground_truth, correct),
            )
            self._conn.commit()

    def user_classified_filenames(self, username: str) -> set[str]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT filename FROM decisions WHERE username=?;", (username,))
            return {r["filename"] for r in cur.fetchall()}

    def user_stats(self, username: str) -> dict[str, int | float | None]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN correct=0 THEN 1 ELSE 0 END) as incorrect,
                    SUM(CASE WHEN correct IS NULL THEN 1 ELSE 0 END) as unknown
                FROM decisions
                WHERE username=?;
                """,
                (username,),
            )
            row = cur.fetchone()

        total = int(row["total"] or 0)
        correct = int(row["correct"] or 0)
        incorrect = int(row["incorrect"] or 0)
        unknown = int(row["unknown"] or 0)

        accuracy: float | None
        denom = correct + incorrect
        accuracy = (correct / denom) if denom > 0 else None

        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "unknown": unknown,
            "accuracy": accuracy,
        }

    def list_filenames_by_correctness(self, *, username: str, correct: int, limit: int = 500) -> list[Decision]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT filename, decision, ground_truth, correct
                FROM decisions
                WHERE username=? AND correct=?
                ORDER BY created_at DESC
                LIMIT ?;
                """,
                (username, int(correct), int(limit)),
            )
            rows = cur.fetchall()

        return [
            Decision(
                filename=r["filename"],
                decision=int(r["decision"]),
                ground_truth=(int(r["ground_truth"]) if r["ground_truth"] is not None else None),
                correct=(int(r["correct"]) if r["correct"] is not None else None),
            )
            for r in rows
        ]

    def refresh_ground_truth(self, *, ground_truth: dict[str, int]) -> None:
        """Backfills/refreshes ground_truth and correct for existing decisions."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT id, filename, decision FROM decisions;")
            rows = cur.fetchall()

            updates: list[tuple[int | None, int | None, int]] = []
            for r in rows:
                filename = r["filename"]
                decision = int(r["decision"])
                gt = ground_truth.get(filename)
                if gt is None:
                    updates.append((None, None, int(r["id"])))
                else:
                    corr = 1 if decision == int(gt) else 0
                    updates.append((int(gt), corr, int(r["id"])))

            cur.executemany(
                "UPDATE decisions SET ground_truth=?, correct=? WHERE id=?;",
                updates,
            )
            self._conn.commit()
