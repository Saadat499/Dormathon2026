"""SQLite storage (standard library only). One process, one lock."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

STUDY_COLUMNS = (
    "study_id", "seq", "uploaded_at", "filename", "has_image", "findings", "confidence_band",
    "ood_flag", "ood_reason", "model_source", "proc_state", "clinician_urgent",
    "override_priority", "reviewed_at", "flagged_disagreement", "has_heatmap",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY,
  seq INTEGER NOT NULL,
  uploaded_at TEXT NOT NULL,
  filename TEXT,
  has_image INTEGER NOT NULL DEFAULT 0,
  findings TEXT NOT NULL DEFAULT '[]',
  confidence_band TEXT,
  ood_flag INTEGER NOT NULL DEFAULT 0,
  ood_reason TEXT,
  model_source TEXT NOT NULL DEFAULT 'none',
  proc_state TEXT NOT NULL DEFAULT 'processing',
  clinician_urgent INTEGER NOT NULL DEFAULT 0,
  override_priority TEXT,
  reviewed_at TEXT,
  flagged_disagreement INTEGER NOT NULL DEFAULT 0,
  has_heatmap INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  study_id TEXT,
  detail TEXT
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.tx() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        with self.lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # settings -------------------------------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.lock:
            row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value)))

    # audit ----------------------------------------------------------------------------------
    def add_audit(self, actor: str, action: str, study_id: Optional[str], detail: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO audit(timestamp,actor,action,study_id,detail) VALUES(?,?,?,?,?)",
                      (iso(now_utc()), actor, action, study_id, detail))

    def list_audit(self, action: Optional[str] = None, q: Optional[str] = None, limit: int = 200) -> list[dict]:
        sql, args = "SELECT id,timestamp,actor,action,study_id,detail FROM audit WHERE 1=1", []
        if action:
            sql += " AND action=?"
            args.append(action)
        if q:
            like = f"%{q.lower()}%"
            sql += " AND (lower(coalesce(study_id,'')) LIKE ? OR lower(actor) LIKE ? OR lower(coalesce(detail,'')) LIKE ?)"
            args += [like, like, like]
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # studies --------------------------------------------------------------------------------
    def count_studies(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM studies").fetchone()[0]

    def next_seq(self) -> int:
        with self.lock:
            return (self.conn.execute("SELECT COALESCE(MAX(seq),0) FROM studies").fetchone()[0]) + 1

    def insert_study(self, row: dict) -> None:
        cols = [c for c in STUDY_COLUMNS if c in row]
        with self.tx() as c:
            c.execute(f"INSERT INTO studies({','.join(cols)}) VALUES({','.join('?' * len(cols))})", [row[k] for k in cols])

    def update_study(self, study_id: str, **fields: Any) -> None:
        bad = set(fields) - set(STUDY_COLUMNS)
        if bad:
            raise ValueError(f"unknown study columns: {bad}")
        sets = ",".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE studies SET {sets} WHERE study_id=?", [*fields.values(), study_id])

    def get_study(self, study_id: str) -> Optional[dict]:
        with self.lock:
            r = self.conn.execute("SELECT * FROM studies WHERE study_id=?", (study_id,)).fetchone()
        return dict(r) if r else None

    def all_studies(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM studies").fetchall()]
