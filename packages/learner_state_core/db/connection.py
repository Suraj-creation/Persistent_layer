"""SQLite connection setup shared by learner and knowledge storage."""
from __future__ import annotations

import sqlite3
import time


def open_connection(path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    deadline = time.monotonic() + 30
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        # SQLite's busy timeout does not reliably retry journal-mode transitions
        # while several workers open a brand-new database simultaneously.
        while True:
            try:
                mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                if mode != "wal":
                    connection.execute("PRAGMA journal_mode=WAL")
                return connection
            except sqlite3.OperationalError as error:
                if not any(word in str(error).lower() for word in ("locked", "busy")) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    except BaseException:
        connection.close()
        raise
