"""SQLite connection setup shared by learner and knowledge storage."""
# Defer evaluation of type annotations.
from __future__ import annotations

# Use the standard SQLite driver for both database users.
import sqlite3
# Use a monotonic clock for the lock retry deadline.
import time


# Open and configure a SQLite connection for the platform's local database.
def open_connection(path: str) -> sqlite3.Connection:
    # Give busy database operations time to finish rather than failing immediately.
    connection = sqlite3.connect(path, timeout=30.0)
    # Return rows whose columns can be accessed by name.
    connection.row_factory = sqlite3.Row
    # Set a 30-second limit for retries while setting journal mode.
    deadline = time.monotonic() + 30
    # Close the connection if setup fails before returning it.
    try:
        # Enforce references between related rows in this connection.
        connection.execute("PRAGMA foreign_keys=ON")
        # SQLite's busy timeout does not reliably retry journal-mode transitions
        # while several workers open a brand-new database simultaneously.
        # Keep checking when another connection temporarily locks journal-mode setup.
        while True:
            # Catch busy errors around only the setup operations that may need a retry.
            try:
                # Find the journal mode already active on this database.
                mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                # Change modes only when write-ahead logging is not already enabled.
                if mode != "wal":
                    # Enable WAL so readers can continue while a writer works.
                    connection.execute("PRAGMA journal_mode=WAL")
                # Return the configured connection to its caller.
                return connection
            # Handle known transient SQLite operational errors.
            except sqlite3.OperationalError as error:
                # Give up on unrelated errors or when the setup deadline expires.
                if not any(word in str(error).lower() for word in ("locked", "busy")) or time.monotonic() >= deadline:
                    # Propagate the original problem.
                    raise
                # Briefly yield before the next attempt.
                time.sleep(0.01)
    # Clean up any failed connection setup.
    except BaseException:
        # Release the SQLite connection.
        connection.close()
        # Preserve the original error for the caller.
        raise
