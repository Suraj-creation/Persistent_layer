"""Keep API tests offline and away from the user's research database."""
import os
import tempfile
from pathlib import Path

import pytest

# Test modules import main during collection, before fixtures can monkeypatch it.
# Keep that initial schema setup away from the user's database as well.
_bootstrap = tempfile.TemporaryDirectory(prefix="learner-pytest-bootstrap-")
os.environ["LEARNER_STATE_DB_PATH"] = str(Path(_bootstrap.name) / "bootstrap.db")
os.environ["LEARNER_EXTRACTOR"] = "rules"
os.environ["EVIDENCE_EXTRACTOR_MODE"] = "rules"
os.environ["TUTOR_RESPONSE_MODE"] = "deterministic"


@pytest.fixture(autouse=True)
def isolated_api(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNER_STATE_DB_PATH", str(tmp_path / "bootstrap.db"))
    monkeypatch.setenv("LEARNER_DB_PATH", str(tmp_path / "bootstrap.db"))
    monkeypatch.setenv("LEARNER_EXTRACTOR", "rules")
    from apps.api.app import main
    from packages.learner_state_core.db import LocalDatabaseManager

    monkeypatch.setattr(main, "db_manager", LocalDatabaseManager(str(tmp_path / "test.db")))
    monkeypatch.setattr(main.instructor_extractor, "client", None)
    monkeypatch.setattr(main, "azure_client", None)
