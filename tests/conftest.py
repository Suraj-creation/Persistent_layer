"""Keep API tests offline and away from the user's research database."""
import pytest


@pytest.fixture(autouse=True)
def isolated_api(tmp_path, monkeypatch):
    from apps.api.app import main
    from packages.learner_state_core.db import LocalDatabaseManager

    monkeypatch.setattr(main, "db_manager", LocalDatabaseManager(str(tmp_path / "test.db")))
    monkeypatch.setattr(main.instructor_extractor, "client", None)
    monkeypatch.setattr(main, "azure_client", None)
