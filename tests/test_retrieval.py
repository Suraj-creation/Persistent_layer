import copy
import sqlite3

import pytest

from packages.learner_state_core.retrieval import KnowledgeStore
from packages.learner_state_core.retrieval.corpus import curriculum_documents


@pytest.fixture
def store(tmp_path):
    store = KnowledgeStore(str(tmp_path / "knowledge.db"))
    store.sync(curriculum_documents())
    return store


@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "naive", "mix"])
def test_modes_are_deterministic_bounded_and_cited_after_restart(store, mode):
    first = store.query("common denominator for unlike fractions", mode=mode)
    second = KnowledgeStore(store.db_path).query("common denominator for unlike fractions", mode=mode)
    assert first.model_dump() == second.model_dump()
    assert first.hits
    assert len(first.context) <= 6000
    assert all(hit.citation_id in first.context and hit.source and hit.source_revision for hit in first.hits)
    assert len({hit.citation_id for hit in first.hits}) == len(first.hits)
    if mode != "naive":
        assert first.relationships


def test_graph_expansion_finds_prerequisites(store):
    result = store.query("", mode="global", concept_ids=["fraction_addition_unlike_denominators"], top_k=10)
    assert any(edge["relation"] == "requires" and edge["target_id"] == "fraction_equivalence" for edge in result.relationships)
    assert any("fraction_equivalence" in hit.entity_ids for hit in result.hits)


@pytest.mark.parametrize("query", ["", "the and of", "quantum quasar astrophysics", '" OR * ; DROP TABLE rag_chunks; --'])
def test_unsupported_or_adversarial_queries_abstain_safely(store, query):
    result = store.query(query)
    assert result.abstained
    assert not result.hits
    assert store.status()["documents"] == 10


def test_context_budget_and_max_results_are_enforced(store):
    result = store.query("fractions common denominator", max_context_chars=128, top_k=1)
    assert len(result.context) <= 128
    assert result.truncated
    assert len(result.hits) <= 1


def test_incremental_reindex_and_deletion_remove_stale_fts_and_graph(store):
    docs = curriculum_documents()
    original = store.status()
    assert store.sync(docs)["updated_documents"] == 0
    assert store.status() == original
    changed = copy.deepcopy(docs)
    changed[0]["text"] += " Unique replacementword provenance."
    update = store.sync(changed)
    assert update["updated_documents"] == 1
    assert update["index_fingerprint"] != original["index_fingerprint"]
    assert store.query("replacementword").hits
    store.sync([])
    assert store.query("replacementword").abstained
    with store.connection() as conn:
        for table in ("rag_entities", "rag_edges", "rag_chunks", "rag_fts"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_namespace_isolation_and_failed_sync_rollback(store):
    other = copy.deepcopy(curriculum_documents()[:1])
    other[0]["document_id"] = "external:isolated"
    other[0]["text"] = "Externalonlymagic phrase"
    store.sync(other, "external")
    assert store.query("Externalonlymagic").abstained
    assert store.query("Externalonlymagic", namespace="external").hits
    original = store.status()
    invalid = copy.deepcopy(curriculum_documents())
    invalid[0]["text"] = "This edit should rollback"
    invalid[1]["entities"][0].pop("id")
    with pytest.raises(KeyError):
        store.sync(invalid)
    assert store.status() == original
    assert store.query("rollback").abstained


def test_shared_document_id_cannot_erase_another_namespace(store):
    docs = curriculum_documents()
    with pytest.raises(ValueError, match="another namespace"):
        store.sync(docs, "external")
    assert store.status()["documents"] == 10


def test_outdated_index_requires_reindex_and_recovers(store):
    with store.connection() as conn:
        conn.execute("UPDATE rag_namespaces SET index_version='old-version'")
    with pytest.raises(ValueError, match="reindexing"):
        store.query("common denominator")
    store.sync(curriculum_documents())
    assert store.query("common denominator").hits
