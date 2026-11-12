"""LightRAG-inspired local/global graph retrieval plus chunk rank fusion.

Uses curated entities instead of generative entity extraction and deterministic
hashed lexical vectors instead of a hosted embedding model. Scores are ranking
signals, never probabilities or evidence about an individual learner.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field
from ..db.connection import open_connection

MODES = {"local", "global", "hybrid", "naive", "mix"}
INDEX_VERSION = "curriculum-graph-rrf-v2"
STOP_WORDS = set("a an the i we to of and or is are as with for from my it this that how what why can do by have in on be".split())


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def identifier(prefix: str, value) -> str:
    return prefix + hashlib.sha256(canonical(value).encode()).hexdigest()[:24]


def tokens(text: str) -> list[str]:
    text = text.lower().replace("_", " ")
    aliases = {"added": "add", "adding": "add", "multiplying": "multiply", "multiplied": "multiply", "bottom": "denominator", "bottoms": "denominator", "top": "numerator", "tops": "numerator", "different": "unlike", "identical": "like", "same": "like", "slices": "parts"}
    result = []
    for word in re.findall(r"[a-z]+|\d+", text):
        word = aliases.get(word, word)
        if word.endswith("s") and word not in {"basics", "is", "this"}:
            word = word[:-1]
        if word not in STOP_WORDS and len(word) > 1:
            result.append(word)
    return result


def vector(text: str) -> dict[str, float]:
    words = tokens(text)
    features = words + [f"{a}:{b}" for a, b in zip(words, words[1:])]
    counts = Counter(str(int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 4096) for word in features)
    norm = math.sqrt(sum(count * count for count in counts.values())) or 1
    return {key: count / norm for key, count in sorted(counts.items())}


class RetrievalHit(BaseModel):
    citation_id: str
    document_id: str
    title: str
    text: str
    source: str
    source_revision: str
    license: str
    entity_ids: list[str]
    score: float
    channels: list[str]


class RetrievalResult(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"]
    namespace: str
    index_version: str = INDEX_VERSION
    index_fingerprint: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    relationships: list[dict] = Field(default_factory=list)
    context: str = ""
    abstained: bool = False
    truncated: bool = False
    ranking_note: str = "Reciprocal rank fusion of deterministic lexical, hashed-vector, and graph channels; scores are not confidence probabilities."


class KnowledgeStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS rag_namespaces (namespace TEXT PRIMARY KEY, index_version TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rag_documents (
                    document_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
                    namespace TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL,
                    entity_ids_json TEXT NOT NULL, vector_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_fts_map (
                    chunk_id TEXT PRIMARY KEY REFERENCES rag_chunks(chunk_id) ON DELETE CASCADE,
                    fts_rowid INTEGER NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS rag_entities (
                    namespace TEXT NOT NULL, entity_id TEXT NOT NULL, document_id TEXT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
                    name TEXT NOT NULL, kind TEXT NOT NULL, PRIMARY KEY(namespace, entity_id, document_id)
                );
                CREATE TABLE IF NOT EXISTS rag_edges (
                    namespace TEXT NOT NULL, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
                    relation TEXT NOT NULL, document_id TEXT NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
                    weight REAL NOT NULL, PRIMARY KEY(namespace, source_id, target_id, relation, document_id)
                );
                CREATE INDEX IF NOT EXISTS rag_chunk_document ON rag_chunks(document_id);
                CREATE INDEX IF NOT EXISTS rag_entity_document ON rag_entities(document_id);
                CREATE INDEX IF NOT EXISTS rag_edge_document ON rag_edges(document_id);
                CREATE INDEX IF NOT EXISTS rag_chunk_namespace ON rag_chunks(namespace);
                CREATE INDEX IF NOT EXISTS rag_edge_source ON rag_edges(namespace,source_id);
                CREATE INDEX IF NOT EXISTS rag_edge_target ON rag_edges(namespace,target_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(chunk_id UNINDEXED, namespace UNINDEXED, content, tokenize='unicode61');
            """)
            conn.execute("INSERT OR IGNORE INTO rag_fts_map SELECT chunk_id,rowid FROM rag_fts")

    @contextmanager
    def connection(self):
        conn = open_connection(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def sync(self, documents: list[dict], namespace: str = "curriculum") -> dict:
        """Atomically synchronize a source namespace, including stale entity/edge deletion."""
        expected = {doc["document_id"] for doc in documents}
        if len(expected) != len(documents):
            raise ValueError("Duplicate document IDs")
        changed = removed = 0
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = {row["document_id"]: row["fingerprint"] for row in conn.execute("SELECT document_id,fingerprint FROM rag_documents WHERE namespace=?", (namespace,))}
            for doc_id in sorted(set(existing) - expected):
                conn.execute("DELETE FROM rag_fts WHERE rowid IN (SELECT f.fts_rowid FROM rag_fts_map f JOIN rag_chunks c USING(chunk_id) WHERE c.document_id=?)", (doc_id,))
                conn.execute("DELETE FROM rag_documents WHERE document_id=? AND namespace=?", (doc_id, namespace))
                removed += 1
            for doc in sorted(documents, key=lambda item: item["document_id"]):
                doc_id = doc["document_id"]
                fingerprint = identifier("D_", [INDEX_VERSION, doc])
                if existing.get(doc_id) == fingerprint:
                    continue
                owner = conn.execute("SELECT namespace FROM rag_documents WHERE document_id=?", (doc_id,)).fetchone()
                if owner and owner[0] != namespace:
                    raise ValueError("Document ID already belongs to another namespace")
                if owner:
                    conn.execute("DELETE FROM rag_fts WHERE rowid IN (SELECT f.fts_rowid FROM rag_fts_map f JOIN rag_chunks c USING(chunk_id) WHERE c.document_id=?)", (doc_id,))
                    conn.execute("DELETE FROM rag_documents WHERE document_id=?", (doc_id,))
                conn.execute("INSERT INTO rag_documents VALUES(?,?,?,?)", (doc_id, namespace, fingerprint, canonical(doc)))
                for entity in doc.get("entities", []):
                    conn.execute("INSERT INTO rag_entities VALUES(?,?,?,?,?)", (namespace, entity["id"], doc_id, entity["name"], entity["kind"]))
                for edge in doc.get("edges", []):
                    conn.execute("INSERT INTO rag_edges VALUES(?,?,?,?,?,?)", (namespace, edge["source"], edge["target"], edge["relation"], doc_id, edge.get("weight", 1.0)))
                # Paragraph chunks preserve source boundaries; overlapping words only for long paragraphs.
                for position, text in enumerate(self._chunks(doc["text"])):
                    chunk_id = identifier("C_", [doc_id, fingerprint, position, text])
                    entity_ids = sorted(entity["id"] for entity in doc.get("entities", []))
                    content = (doc["title"] + " ") * 3 + text
                    conn.execute("INSERT INTO rag_chunks VALUES(?,?,?,?,?,?,?)", (chunk_id, doc_id, namespace, doc["title"], text, canonical(entity_ids), canonical(vector(content))))
                    fts_rowid = conn.execute("INSERT INTO rag_fts VALUES(?,?,?)", (chunk_id, namespace, " ".join(tokens(content)))).lastrowid
                    conn.execute("INSERT INTO rag_fts_map VALUES(?,?)", (chunk_id,fts_rowid))
                changed += 1
            conn.execute("INSERT INTO rag_namespaces VALUES(?,?) ON CONFLICT(namespace) DO UPDATE SET index_version=excluded.index_version", (namespace,INDEX_VERSION))
        return {"updated_documents": changed, "deleted_documents": removed, **self.status(namespace)}

    @staticmethod
    def _chunks(text: str, size: int = 180, overlap: int = 24):
        for paragraph in re.split(r"\n\s*\n", text.strip()):
            words = paragraph.split()
            for start in range(0, len(words), size - overlap):
                yield " ".join(words[start:start + size])
                if start + size >= len(words):
                    break

    def status(self, namespace: str = "curriculum") -> dict:
        with self.connection() as conn:
            rows = list(conn.execute("SELECT document_id,fingerprint FROM rag_documents WHERE namespace=? ORDER BY document_id", (namespace,)))
            counts = {name: conn.execute(f"SELECT COUNT(*) FROM rag_{name} WHERE namespace=?", (namespace,)).fetchone()[0] for name in ("chunks", "entities", "edges")}
            unique_entities = conn.execute("SELECT COUNT(DISTINCT entity_id) FROM rag_entities WHERE namespace=?", (namespace,)).fetchone()[0]
        return {"namespace": namespace, "documents": len(rows), "unique_entities": unique_entities, **counts, "index_version": INDEX_VERSION, "index_fingerprint": identifier("INDEX_", [tuple(row) for row in rows])}

    def query(self, query: str, mode: str = "mix", namespace: str = "curriculum", top_k: int = 6, max_context_chars: int = 6000, concept_ids: list[str] | None = None) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError("Unsupported retrieval mode")
        if not 1 <= top_k <= 20 or not 128 <= max_context_chars <= 20000:
            raise ValueError("Invalid retrieval budget")
        status = self.status(namespace)
        result = RetrievalResult(query=query, mode=mode, namespace=namespace, index_fingerprint=status["index_fingerprint"])
        query_tokens = tokens(query)
        hints = set(concept_ids or [])
        if not query_tokens and not hints:
            result.abstained = True
            return result
        with self.connection() as conn:
            conn.execute("BEGIN")
            manifest = [tuple(row) for row in conn.execute("SELECT document_id,fingerprint FROM rag_documents WHERE namespace=? ORDER BY document_id", (namespace,))]
            result.index_fingerprint = identifier("INDEX_", manifest)
            version = conn.execute("SELECT index_version FROM rag_namespaces WHERE namespace=?", (namespace,)).fetchone()
            if manifest and (version is None or version[0] != INDEX_VERSION):
                raise ValueError("Source collection needs reindexing with the current retrieval version")
            chunks = {row["chunk_id"]: dict(row) for row in conn.execute("SELECT c.*,d.payload_json FROM rag_chunks c JOIN rag_documents d USING(document_id) WHERE c.namespace=? ORDER BY c.chunk_id", (namespace,))}
            if not chunks:
                result.abstained = True
                return result
            lists: dict[str, list[str]] = {}
            expression = " OR ".join('"' + word + '"' for word in sorted(set(query_tokens))[:48])
            lexical = []
            if expression:
                lexical = [row["chunk_id"] for row in conn.execute("SELECT chunk_id,bm25(rag_fts) AS rank FROM rag_fts WHERE rag_fts MATCH ? AND namespace=? ORDER BY rank,chunk_id LIMIT 80", (expression, namespace))]
            qvec = vector(query)
            vector_scores = []
            for chunk_id, chunk in chunks.items():
                # Hash collisions alone cannot make an unrelated source relevant.
                if not set(query_tokens).intersection(tokens(chunk["title"] + " " + chunk["text"])):
                    continue
                stored = json.loads(chunk["vector_json"])
                score = sum(value * stored.get(key, 0) for key, value in qvec.items())
                if score > 0:
                    vector_scores.append((chunk_id, score))
            vector_rank = [cid for cid, score in sorted(vector_scores, key=lambda entry: (-entry[1], entry[0]))[:80]]
            entities = list(conn.execute("SELECT DISTINCT entity_id,name FROM rag_entities WHERE namespace=? ORDER BY entity_id", (namespace,)))
            local_entities = set(hints)
            for entity in entities:
                name_tokens = set(tokens(entity["name"] + " " + entity["entity_id"]))
                overlap = set(query_tokens) & name_tokens
                if len(overlap) >= min(2, len(name_tokens)) and overlap:
                    local_entities.add(entity["entity_id"])
            # Seed graph from the most relevant chunks as well as explicit concept hints.
            for cid in vector_rank[:2]:
                local_entities.update(json.loads(chunks[cid]["entity_ids_json"]))
            all_edges = [dict(row) for row in conn.execute("SELECT DISTINCT source_id,target_id,relation,weight FROM rag_edges WHERE namespace=? ORDER BY source_id,target_id,relation,weight", (namespace,))]
            frontier = set(local_entities)
            global_entities = set()
            selected_edges = []
            for _ in range(2):
                next_frontier = set()
                for edge in all_edges:
                    if edge["source_id"] in frontier or edge["target_id"] in frontier:
                        selected_edges.append(edge)
                        next_frontier.update((edge["source_id"], edge["target_id"]))
                next_frontier -= local_entities | global_entities
                global_entities.update(next_frontier)
                frontier = next_frontier
            def graph_rank(entity_ids):
                scores = [(cid, len(set(json.loads(chunk["entity_ids_json"])) & entity_ids)) for cid, chunk in chunks.items()]
                query_scores = dict(vector_scores)
                return [cid for cid, score in sorted(scores, key=lambda entry: (-entry[1], -query_scores.get(entry[0], 0), entry[0])) if score > 0][:80]
            if mode in {"naive", "mix"}:
                lists["bm25"] = lexical
                lists["hashed_vector"] = vector_rank
            if mode in {"local", "hybrid", "mix"}:
                lists["local_entity"] = graph_rank(local_entities)
            if mode in {"global", "hybrid", "mix"}:
                lists["global_relationship"] = graph_rank(global_entities | local_entities)
            # Graph support augments precision; direct lexical ranks break graph ties.
            if mode in {"local", "global", "hybrid"} and (local_entities or global_entities):
                lists["query_rerank"] = vector_rank
            fused = defaultdict(float)
            channels = defaultdict(list)
            for channel, ranked in lists.items():
                for rank, cid in enumerate(ranked, 1):
                    weight = {"bm25": 2.0, "hashed_vector": 2.0, "local_entity": 0.35, "global_relationship": 0.2, "query_rerank": 2.0}[channel]
                    fused[cid] += weight / (60 + rank)
                    channels[cid].append(channel)
            remaining = max_context_chars
            for cid in sorted(fused, key=lambda key: (-fused[key], key))[:top_k]:
                chunk = chunks[cid]
                doc = json.loads(chunk["payload_json"])
                citation_id = cid
                rendered = f"[{citation_id}] {chunk['title']}\n{chunk['text']}\nSource: {doc['source']}\n"
                if len(rendered) > remaining:
                    result.truncated = True
                    continue
                remaining -= len(rendered)
                result.hits.append(RetrievalHit(citation_id=citation_id, document_id=chunk["document_id"], title=chunk["title"], text=chunk["text"], source=doc["source"], source_revision=doc["revision"], license=doc["license"], entity_ids=json.loads(chunk["entity_ids_json"]), score=round(fused[cid], 8), channels=channels[cid]))
                result.context += rendered
            hit_entities = {eid for hit in result.hits for eid in hit.entity_ids}
            unique_edges = {canonical(edge): edge for edge in selected_edges if edge["source_id"] in hit_entities and edge["target_id"] in hit_entities}
            result.relationships = [unique_edges[key] for key in sorted(unique_edges)][:60]
            result.abstained = not result.hits
        return result
