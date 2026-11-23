"""LightRAG-inspired local/global graph retrieval plus chunk rank fusion.

Uses curated entities instead of generative entity extraction and deterministic
hashed lexical vectors instead of a hosted embedding model. Scores are ranking
signals, never probabilities or evidence about an individual learner.
"""
# Defer annotation evaluation for type hints.
from __future__ import annotations

# Count lexical features and collect scores by chunk.
from collections import Counter, defaultdict
# Manage database lifetime around each index operation.
from contextlib import contextmanager
# Generate stable document, chunk and vector identifiers.
import hashlib
# Encode the graph, source records and vectors for SQLite.
import json
# Normalize the vector to unit length.
import math
# Ensure the database directory exists before opening it.
from pathlib import Path
# Split text into searchable terms and paragraphs.
import re
# Use SQLite along with its full-text extension.
import sqlite3
# Restrict requested retrieval modes to known values.
from typing import Literal

# Validate and serialize results for the API.
from pydantic import BaseModel, Field
# Use the same SQLite configuration as the rest of the platform.
from ..db.connection import open_connection

# Enumerate the five ranking modes exposed by the research workspace.
MODES = {"local", "global", "hybrid", "naive", "mix"}
# Track the scoring and data layout version so old indexes can be detected.
INDEX_VERSION = "curriculum-graph-rrf-v2"
# Exclude high-frequency, low-information words from research token matching.
STOP_WORDS = set("a an the i we to of and or is are as with for from my it this that how what why can do by have in on be".split())


# Serialize a value consistently so source and index IDs are reproducible.
def canonical(value) -> str:
    # Sort keys, remove formatting spaces and keep non-ASCII text.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


# Build a stable ID from the source content or retrieval manifest.
def identifier(prefix: str, value) -> str:
    # Prefix a shortened digest so the ID also indicates the record type.
    return prefix + hashlib.sha256(canonical(value).encode()).hexdigest()[:24]


# Convert input text into comparable research search terms.
def tokens(text: str) -> list[str]:
    # Lowercase and split underscores in identifiers such as concept IDs.
    text = text.lower().replace("_", " ")
    # Map common wording and inflections to the curriculum's vocabulary.
    aliases = {"added": "add", "adding": "add", "multiplying": "multiply", "multiplied": "multiply", "bottom": "denominator", "bottoms": "denominator", "top": "numerator", "tops": "numerator", "different": "unlike", "identical": "like", "same": "like", "slices": "parts"}
    # Collect the retained lexical terms in their original order.
    result = []
    # Read ASCII words and digit sequences from the text.
    for word in re.findall(r"[a-z]+|\d+", text):
        # Replace each recognized alias with its canonical term.
        word = aliases.get(word, word)
        # Apply a simple plural-suffix rule with common exceptions.
        if word.endswith("s") and word not in {"basics", "is", "this"}:
            # Remove the trailing plural marker.
            word = word[:-1]
        # Skip stop words and one-character terms.
        if word not in STOP_WORDS and len(word) > 1:
            # Retain the canonical term for indexing or search.
            result.append(word)
    # Return the list used by vectors, BM25 and entity matching.
    return result


# Build a sparse deterministic lexical vector with 4,096 possible positions.
def vector(text: str) -> dict[str, float]:
    # Apply the research tokenizer before creating features.
    words = tokens(text)
    # Include individual words and pairs of adjacent words.
    features = words + [f"{a}:{b}" for a, b in zip(words, words[1:])]
    # Hash each feature to a bucket and count occurrences.
    counts = Counter(str(int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 4096) for word in features)
    # Compute vector length, using one when there are no features.
    norm = math.sqrt(sum(count * count for count in counts.values())) or 1
    # Divide every occupied bucket by that length.
    return {key: count / norm for key, count in sorted(counts.items())}


# Describe one cited passage returned by the research retriever.
class RetrievalHit(BaseModel):
    # Give the passage a stable citable ID.
    citation_id: str
    # Link back to the document it belongs to.
    document_id: str
    # Show the source passage title.
    title: str
    # Return its retrieved text.
    text: str
    # Identify where the material came from.
    source: str
    # Record which source revision was indexed.
    source_revision: str
    # Preserve the source's license or access note.
    license: str
    # Include the curriculum or research entities linked to this passage.
    entity_ids: list[str]
    # Return its fused ranking score.
    score: float
    # Name the channels contributing to that score.
    channels: list[str]


# Define the query response used by the research API and tutor.
class RetrievalResult(BaseModel):
    # Echo the query for inspection.
    query: str
    # Record the applied retrieval mode.
    mode: Literal["local", "global", "hybrid", "naive", "mix"]
    # Label the searched source collection.
    namespace: str
    # Include the current ranking and index version.
    index_version: str = INDEX_VERSION
    # Fingerprint the exact indexed source collection.
    index_fingerprint: str
    # Start with no passage hits.
    hits: list[RetrievalHit] = Field(default_factory=list)
    # Start with no supported relationship edges.
    relationships: list[dict] = Field(default_factory=list)
    # Accumulate the formatted source text within the requested limit.
    context: str = ""
    # Indicate an empty or unsupported result explicitly.
    abstained: bool = False
    # Flag passages skipped to respect the context size limit.
    truncated: bool = False
    # Explain how to interpret the score.
    ranking_note: str = "Reciprocal rank fusion of deterministic lexical, hashed-vector, and graph channels; scores are not confidence probabilities."


# Store and query the research index independently from the live assessment search tables.
class KnowledgeStore:
    # Accept the SQLite database path shared by the research application.
    def __init__(self, db_path: str):
        # Keep the path for future connections.
        self.db_path = db_path
        # Create the parent directory when needed.
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Use an owned connection to initialize the index.
        with self.connection() as conn:
            # Create namespace, source, chunk, entity and edge tables, a full-text index and their
            # lookup indexes.
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
            # Backfill the mapping from previously indexed passage IDs to their FTS row numbers.
            conn.execute("INSERT OR IGNORE INTO rag_fts_map SELECT chunk_id,rowid FROM rag_fts")

    # Ensure each use of the index has managed connection lifetime.
    @contextmanager
    # Open a configured connection for one research index operation.
    def connection(self):
        # Apply the platform's standard SQLite settings.
        conn = open_connection(self.db_path)
        # Guarantee cleanup even when the operation raises.
        try:
            # Let the connection commit or roll back as appropriate.
            with conn:
                # Expose the open connection to the caller.
                yield conn
        # Run cleanup for a normal or failed operation.
        finally:
            # Release the owned connection.
            conn.close()

    # Replace a namespace's indexed entries with the supplied source manifest.
    def sync(self, documents: list[dict], namespace: str = "curriculum") -> dict:
        """Atomically synchronize a source namespace, including stale entity/edge deletion."""
        # Collect all expected document IDs before writing.
        expected = {doc["document_id"] for doc in documents}
        # Detect duplicate IDs that would make source provenance ambiguous.
        if len(expected) != len(documents):
            # Reject the incomplete manifest before making changes.
            raise ValueError("Duplicate document IDs")
        # Count updated documents and removals separately.
        changed = removed = 0
        # Use one connection for the whole synchronization.
        with self.connection() as conn:
            # Take the write lock up front so competing writers cannot interleave.
            conn.execute("BEGIN IMMEDIATE")
            # Read existing IDs and fingerprints only from the selected namespace.
            existing = {row["document_id"]: row["fingerprint"] for row in conn.execute("SELECT document_id,fingerprint FROM rag_documents WHERE namespace=?", (namespace,))}
            # Remove indexed documents missing from the supplied manifest.
            for doc_id in sorted(set(existing) - expected):
                # Delete their FTS rows explicitly, since the full-text table has no cascading
                # foreign key.
                conn.execute("DELETE FROM rag_fts WHERE rowid IN (SELECT f.fts_rowid FROM rag_fts_map f JOIN rag_chunks c USING(chunk_id) WHERE c.document_id=?)", (doc_id,))
                # Delete each source and let dependent relational records cascade.
                conn.execute("DELETE FROM rag_documents WHERE document_id=? AND namespace=?", (doc_id, namespace))
                # Count the deleted source.
                removed += 1
            # Work through the supplied sources in stable ID order.
            for doc in sorted(documents, key=lambda item: item["document_id"]):
                # Read the source ID used in the index.
                doc_id = doc["document_id"]
                # Fingerprint both the document contents and the index version.
                fingerprint = identifier("D_", [INDEX_VERSION, doc])
                # Avoid relinking an unchanged document.
                if existing.get(doc_id) == fingerprint:
                    # Leave its existing chunks and graph relationships in place.
                    continue
                # Check if this document ID is already used in the wider database.
                owner = conn.execute("SELECT namespace FROM rag_documents WHERE document_id=?", (doc_id,)).fetchone()
                # Reject an ID collision across otherwise independent namespaces.
                if owner and owner[0] != namespace:
                    # Ask the caller to use an ID unique to its collection.
                    raise ValueError("Document ID already belongs to another namespace")
                # Replace the entries for an existing source whose contents changed.
                if owner:
                    # Remove its previous full-text index rows.
                    conn.execute("DELETE FROM rag_fts WHERE rowid IN (SELECT f.fts_rowid FROM rag_fts_map f JOIN rag_chunks c USING(chunk_id) WHERE c.document_id=?)", (doc_id,))
                    # Delete the prior source and its relational graph/index records.
                    conn.execute("DELETE FROM rag_documents WHERE document_id=?", (doc_id,))
                # Save the source's current content and fingerprint.
                conn.execute("INSERT INTO rag_documents VALUES(?,?,?,?)", (doc_id, namespace, fingerprint, canonical(doc)))
                # Walk the entities supplied by the corpus for this document.
                for entity in doc.get("entities", []):
                    # Link each entity to the source providing it.
                    conn.execute("INSERT INTO rag_entities VALUES(?,?,?,?,?)", (namespace, entity["id"], doc_id, entity["name"], entity["kind"]))
                # Walk the supplied relationships for this document.
                for edge in doc.get("edges", []):
                    # Store each directed relationship and its traceable source.
                    conn.execute("INSERT INTO rag_edges VALUES(?,?,?,?,?,?)", (namespace, edge["source"], edge["target"], edge["relation"], doc_id, edge.get("weight", 1.0)))
                # Paragraph chunks preserve source boundaries; overlapping words only for long paragraphs.
                # Split source text into passages with a stable position.
                for position, text in enumerate(self._chunks(doc["text"])):
                    # Hash source, revision, position and content to produce a citation ID.
                    chunk_id = identifier("C_", [doc_id, fingerprint, position, text])
                    # Record which curated entities are associated with the document.
                    entity_ids = sorted(entity["id"] for entity in doc.get("entities", []))
                    # Give title terms more weight by repeating the title before indexing.
                    content = (doc["title"] + " ") * 3 + text
                    # Save the original passage alongside its lexical vector and entity IDs.
                    conn.execute("INSERT INTO rag_chunks VALUES(?,?,?,?,?,?,?)", (chunk_id, doc_id, namespace, doc["title"], text, canonical(entity_ids), canonical(vector(content))))
                    # Index normalized terms for BM25 and capture the FTS row number.
                    fts_rowid = conn.execute("INSERT INTO rag_fts VALUES(?,?,?)", (chunk_id, namespace, " ".join(tokens(content)))).lastrowid
                    # Save the link used for later replacement or deletion.
                    conn.execute("INSERT INTO rag_fts_map VALUES(?,?)", (chunk_id,fts_rowid))
                # Count this document as re-indexed.
                changed += 1
            # Record the index version for this namespace.
            conn.execute("INSERT INTO rag_namespaces VALUES(?,?) ON CONFLICT(namespace) DO UPDATE SET index_version=excluded.index_version", (namespace,INDEX_VERSION))
        # Return the updated manifest summary and counts.
        return {"updated_documents": changed, "deleted_documents": removed, **self.status(namespace)}

    # This chunking rule needs only the source text.
    @staticmethod
    # Divide long source paragraphs into smaller overlapping word windows.
    def _chunks(text: str, size: int = 180, overlap: int = 24):
        # Split the original document at blank paragraph boundaries.
        for paragraph in re.split(r"\n\s*\n", text.strip()):
            # Turn each paragraph into a sequence of words.
            words = paragraph.split()
            # Move by the window length minus the overlap so context can carry forward.
            for start in range(0, len(words), size - overlap):
                # Return one search passage with up to this many words.
                yield " ".join(words[start:start + size])
                # Stop when this passage reaches the last word in the paragraph.
                if start + size >= len(words):
                    # Avoid an unnecessary extra overlapping passage at the end.
                    break

    # Describe the indexed source collection without running a search.
    def status(self, namespace: str = "curriculum") -> dict:
        # Open a connection for the status query.
        with self.connection() as conn:
            # Build a manifest from sorted document IDs and fingerprints.
            rows = list(conn.execute("SELECT document_id,fingerprint FROM rag_documents WHERE namespace=? ORDER BY document_id", (namespace,)))
            # Count chunks, entity records and relationship records within this namespace.
            counts = {name: conn.execute(f"SELECT COUNT(*) FROM rag_{name} WHERE namespace=?", (namespace,)).fetchone()[0] for name in ("chunks", "entities", "edges")}
            # Count unique entity IDs because an entity can occur in multiple documents.
            unique_entities = conn.execute("SELECT COUNT(DISTINCT entity_id) FROM rag_entities WHERE namespace=?", (namespace,)).fetchone()[0]
        # Summarize the collection and identify its manifest with a stable fingerprint.
        return {"namespace": namespace, "documents": len(rows), "unique_entities": unique_entities, **counts, "index_version": INDEX_VERSION, "index_fingerprint": identifier("INDEX_", [tuple(row) for row in rows])}

    # Search one research collection using the requested ranking channels.
    def query(self, query: str, mode: str = "mix", namespace: str = "curriculum", top_k: int = 6, max_context_chars: int = 6000, concept_ids: list[str] | None = None) -> RetrievalResult:
        # Reject modes the store cannot serve.
        if mode not in MODES:
            # Return an explicit validation error for an unknown mode.
            raise ValueError("Unsupported retrieval mode")
        # Bound both the maximum number of hits and the formatted context length.
        if not 1 <= top_k <= 20 or not 128 <= max_context_chars <= 20000:
            # Reject budgets outside those limits.
            raise ValueError("Invalid retrieval budget")
        # Load the namespace's metadata for the result.
        status = self.status(namespace)
        # Create a result that identifies the requested scope and fingerprint.
        result = RetrievalResult(query=query, mode=mode, namespace=namespace, index_fingerprint=status["index_fingerprint"])
        # Extract vocabulary using the same tokenizer as the index.
        query_tokens = tokens(query)
        # Gather explicit curriculum hints passed by the tutor or explorer.
        hints = set(concept_ids or [])
        # There is nothing to search without query terms or entity hints.
        if not query_tokens and not hints:
            # Set the abstention marker for the caller.
            result.abstained = True
            # Return without claiming any relevant sources.
            return result
        # Use one read connection for a consistent indexed collection.
        with self.connection() as conn:
            # Freeze the read view while loading the manifest, passages and graph.
            conn.execute("BEGIN")
            # Read the source manifest from this same database view.
            manifest = [tuple(row) for row in conn.execute("SELECT document_id,fingerprint FROM rag_documents WHERE namespace=? ORDER BY document_id", (namespace,))]
            # Tie the result to the exact in-transaction collection fingerprint.
            result.index_fingerprint = identifier("INDEX_", manifest)
            # Check which index format produced the stored rows.
            version = conn.execute("SELECT index_version FROM rag_namespaces WHERE namespace=?", (namespace,)).fetchone()
            # Detect a manifest whose stored format cannot support the current ranking.
            if manifest and (version is None or version[0] != INDEX_VERSION):
                # Ask the caller to re-index the collection before querying it.
                raise ValueError("Source collection needs reindexing with the current retrieval version")
            # Load scoped passage content, features and source metadata by chunk ID.
            chunks = {row["chunk_id"]: dict(row) for row in conn.execute("SELECT c.*,d.payload_json FROM rag_chunks c JOIN rag_documents d USING(document_id) WHERE c.namespace=? ORDER BY c.chunk_id", (namespace,))}
            # There are no possible hits if the scope contains no chunks.
            if not chunks:
                # Record explicit abstention for an empty source collection.
                result.abstained = True
                # Return the empty result to the caller.
                return result
            # Collect the ranked chunks from each selected channel.
            lists: dict[str, list[str]] = {}
            # Match any of up to 48 unique query terms in FTS.
            expression = " OR ".join('"' + word + '"' for word in sorted(set(query_tokens))[:48])
            # Keep the BM25 candidate list empty if there are no usable terms.
            lexical = []
            # Only run FTS on a non-empty query expression.
            if expression:
                # Rank the best lexical hits in this namespace using BM25.
                lexical = [row["chunk_id"] for row in conn.execute("SELECT chunk_id,bm25(rag_fts) AS rank FROM rag_fts WHERE rag_fts MATCH ? AND namespace=? ORDER BY rank,chunk_id LIMIT 80", (expression, namespace))]
            # Compute a normalized hashed lexical vector for the query.
            qvec = vector(query)
            # Record candidate dot-product scores before sorting them.
            vector_scores = []
            # Compare the query against each passage's stored vector.
            for chunk_id, chunk in chunks.items():
                # Hash collisions alone cannot make an unrelated source relevant.
                # Require actual token overlap before awarding any vector match.
                if not set(query_tokens).intersection(tokens(chunk["title"] + " " + chunk["text"])):
                    # Skip passages with no shared search terms even if vector buckets collide.
                    continue
                # Load the passage's saved nonzero vector buckets.
                stored = json.loads(chunk["vector_json"])
                # Compute cosine similarity as the dot product of unit-normalized vectors.
                score = sum(value * stored.get(key, 0) for key, value in qvec.items())
                # Keep only positively scoring candidates.
                if score > 0:
                    # Retain each candidate ID and score.
                    vector_scores.append((chunk_id, score))
            # Sort by descending similarity with a stable ID tie-breaker.
            vector_rank = [cid for cid, score in sorted(vector_scores, key=lambda entry: (-entry[1], entry[0]))[:80]]
            # Read the namespace's known graph entities in a stable order.
            entities = list(conn.execute("SELECT DISTINCT entity_id,name FROM rag_entities WHERE namespace=? ORDER BY entity_id", (namespace,)))
            # Seed the local graph set from explicit concept hints.
            local_entities = set(hints)
            # Test each graph entity against the query terminology.
            for entity in entities:
                # Tokenize both the display name and the entity ID.
                name_tokens = set(tokens(entity["name"] + " " + entity["entity_id"]))
                # Count vocabulary shared by the entity and query.
                overlap = set(query_tokens) & name_tokens
                # Require up to two overlapping words before adding an entity seed.
                if len(overlap) >= min(2, len(name_tokens)) and overlap:
                    # Add this matched entity to the local neighborhood.
                    local_entities.add(entity["entity_id"])
            # Seed graph from the most relevant chunks as well as explicit concept hints.
            # Use the top two vector results as further graph seeds.
            for cid in vector_rank[:2]:
                # Add the curated entity IDs attached to those passages.
                local_entities.update(json.loads(chunks[cid]["entity_ids_json"]))
            # Load distinct relationships within the searched namespace.
            all_edges = [dict(row) for row in conn.execute("SELECT DISTINCT source_id,target_id,relation,weight FROM rag_edges WHERE namespace=? ORDER BY source_id,target_id,relation,weight", (namespace,))]
            # Begin graph expansion from the local entity set.
            frontier = set(local_entities)
            # Collect newly discovered entities separately from the seeds.
            global_entities = set()
            # Track relationships visited while walking the graph.
            selected_edges = []
            # Expand at most two hops outward from the seed entities.
            for _ in range(2):
                # Start the next hop with an empty frontier.
                next_frontier = set()
                # Consider edges tied to an entity reached at the current hop.
                for edge in all_edges:
                    # Treat either endpoint as a possible neighbor during traversal.
                    if edge["source_id"] in frontier or edge["target_id"] in frontier:
                        # Save the relationship for result provenance.
                        selected_edges.append(edge)
                        # Consider both ends of the edge for the next level.
                        next_frontier.update((edge["source_id"], edge["target_id"]))
                # Exclude the seeds and entities already discovered.
                next_frontier -= local_entities | global_entities
                # Add the newly discovered neighbors to the broader graph set.
                global_entities.update(next_frontier)
                # Continue expansion from those neighbors.
                frontier = next_frontier
            # Rank passages against a chosen set of graph entities.
            def graph_rank(entity_ids):
                # Score each passage by the number of linked entities it shares with that set.
                scores = [(cid, len(set(json.loads(chunk["entity_ids_json"])) & entity_ids)) for cid, chunk in chunks.items()]
                # Prepare the vector score lookup for ties.
                query_scores = dict(vector_scores)
                # Sort by entity overlap, then vector score and ID, and limit to 80 hits.
                return [cid for cid, score in sorted(scores, key=lambda entry: (-entry[1], -query_scores.get(entry[0], 0), entry[0])) if score > 0][:80]
            # Use direct lexical search channels in naive and mix modes.
            if mode in {"naive", "mix"}:
                # Include the BM25 ranking.
                lists["bm25"] = lexical
                # Include the hashed lexical vector ranking.
                lists["hashed_vector"] = vector_rank
            # Add a graph channel for directly related entities in these modes.
            if mode in {"local", "hybrid", "mix"}:
                # Rank passages against the query's local entity set.
                lists["local_entity"] = graph_rank(local_entities)
            # Add relationship expansion for global, hybrid and mix modes.
            if mode in {"global", "hybrid", "mix"}:
                # Rank against both graph seeds and discovered neighbors.
                lists["global_relationship"] = graph_rank(global_entities | local_entities)
            # Graph support augments precision; direct lexical ranks break graph ties.
            # Use vector ranking in the graph-focused modes when graph entities were found.
            if mode in {"local", "global", "hybrid"} and (local_entities or global_entities):
                # Add the direct query signal as a tie-breaking ranking channel.
                lists["query_rerank"] = vector_rank
            # Accumulate one fused score per passage.
            fused = defaultdict(float)
            # Track which channels justified each candidate's overall rank.
            channels = defaultdict(list)
            # Walk the selected ranking channels.
            for channel, ranked in lists.items():
                # Use ranks starting from one in the fusion formula.
                for rank, cid in enumerate(ranked, 1):
                    # Give lexical channels stronger weights than graph channels.
                    weight = {"bm25": 2.0, "hashed_vector": 2.0, "local_entity": 0.35, "global_relationship": 0.2, "query_rerank": 2.0}[channel]
                    # Add this channel's weight divided by 60 plus rank to the passage total.
                    fused[cid] += weight / (60 + rank)
                    # Record the individual channel as ranking provenance.
                    channels[cid].append(channel)
            # Start with the caller's full character budget.
            remaining = max_context_chars
            # Take up to the requested number of candidates in fused score order.
            for cid in sorted(fused, key=lambda key: (-fused[key], key))[:top_k]:
                # Read the stored passage corresponding to the candidate.
                chunk = chunks[cid]
                # Decode the document's source and licensing metadata.
                doc = json.loads(chunk["payload_json"])
                # Use the passage ID as its citation.
                citation_id = cid
                # Format a citation, title, passage and source for the tutor context.
                rendered = f"[{citation_id}] {chunk['title']}\n{chunk['text']}\nSource: {doc['source']}\n"
                # Check whether the formatted passage fits the remaining budget.
                if len(rendered) > remaining:
                    # Flag context truncation when a selected passage cannot fit.
                    result.truncated = True
                    # Skip it and consider any remaining candidates.
                    continue
                # Reduce the remaining character budget by the accepted passage length.
                remaining -= len(rendered)
                # Return a typed hit with citations, source provenance and score contributions.
                result.hits.append(RetrievalHit(citation_id=citation_id, document_id=chunk["document_id"], title=chunk["title"], text=chunk["text"], source=doc["source"], source_revision=doc["revision"], license=doc["license"], entity_ids=json.loads(chunk["entity_ids_json"]), score=round(fused[cid], 8), channels=channels[cid]))
                # Append its rendered text to the combined context.
                result.context += rendered
            # Collect entities actually supported by the passages that fit.
            hit_entities = {eid for hit in result.hits for eid in hit.entity_ids}
            # Keep only distinct visited edges whose endpoints appear in those passages.
            unique_edges = {canonical(edge): edge for edge in selected_edges if edge["source_id"] in hit_entities and edge["target_id"] in hit_entities}
            # Return up to 60 of those supporting relationship traces in stable order.
            result.relationships = [unique_edges[key] for key in sorted(unique_edges)][:60]
            # Abstain explicitly if no passages could be returned.
            result.abstained = not result.hits
        # Return the scoped research search result.
        return result
