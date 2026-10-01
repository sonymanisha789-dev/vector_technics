"""
Store chunk embeddings in Postgres + pgvector.

One table, one `vector` column with NO fixed dimension. Normally pgvector
wants `vector(384)` etc. so it can build an ANN index, but your model_dict
has three different embedding sizes (384/768/1024) and this is an eval POC,
not a production index - so a dimension-less `vector` column (pgvector
0.5+) that accepts any size, with a plain sequential scan, is the lazy
correct choice here. Add a fixed-dimension column + ivfflat/hnsw index only
once you pick ONE model to actually serve queries from.

Needs: pip install psycopg[binary]
Needs: docker compose up -d  (see docker-compose.yml)
"""
import json

import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://rag:rag@localhost:5432/rag"


def get_connection() -> psycopg.Connection:
    return psycopg.connect(DSN, autocommit=True)


def init_db(conn: psycopg.Connection) -> None:
    """
    Create the pgvector extension, the one table this POC needs, and teach
    THIS psycopg connection how to speak pgvector's wire format. Without
    register_vector(), psycopg has no idea what a Postgres `vector` column
    is - it comes back as a raw string like "[0.1,0.2,0.3]" instead of a
    list of floats, and inserts would need to be hand-formatted the same way.
    Must run AFTER `CREATE EXTENSION vector` - it looks up the type in the
    DB to register it, so the extension has to already exist.
    """
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id SERIAL PRIMARY KEY,
            doc_name TEXT NOT NULL,           -- chunks_dict key, e.g. "fixed_size_chunks_doc1"
            model_name TEXT NOT NULL,         -- model_dict key, e.g. "BAAI/bge-m3"
            chunk_index INT NOT NULL,         -- position within that doc's chunk list
            chunk_text TEXT NOT NULL,
            embedding vector NOT NULL,
            UNIQUE (doc_name, model_name, chunk_index)
        )
    """)
    # Sparse (lexical) embeddings are {token_id: weight} dicts, not fixed-
    # length vectors - can't live in a `vector` column, hence a separate
    # table. No model_name column here (unlike chunk_embeddings above):
    # sparse embeddings only ever come from BGE-m3's lexical-weights output
    # (see sparse_embeddings.py), so there's only ever one sparse embedding
    # per (doc_name, chunk_index), same shape as the notebook's
    # sparse_embeddings_dict which is keyed by doc_name alone.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_sparse_weights (
            id SERIAL PRIMARY KEY,
            doc_name TEXT NOT NULL,
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL,
            weights JSONB NOT NULL,
            UNIQUE (doc_name, chunk_index)
        )
    """)


def store_embeddings(
    conn: psycopg.Connection,
    doc_name: str,
    model_name: str,
    chunks: list,
    embeddings,  # torch tensor from embed_chunks(), shape (n_chunks, dim)
) -> None:
    """
    Insert one (doc_name, model_name) combo's chunks + embeddings.

    DELETE-then-INSERT, not ON CONFLICT DO UPDATE: if you re-chunk a doc and
    end up with FEWER chunks than last time (e.g. you tweak chunk_size and
    118 chunks becomes 90), ON CONFLICT only touches chunk_index 0..89 -
    the old rows for index 90..117 have no new row to conflict with, so they
    silently stick around as stale leftovers with old text. Deleting every
    row for this (doc_name, model_name) first guarantees the table always
    matches exactly what you just passed in, however many chunks that is.
    """
    conn.execute(
        "DELETE FROM chunk_embeddings WHERE doc_name = %s AND model_name = %s",
        (doc_name, model_name),
    )
    rows = [
        (doc_name, model_name, i, str(chunk), embedding.tolist() if hasattr(embedding, "tolist") else list(embedding))
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    conn.cursor().executemany(
        """
        INSERT INTO chunk_embeddings (doc_name, model_name, chunk_index, chunk_text, embedding)
        VALUES (%s, %s, %s, %s, %s)
        """,
        rows,
    )


def store_all_embeddings(conn: psycopg.Connection, chunks_dict: dict, chunk_embeddings_dict: dict) -> None:
    """
    Store every (doc_name, model_name) combo already computed in the
    notebook - one call instead of hand-writing the loop in the notebook
    itself. Fully dynamic: it just walks whatever keys chunk_embeddings_dict
    happens to have, so adding/removing a chunking strategy or a model in
    model_dict needs no change here.
    """
    for (doc_name, model_name), embeddings in chunk_embeddings_dict.items():
        store_embeddings(conn, doc_name, model_name, chunks_dict[doc_name], embeddings)


def store_sparse_embeddings(conn: psycopg.Connection, doc_name: str, chunks: list, sparse_weights: list[dict]) -> None:
    """
    Insert one doc_name's chunks + sparse lexical-weight dicts.

    Same DELETE-then-INSERT pattern as store_embeddings, for the same reason
    (see that function's docstring): guarantees the table always matches
    exactly what you pass in, even if the chunk count shrank since last run.
    `json.dumps(weights)` turns the {token_id: weight} dict into a JSON
    string for the `JSONB` column - psycopg doesn't auto-convert plain dicts
    the way pgvector.psycopg's register_vector() does for vectors.

    `float(weight)` on every value: BGE-m3's lexical_weights dict holds
    numpy float32s, not plain Python floats - json.dumps only knows how to
    serialize the built-in float type, so a bare `json.dumps(weights)`
    raises `TypeError: Object of type float32 is not JSON serializable`.
    """
    conn.execute("DELETE FROM chunk_sparse_weights WHERE doc_name = %s", (doc_name,))
    rows = [
        (doc_name, i, str(chunk), json.dumps({token: float(weight) for token, weight in weights.items()}))
        for i, (chunk, weights) in enumerate(zip(chunks, sparse_weights))
    ]
    conn.cursor().executemany(
        """
        INSERT INTO chunk_sparse_weights (doc_name, chunk_index, chunk_text, weights)
        VALUES (%s, %s, %s, %s)
        """,
        rows,
    )


def store_all_sparse_embeddings(conn: psycopg.Connection, chunks_dict: dict, sparse_embeddings_dict: dict) -> None:
    """Store every doc_name's sparse embeddings already computed in the notebook - mirrors store_all_embeddings."""
    for doc_name, sparse_weights in sparse_embeddings_dict.items():
        store_sparse_embeddings(conn, doc_name, chunks_dict[doc_name], sparse_weights)


def fetch_dense_ranking(conn: psycopg.Connection, doc_name: str, model_name: str, query_embedding) -> list[tuple[int, str]]:
    """
    Rank every chunk of (doc_name, model_name) by how close its stored
    embedding is to query_embedding, using pgvector's `<=>` operator
    (cosine DISTANCE - smaller is closer, so `ORDER BY` ascending already
    puts the best match first). Postgres does the ranking here, unlike the
    in-memory retrieve_top_k functions elsewhere in this project which pull
    every embedding into Python first and call util.cos_sim by hand - there's
    nothing left to compute locally once this query returns.

    Returns the FULL ranking (no LIMIT) - retrieve_hybrid_pgvector in
    rag_query.py needs every chunk's rank to fuse with the sparse ranking via
    Reciprocal Rank Fusion, not just a small top-k cutoff (same reason
    hybrid_search.py's retrieve_top_k asks its dense/sparse retrieve_top_k
    calls for k=len(chunks) instead of a small k).
    """
    rows = conn.execute(
        """
        SELECT chunk_index, chunk_text
        FROM chunk_embeddings
        WHERE doc_name = %s AND model_name = %s
        ORDER BY embedding <=> %s::vector
        """,
        (doc_name, model_name, query_embedding),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def fetch_sparse_weights(conn: psycopg.Connection, doc_name: str) -> list[tuple[int, str, dict]]:
    """
    Read back every chunk's raw sparse weights for doc_name - NOT ranked
    against a query, unlike fetch_dense_ranking. Scoring a lexical-weights
    dict against a query needs BGE-m3's own compute_lexical_matching_score,
    which lives with the model in sparse_embeddings.py, not here (same
    "storage module fetches, embedding module scores" split dense_embeddings.py/
    sparse_embeddings.py already use) - rag_query.py's retrieve_hybrid_pgvector
    does that scoring with the rows this returns.
    """
    rows = conn.execute(
        """
        SELECT chunk_index, chunk_text, weights
        FROM chunk_sparse_weights
        WHERE doc_name = %s
        ORDER BY chunk_index
        """,
        (doc_name,),
    ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


if __name__ == "__main__":
    # Cheap self-check: round-trips a fake embedding through a real DB
    # connection and confirms it comes back byte-for-byte (well, float-for-
    # float) the same. Needs `docker compose up -d` running first.
    conn = get_connection()
    init_db(conn)
    store_embeddings(conn, "_selftest_doc", "_selftest_model", ["a", "b", "c"], [[0.1], [0.2], [0.3]])

    row = conn.execute(
        "SELECT chunk_text, embedding FROM chunk_embeddings WHERE doc_name = %s AND chunk_index = 0",
        ("_selftest_doc",),
    ).fetchone()
    assert row[0] == "a"
    assert round(row[1][0], 4) == 0.1, row[1]

    # re-store with FEWER chunks than before - this is the stale-row bug
    # store_embeddings' DELETE-then-INSERT is guarding against
    store_embeddings(conn, "_selftest_doc", "_selftest_model", ["only one now"], [[0.9]])
    count = conn.execute(
        "SELECT count(*) FROM chunk_embeddings WHERE doc_name = %s", ("_selftest_doc",)
    ).fetchone()[0]
    assert count == 1, f"expected old chunk_index 1,2 rows to be gone, found {count} rows"

    conn.execute("DELETE FROM chunk_embeddings WHERE doc_name = %s", ("_selftest_doc",))

    # Same round-trip, now for sparse weights: store 2 fake {token: weight}
    # dicts, read them back, confirm the JSONB column preserved them exactly
    # (dict equality - JSON round-trips numbers/strings exactly, no float
    # rounding concern like the vector column above).
    fake_weights = [{"101": 0.5, "202": 0.25}, {"303": 0.9}]
    store_sparse_embeddings(conn, "_selftest_doc", ["a", "b"], fake_weights)

    fetched = fetch_sparse_weights(conn, "_selftest_doc")
    assert len(fetched) == 2, fetched
    assert fetched[0] == (0, "a", fake_weights[0]), fetched[0]
    assert fetched[1] == (1, "b", fake_weights[1]), fetched[1]

    # re-store with FEWER chunks - same stale-row guard as the dense check above
    store_sparse_embeddings(conn, "_selftest_doc", ["only one now"], [{"999": 1.0}])
    count = conn.execute(
        "SELECT count(*) FROM chunk_sparse_weights WHERE doc_name = %s", ("_selftest_doc",)
    ).fetchone()[0]
    assert count == 1, f"expected old chunk_index 1 row to be gone, found {count} rows"

    conn.execute("DELETE FROM chunk_sparse_weights WHERE doc_name = %s", ("_selftest_doc",))
    print("self-check passed")
