"""
Hybrid search: combine dense_embeddings.py's semantic ranking with
sparse_embeddings.py's keyword ranking into ONE ranking, instead of picking
just one. This is the "best of both worlds" pattern both those files'
docstrings point at but never wire up - dense alone misses exact-term
needles (a part number, an acronym that never appeared in training data),
sparse alone misses paraphrases ("car" vs "automobile"). Fusing both means a
chunk only has to win on ONE axis to rank well, not both at once.

Fusion method: Reciprocal Rank Fusion (RRF), not a weighted score average.
Dense's cosine similarity (always 0-1) and sparse's lexical dot-product score
(unbounded - could be 0.02 or 8.0 depending on how many tokens overlap) live
on completely different scales. A naive weighted average like
`0.5 * dense_score + 0.5 * sparse_score` would let whichever score happens to
produce bigger raw numbers quietly dominate the result - "fixing" that needs
per-query min-max normalization AND a hand-tuned weight. RRF sidesteps all of
that: it only looks at each chunk's RANK (1st place, 2nd place, ...) in each
ranking, never the raw score, so there's nothing to normalize or tune. This
is also why RRF - not weighted score averaging - is the default hybrid-search
fusion method in Elasticsearch, OpenSearch, and Weaviate.

Formula per chunk: RRF(chunk) = sum, over every ranking it appears in, of
1 / (rrf_k + rank_in_that_ranking). rrf_k=60 is the constant from the
original RRF paper (Cormack, Clarke & Buettcher, 2009) - it softens how much
rank 1 beats rank 2: without it (rrf_k=0), rank 1 would score exactly double
rank 2 (1/1 vs 1/2), overweighting small, often noisy differences right at
the top of a ranking.
"""
from dense_embeddings import _metrics_from_relevance
from dense_embeddings import embed_chunks as embed_chunks_dense
from dense_embeddings import retrieve_top_k as retrieve_top_k_dense
from sparse_embeddings import embed_chunks_sparse
from sparse_embeddings import retrieve_top_k as retrieve_top_k_sparse


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 3, rrf_k: int = 60) -> list[int]:
    """
    The general fusion step, kept separate from any specific embedding
    model: takes any number of chunk-index rankings (each one a list of
    chunk positions, best match first - exactly what retrieve_top_k already
    returns) and merges them into one ranking. Works for 2 rankings (dense +
    sparse, this file's main use) just as well as 3+ (e.g. fusing multiple
    dense models' rankings too) - nothing here is dense- or sparse-specific.

    `scores.get(chunk_index, 0.0)` instead of a `defaultdict` - a chunk that
    a given ranking doesn't rank at all (rare here since we pass in FULL
    rankings, but this keeps the function correct even for partial ones)
    simply contributes 0 from that ranking instead of needing a KeyError
    guard.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_index in enumerate(ranking, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0.0) + 1 / (rrf_k + rank)

    ranked_indices = sorted(scores, key=lambda i: scores[i], reverse=True)
    return ranked_indices[:k]


def retrieve_top_k(chunks: list[str], chunk_dense_embeddings, chunk_sparse_weights: list[dict], query: str, k: int = 3, rrf_k: int = 60) -> list[int]:
    """
    Convenience wrapper pairing dense_embeddings.py's MiniLM ranking with
    sparse_embeddings.py's BGE-m3 sparse ranking specifically (mirrors those
    two files' own single-model defaults). Calls each one's own retrieve_top_k
    with `k=len(chunks)` - not this function's own `k` - to get each
    retriever's FULL ranking of every chunk before fusing; RRF needs to know
    where a chunk landed in the full order, not just whether it made a small
    top-k cutoff. Reuses both files' existing, already-tested retrieve_top_k
    instead of re-scoring chunks by hand.
    """
    n = len(chunks)
    dense_ranking = retrieve_top_k_dense(chunks, chunk_dense_embeddings, query, k=n)
    sparse_ranking = retrieve_top_k_sparse(chunks, chunk_sparse_weights, query, k=n)
    return reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=k, rrf_k=rrf_k)


def evaluate_retrieval(chunks: list[str], chunk_dense_embeddings, chunk_sparse_weights: list[dict], eval_set: list[tuple[str, str]], k: int = 3, rrf_k: int = 60) -> dict[str, float]:
    """Same shape as dense_embeddings.py/sparse_embeddings.py's evaluate_retrieval - only the retrieval call differs."""
    per_question = []
    for question, answer in eval_set:
        top_k_indices = retrieve_top_k(chunks, chunk_dense_embeddings, chunk_sparse_weights, question, k, rrf_k)
        relevant = [1 if answer.lower() in str(chunks[i]).lower() else 0 for i in top_k_indices]
        per_question.append(_metrics_from_relevance(relevant))

    n = len(per_question)
    return {
        "recall@k": sum(m["recall"] for m in per_question) / n,
        "precision@k": sum(m["precision"] for m in per_question) / n,
        "mrr": sum(m["reciprocal_rank"] for m in per_question) / n,
        "ndcg@k": sum(m["ndcg"] for m in per_question) / n,
    }


class _FakeElement:
    """Mimics an `unstructured` Element - see dense_embeddings.py/sparse_embeddings.py for why this matters for evaluate_retrieval."""
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


if __name__ == "__main__":
    # Cheap self-check - both models are already downloaded from running
    # dense_embeddings.py/sparse_embeddings.py earlier, so this is fast.
    # Confirms fusion actually favors the on-topic chunk, and that
    # evaluate_retrieval works against non-string (Element-like) chunks.
    demo_chunks = [_FakeElement("Vector Technics builds industrial sensors."), _FakeElement("Q3 revenue grew 40%.")]
    demo_query = "industrial sensors"

    demo_dense_embeddings = embed_chunks_dense(demo_chunks)
    demo_sparse_weights = embed_chunks_sparse(demo_chunks)

    top_k = retrieve_top_k(demo_chunks, demo_dense_embeddings, demo_sparse_weights, demo_query, k=1)
    assert top_k == [0], f"expected the sensors chunk to rank first, got index {top_k}"

    demo_eval_set = [("industrial sensors?", "industrial sensors")]
    metrics = evaluate_retrieval(demo_chunks, demo_dense_embeddings, demo_sparse_weights, demo_eval_set, k=1)
    assert metrics["recall@k"] == 1.0, f"expected a perfect recall@k on this trivial eval set, got {metrics}"

    print("self-check passed")
