"""
Dense embeddings: turn each chunk into a vector, turn a query into a vector,
rank chunks by how close their vector is to the query's vector.

Uses sentence-transformers directly (already installed - it's what
langchain_huggingface's HuggingFaceEmbeddings wraps under the hood for the
SemanticChunker code from earlier). No LangChain layer needed here - we just
want vectors + cosine similarity, so the direct library is less code.

First run downloads the model (~80MB for MiniLM) from Hugging Face and caches
it under ~/.cache/huggingface - every run after that is fully offline.
"""
import math

from sentence_transformers import SentenceTransformer, util

# all-MiniLM-L6-v2: smallest/fastest option from the size table in rag-strategy.md,
# good default for a POC. Unlike BGE models, it needs no special "instruction"
# prefix on the query - if you swap in a BGE model later, prefix the QUERY (not
# the chunks) with "Represent this sentence for searching relevant passages: "
# or recall silently gets worse.
#
# To swap in a different model, change this one line - everything below reads
# MODEL_NAME, nothing else needs to change:
#   MODEL_NAME = "intfloat/e5-base-v2"   # E5 family (small/base/large, or the
#                                          # multilingual-e5-* variants)
#   MODEL_NAME = "BAAI/bge-m3"           # BGE-m3 (~2.2GB - over the <1GB bar
#                                          # in rag-strategy.md, so only pick
#                                          # this if MiniLM's recall is too low)
# Both download automatically on first use, same as MiniLM did.
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# Cap how many tokens the model will ever try to process in one go. MiniLM
# already truncates at 256 by default, so this line is a no-op for it - but
# E5 (512) and especially BGE-m3 (8192) don't truncate nearly as early, and
# self-attention's memory cost grows with the SQUARE of sequence length, then
# multiplies again by how many chunks get batched together (sentence-
# transformers pads every chunk in a batch up to the length of the longest
# one in that batch). One unusually long chunk sneaking into a batch with a
# big model is exactly what produces a "RuntimeError: Invalid buffer size" -
# Apple's Metal/MPS GPU backend has a hard per-allocation size ceiling and
# refuses outright instead of just running slow. Capping the length here
# means no chunking strategy or model swap can trigger that again.
model.max_seq_length = 512


def embed_chunks(chunks: list[str], passage_prefix: str = ""):
    """
    Encode every chunk into a vector. `normalize_embeddings=True` scales each
    vector to length 1 - with unit-length vectors, cosine similarity (the
    "angle" between two vectors) becomes just a dot product, which is what
    `util.cos_sim` computes cheaply. `convert_to_tensor=True` keeps the result
    as a PyTorch tensor so `util.cos_sim`/`.topk()` below can operate on it
    directly instead of converting types back and forth.

    `passage_prefix` exists for E5 models, which - unlike MiniLM or BGE - need
    a literal "passage: " glued onto the front of every CHUNK, not just the
    query. Leave it as "" for MiniLM/BGE-m3 (see retrieve_top_k for the query
    side of this).

    `str(chunk)` (not just `chunk`) because `chunk_by_title` hands back
    `unstructured` Element objects, not plain strings - `str()` is a no-op on
    an already-plain string and pulls the text out of an Element, so this
    works no matter which chunker produced the list.
    """
    prefixed_chunks = [passage_prefix + str(chunk) for chunk in chunks]
    return model.encode(prefixed_chunks, normalize_embeddings=True, convert_to_tensor=True)


def retrieve_top_k(chunks: list[str], chunk_embeddings, query: str, k: int = 3, query_prefix: str = "") -> list[int]:
    """
    Embed the query the same way, compare it against every chunk's vector at
    once with `util.cos_sim` (returns a similarity score per chunk, 1.0 =
    identical direction, 0.0 = unrelated), then `.topk(k)` picks the k
    highest-scoring chunks and returns their positions in `chunks`.

    `query_prefix` is the query-side counterpart to `embed_chunks`'s
    `passage_prefix`:
    - MiniLM (default model here): leave both prefixes "" - it needs neither.
    - BGE (small/base/large, not bge-m3): query_prefix="Represent this
      sentence for searching relevant passages: ", passage_prefix="" - only
      the query gets prefixed.
    - E5 (any size, including multilingual-e5-*): query_prefix="query: ",
      passage_prefix="passage: " - BOTH sides get prefixed. Forgetting the
      passage side is the easy mistake, since it looks optional by analogy
      with BGE - it isn't, for E5.
    - BGE-m3 via this file (sentence-transformers): leave both "" - you get
      BGE-m3's dense vector only. Its sparse and multi-vector (ColBERT-style)
      outputs aren't exposed through sentence-transformers at all - those
      need the separate `FlagEmbedding` library's `BGEM3FlagModel` class
      instead, which is a different embedding call, not a prefix change.
    """
    query_embedding = model.encode(query_prefix + query, normalize_embeddings=True, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = scores.topk(min(k, len(chunks)))
    return top_k.indices.tolist()


def _metrics_from_relevance(relevant: list[int]) -> dict[str, float]:
    """
    All four metrics boil down to the same input: a list of 0/1 labels in
    rank order (relevant[0] = top-ranked retrieved chunk, relevant[1] =
    second, ...; 1 = this chunk actually contains the expected answer).
    Pulled out on its own so the metric math can be unit-tested directly,
    without needing a real embedding model to produce a known ranking.

    - recall: did ANY retrieved chunk hit? (0 or 1)
    - precision: what FRACTION of the k retrieved chunks hit?
    - reciprocal_rank: 1 / (rank of the first hit) - rank 1 scores 1.0,
      rank 2 scores 0.5, no hit scores 0.0. Averaging this across questions
      gives MRR.
    - ndcg: DCG (Discounted Cumulative Gain) discounts a hit by how far down
      the ranking it is - log2(rank + 1) grows slowly, so a hit at rank 1
      barely gets discounted (divided by log2(2)=1) while a hit at rank 4
      gets discounted hard (divided by log2(5)~2.32). IDCG is the best
      possible DCG for this same set of labels (all the 1s moved to the
      front) - dividing by it normalizes the score to 0-1. Note: with our
      eval set's binary "does the answer substring appear" labels, this is
      binary NDCG, not the graded-relevance version the table describes -
      true graded NDCG would need eval_set to carry a relevance score (e.g.
      0-3) per chunk instead of a pass/fail substring check.
    """
    recall = 1 if any(relevant) else 0
    precision = sum(relevant) / len(relevant)

    first_hit_rank = next((rank for rank, is_relevant in enumerate(relevant, start=1) if is_relevant), None)
    reciprocal_rank = 1 / first_hit_rank if first_hit_rank else 0.0

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevant, start=1))
    ideal = sorted(relevant, reverse=True)
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {"recall": recall, "precision": precision, "reciprocal_rank": reciprocal_rank, "ndcg": ndcg}


def evaluate_retrieval(chunks: list[str], chunk_embeddings, eval_set: list[tuple[str, str]], k: int = 3) -> dict[str, float]:
    """
    Retrieve the top-k chunks for every (question, answer) pair ONCE - that's
    the expensive part (an embedding + a similarity comparison) - then derive
    all four metrics from that same result instead of re-retrieving per
    metric, and average each metric across the whole eval set.
    """
    per_question = []
    for question, answer in eval_set:
        top_k_indices = retrieve_top_k(chunks, chunk_embeddings, question, k)
        relevant = [1 if answer.lower() in str(chunks[i]).lower() else 0 for i in top_k_indices]
        per_question.append(_metrics_from_relevance(relevant))

    n = len(per_question)
    return {
        "recall@k": sum(m["recall"] for m in per_question) / n,
        "precision@k": sum(m["precision"] for m in per_question) / n,
        "mrr": sum(m["reciprocal_rank"] for m in per_question) / n,
        "ndcg@k": sum(m["ndcg"] for m in per_question) / n,
    }


def recall_at_k(chunks: list[str], chunk_embeddings, eval_set: list[tuple[str, str]], k: int = 3) -> float:
    """Kept for backward compatibility with earlier notebook cells - now just reads recall@k out of evaluate_retrieval."""
    return evaluate_retrieval(chunks, chunk_embeddings, eval_set, k)["recall@k"]


class _FakeElement:
    """
    Stands in for an `unstructured` Element (what `chunk_by_title` etc.
    actually return) without needing `unstructured` installed just for a
    self-check: has `__str__` but deliberately NO `.lower()`, so if
    evaluate_retrieval ever regresses back to calling `chunks[i].lower()`
    directly instead of `str(chunks[i]).lower()`, this self-check fails loudly
    instead of the bug only surfacing later against real chunk objects (this
    exact bug shipped once already - see sparse_embeddings.py's copy of this
    same evaluate loop, which hit it first).
    """
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


if __name__ == "__main__":
    # Cheap self-check - MiniLM is small (~80MB) and runs locally once
    # downloaded. Confirms embedding + retrieval + evaluate_retrieval all
    # work end to end, using Element-like (non-string) chunks specifically -
    # see _FakeElement above for why.
    demo_chunks = [_FakeElement("Vector Technics builds industrial sensors."), _FakeElement("Q3 revenue grew 40%.")]
    demo_query = "industrial sensors"

    demo_embeddings = embed_chunks(demo_chunks)
    top_k = retrieve_top_k(demo_chunks, demo_embeddings, demo_query, k=1)
    assert top_k == [0], f"expected the sensors chunk to rank first, got index {top_k}"

    demo_eval_set = [("industrial sensors?", "industrial sensors")]
    metrics = evaluate_retrieval(demo_chunks, demo_embeddings, demo_eval_set, k=1)
    assert metrics["recall@k"] == 1.0, f"expected a perfect recall@k on this trivial eval set, got {metrics}"

    print("self-check passed")


