"""
Sparse embeddings: the other half of BGE-m3 that dense_embeddings.py doesn't
use. Where a DENSE embedding is one fixed-length vector of floats (384 or
1024 numbers, every one of them non-zero, capturing "meaning" in a way you
can't read directly), a SPARSE embedding is a dict of {token: weight} - most
possible tokens get weight 0 and are simply left out of the dict, only the
tokens that actually mattered to that sentence show up. This is much closer
to classic keyword search (like BM25) than to semantic search: two chunks
score high here mainly when they share literal words/subwords, weighted by
how important BGE-m3's model decided each word was - not because their
*meaning* is similar despite using different words (that's what the dense
vectors from dense_embeddings.py are for).

Why bother with both? Dense catches paraphrases ("car" ~ "automobile") but
can miss exact needles like a part number or an acronym that never appeared
in training data. Sparse is the opposite: weak on paraphrase, strong on
"this exact token is in here". Hybrid search (combine both scores) is the
common way to get both strengths - not implemented here, this file only
wires up the sparse half so it CAN be combined later.

Loading BGE-m3 through plain sentence-transformers (like dense_embeddings.py
does for its dense vectors) only ever gives you the dense output - sparse and
ColBERT-style multi-vector outputs need FlagEmbedding's own BGEM3FlagModel
class instead, which is a different call shape (see rag-strategy.md, "Using
E5 and BGE-m3"). Needs `pip install FlagEmbedding` (not yet installed).

First run downloads BGE-m3's weights (~2.2GB) and caches them under
~/.cache/huggingface - same model dense_embeddings.py's MODEL_NAME comment
mentions, but FlagEmbedding manages its own copy, so this re-downloads even
if you already pulled BGE-m3 through sentence-transformers.
"""
from FlagEmbedding import BGEM3FlagModel

from dense_embeddings import _metrics_from_relevance

MODEL_NAME = "BAAI/bge-m3"

# use_fp16=True is the usual recommendation (halves memory, small quality
# hit) - but that's a CUDA GPU optimization. On Apple Silicon (MPS) or plain
# CPU, half-precision matmuls aren't reliably supported and can be SLOWER or
# just error out (see rag-strategy.md's Apple Silicon gotchas section for
# the same kind of hardware-specific trap on the dense side). Leave this
# False unless you're running on an actual CUDA GPU.
model = BGEM3FlagModel(MODEL_NAME, use_fp16=False)


def embed_chunks_sparse(chunks: list[str]) -> list[dict]:
    """
    Encode every chunk into a sparse "lexical weights" dict: {token_id:
    weight}. `return_sparse=True` is what turns this on; `return_dense=False`
    and `return_colbert_vecs=False` skip BGE-m3's other two output types
    since this file only wants the sparse one - computing them costs extra
    time for nothing if you're going to throw them away.

    `str(chunk)` (not just `chunk`) for the same reason as dense_embeddings.py:
    `chunk_by_title` and friends hand back `unstructured` Element objects,
    not plain strings, and `str()` is a no-op if it's already a plain string.
    """
    prefixed_chunks = [str(chunk) for chunk in chunks]
    output = model.encode(prefixed_chunks, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    return output["lexical_weights"]


def retrieve_top_k(chunks: list[str], chunk_sparse_weights: list[dict], query: str, k: int = 3) -> list[int]:
    """
    Embed the query into its own lexical-weights dict, then score it against
    every chunk with `model.compute_lexical_matching_score` - this is BGE-m3's
    built-in dot product over shared token ids (weight_a * weight_b summed
    over every token both dicts contain; tokens only one side has contribute
    nothing). No prefix arguments here unlike dense_embeddings.py's
    retrieve_top_k - sparse matching is symmetric, there's no separate
    "passage" vs "query" phrasing trick for it the way E5/BGE's dense side
    has.

    `sorted(..., reverse=True)[:k]` instead of a tensor `.topk()` call like
    the dense version - these scores are a plain Python list of floats, not
    a PyTorch tensor, so this is the plain Python equivalent of the same
    "give me the k biggest, in order" operation.
    """
    query_weights = model.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False)["lexical_weights"][0]
    scores = [model.compute_lexical_matching_score(query_weights, chunk_weights) for chunk_weights in chunk_sparse_weights]
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked_indices[:k]


def evaluate_retrieval_sparse(chunks: list[str], chunk_sparse_weights: list[dict], eval_set: list[tuple[str, str]], k: int = 3) -> dict[str, float]:
    """
    Same shape as dense_embeddings.py's evaluate_retrieval, and reuses its
    `_metrics_from_relevance` (pure math, doesn't care which kind of
    embedding produced the ranking) instead of redefining it - only the
    retrieval call itself differs (sparse's retrieve_top_k above, not
    dense's), so only this loop needs its own copy.
    """
    per_question = []
    for question, answer in eval_set:
        top_k_indices = retrieve_top_k(chunks, chunk_sparse_weights, question, k)
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
    """
    Stands in for an `unstructured` Element (what `chunk_by_title` etc.
    actually return) without needing `unstructured` installed just for a
    self-check: has `__str__` but deliberately NO `.lower()`, so if
    evaluate_retrieval_sparse ever regresses back to calling `chunks[i].lower()`
    directly instead of `str(chunks[i]).lower()`, this self-check fails loudly
    instead of the bug only surfacing later against real chunk objects.
    """
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


if __name__ == "__main__":
    # Cheap self-check - no API key needed (BGE-m3 runs locally once
    # downloaded), just confirms FlagEmbedding is installed, the model
    # loads, and lexical scoring actually favors the chunk sharing the
    # query's exact words over one that doesn't.
    demo_chunks = [_FakeElement("Vector Technics builds industrial sensors."), _FakeElement("Q3 revenue grew 40%.")]
    demo_query = "industrial sensors"

    sparse_weights = embed_chunks_sparse(demo_chunks)
    assert len(sparse_weights) == len(demo_chunks)
    assert all(isinstance(w, dict) and len(w) > 0 for w in sparse_weights)

    top_k = retrieve_top_k(demo_chunks, sparse_weights, demo_query, k=1)
    assert top_k == [0], f"expected the sensors chunk to rank first, got index {top_k}"

    # evaluate_retrieval_sparse is the function that broke on real (non-string)
    # chunks - exercise it here with the same Element-like objects, not plain
    # strings, so this self-check would have caught that bug.
    demo_eval_set = [("industrial sensors?", "industrial sensors")]
    metrics = evaluate_retrieval_sparse(demo_chunks, sparse_weights, demo_eval_set, k=1)
    assert metrics["recall@k"] == 1.0, f"expected a perfect recall@k on this trivial eval set, got {metrics}"

    print("self-check passed")
