"""
The last step of RAG: turn a retrieved set of chunks into an actual answer.

Everything up to this point (dense_embeddings.py, sparse_embeddings.py,
hybrid_search.py) only ever produces a RANKING - a list of chunk indices,
best match first. That's retrieval, not an answer. This file takes the
question, retrieves chunks with the best-scoring combo found in
rag-strategy.md's eval (recursive-character chunking + BGE-m3 dense fused
with BGE-m3 sparse via RRF - recall@k=1.0, ndcg@k=0.88), stuffs those chunks
into a prompt, and asks Claude Haiku to write the actual answer, grounded in
only what the chunks say.

Needs: pip install anthropic (already installed - see contextual_chunking.py)
Needs: ANTHROPIC_API_KEY set in the environment (already set).
"""
import anthropic
from dotenv import load_dotenv
from sentence_transformers import util

import pgvector_store
from dense_embeddings import _metrics_from_relevance
from hybrid_search import reciprocal_rank_fusion
from sparse_embeddings import model as sparse_model
from sparse_embeddings import retrieve_top_k as retrieve_top_k_sparse

# override=True - see contextual_chunking.py's identical call for why: a
# local router (e.g. OmniRoute) sets ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL
# globally, and load_dotenv()'s default would leave that in place instead of
# using this project's own key.
load_dotenv(override=True)

MODEL = "claude-haiku-4-5"  # same cheap model contextual_chunking.py uses - fine for answering from a handful of chunks


def retrieve_hybrid(
    query: str,
    chunks: list,
    chunk_dense_embeddings,
    chunk_sparse_weights: list[dict],
    dense_model,
    k: int = 5,
    rrf_k: int = 60,
) -> list[int]:
    """
    Same fusion idea as hybrid_search.py's retrieve_top_k, but NOT tied to
    MiniLM. hybrid_search.py's own retrieve_top_k always embeds the query
    with dense_embeddings.py's module-level model, which is hardcoded to
    MiniLM - fine if chunk_dense_embeddings also came from MiniLM, silently
    WRONG if you hand it BGE-m3 embeddings instead (query and chunks would
    then live in two different, incompatible vector spaces, and cos_sim
    between them would be meaningless). This function takes the model to use
    as an explicit argument instead, so it works with whichever dense model
    actually produced chunk_dense_embeddings - here, BGE-m3, the eval winner.

    Dense ranking: encode the query with dense_model (whatever
    SentenceTransformer instance you pass, e.g. model_dict["BAAI/bge-m3"]
    from the notebook), compare against every chunk's precomputed embedding
    with cos_sim, same pattern as dense_embeddings.py's retrieve_top_k.

    Sparse ranking: sparse_embeddings.py's retrieve_top_k, unchanged - it's
    always BGE-m3 lexical weights regardless of which dense model is used.

    Both rankings are computed FULL (k=len(chunks), not this function's own
    k) before fusing - reciprocal_rank_fusion (from hybrid_search.py, reused
    as-is since it only looks at ranks, not which model produced them) needs
    to know where a chunk landed in the whole order, not just whether it made
    a small top-k cutoff.
    """
    n = len(chunks)
    query_embedding = dense_model.encode(query, normalize_embeddings=True, convert_to_tensor=True)
    dense_scores = util.cos_sim(query_embedding, chunk_dense_embeddings)[0]
    dense_ranking = dense_scores.topk(n).indices.tolist()

    sparse_ranking = retrieve_top_k_sparse(chunks, chunk_sparse_weights, query, k=n)

    return reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=k, rrf_k=rrf_k)


def _answer_from_sources(query: str, sources: list[str], top_k_indices: list[int], client: anthropic.Anthropic, model: str) -> dict:
    """
    The part generate_answer and generate_answer_pgvector share: given
    already-retrieved chunk texts, build the numbered context block and make
    the one Claude call. Pulled out so retrieving from memory vs. from
    Postgres doesn't mean maintaining two copies of the same prompt.

    No prompt caching here (unlike contextual_chunking.py's system block):
    caching pays off when the SAME big block of text is sent over and over in
    a loop (there, the whole document, once per chunk). Here, each call has
    its own small, different context block (whichever chunks THIS question
    retrieved) - there's nothing repeated to cache.
    """
    # Numbered so the model (and a human reading its citations) can point at
    # "[2]" instead of quoting the whole chunk back.
    context_block = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(sources))

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=(
            "Answer the user's question using ONLY the numbered context "
            "below. Cite the sources you used like [1] or [2]. If the "
            "context doesn't contain the answer, say you don't know - never "
            "make up an answer that isn't in the context.\n\n"
            f"{context_block}"
        ),
        messages=[{"role": "user", "content": query}],
    )

    return {
        "answer": response.content[0].text.strip(),
        "source_indices": top_k_indices,
        "sources": sources,
    }


def generate_answer(
    query: str,
    chunks: list,
    chunk_dense_embeddings,
    chunk_sparse_weights: list[dict],
    dense_model,
    client: anthropic.Anthropic = None,
    k: int = 5,
    model: str = MODEL,
) -> dict:
    """
    Retrieve (from in-memory chunks/embeddings), then generate. Returns a
    dict instead of a bare string so the caller can show which chunks the
    answer actually came from (useful for trust/debugging - "why did it say
    that?") without a second retrieval call.
    """
    client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment - never hardcode the key
    top_k_indices = retrieve_hybrid(query, chunks, chunk_dense_embeddings, chunk_sparse_weights, dense_model, k=k)
    sources = [str(chunks[i]) for i in top_k_indices]
    return _answer_from_sources(query, sources, top_k_indices, client, model)


def retrieve_hybrid_pgvector(
    conn,
    doc_name: str,
    model_name: str,
    query: str,
    dense_model,
    k: int = 5,
    rrf_k: int = 60,
) -> tuple[list[int], dict[int, str]]:
    """
    Same fusion as retrieve_hybrid above, but the chunks + embeddings come
    from Postgres (pgvector_store.py) instead of in-memory dicts - so this
    works as long as store_all_embeddings/store_all_sparse_embeddings have
    been run once, without needing chunks_dict/chunk_embeddings_dict/
    sparse_embeddings_dict to still be sitting in memory (e.g. a fresh
    notebook kernel, or a separate script entirely).

    Dense ranking: pgvector_store.fetch_dense_ranking already does the
    ranking IN SQL (ORDER BY embedding <=> query - pgvector's cosine
    distance operator), so there's no local cos_sim call here at all, unlike
    retrieve_hybrid above.

    Sparse ranking: pgvector_store.fetch_sparse_weights only returns the raw
    {token: weight} dicts (unranked) - scoring them against the query still
    needs BGE-m3's own compute_lexical_matching_score, so that part happens
    here, same math as sparse_embeddings.py's retrieve_top_k.

    Returns (fused chunk indices, {chunk_index: chunk_text}) - the text
    lookup comes for free from the two fetch calls, so callers (e.g.
    generate_answer_pgvector) don't need a third query just to get chunk text.
    """
    query_embedding = dense_model.encode(query, normalize_embeddings=True).tolist()
    dense_rows = pgvector_store.fetch_dense_ranking(conn, doc_name, model_name, query_embedding)
    dense_ranking = [chunk_index for chunk_index, _ in dense_rows]

    sparse_rows = pgvector_store.fetch_sparse_weights(conn, doc_name)
    query_weights = sparse_model.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False)["lexical_weights"][0]
    scored = [
        (chunk_index, sparse_model.compute_lexical_matching_score(query_weights, weights))
        for chunk_index, _, weights in sparse_rows
    ]
    sparse_ranking = [chunk_index for chunk_index, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)]

    fused_indices = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=k, rrf_k=rrf_k)
    chunk_text_by_index = {chunk_index: text for chunk_index, text in dense_rows}
    return fused_indices, chunk_text_by_index


def generate_answer_pgvector(
    conn,
    doc_name: str,
    model_name: str,
    query: str,
    dense_model,
    client: anthropic.Anthropic = None,
    k: int = 5,
    model: str = MODEL,
) -> dict:
    """
    Same as generate_answer, but retrieves from Postgres via
    retrieve_hybrid_pgvector instead of in-memory embeddings - the "real"
    query-time path once chunks/embeddings are already stored, no need to
    re-chunk or re-embed the whole document just to answer one question.
    """
    client = client or anthropic.Anthropic()
    top_k_indices, chunk_text_by_index = retrieve_hybrid_pgvector(conn, doc_name, model_name, query, dense_model, k=k)
    sources = [chunk_text_by_index[i] for i in top_k_indices]
    return _answer_from_sources(query, sources, top_k_indices, client, model)


def score_pgvector(
    conn,
    doc_name: str,
    model_name: str,
    eval_set: list[tuple[str, str]],
    dense_model,
    k: int = 5,
    rrf_k: int = 60,
) -> dict[str, float]:
    """
    Same shape as hybrid_search.py's evaluate_retrieval (and reuses
    dense_embeddings.py's _metrics_from_relevance for the same
    recall@k/precision@k/mrr/ndcg@k math) - only the retrieval call differs:
    this one goes through retrieve_hybrid_pgvector, so it scores whatever a
    real `run_rag.py store` run actually put in Postgres, not a copy of
    chunks/embeddings still sitting in memory.
    """
    per_question = []
    for question, answer in eval_set:
        top_k_indices, chunk_text_by_index = retrieve_hybrid_pgvector(conn, doc_name, model_name, question, dense_model, k=k, rrf_k=rrf_k)
        relevant = [1 if answer.lower() in chunk_text_by_index[i].lower() else 0 for i in top_k_indices]
        per_question.append(_metrics_from_relevance(relevant))

    n = len(per_question)
    return {
        "recall@k": sum(m["recall"] for m in per_question) / n,
        "precision@k": sum(m["precision"] for m in per_question) / n,
        "mrr": sum(m["reciprocal_rank"] for m in per_question) / n,
        "ndcg@k": sum(m["ndcg"] for m in per_question) / n,
    }


def score_ragas_pgvector(
    conn,
    doc_name: str,
    model_name: str,
    eval_set: list[tuple[str, str]],
    dense_model,
    k: int = 5,
    rrf_k: int = 60,
    judge_model: str = MODEL,
) -> dict[str, float]:
    """
    LLM-judged alternative to score_pgvector - use this when eval_set's
    answers are natural-language sentences (like "AeroDrone is based in
    Kyiv, Ukraine.") instead of exact substrings copied from the source text.
    score_pgvector's `answer.lower() in chunk.lower()` check can never match
    a paraphrase, no matter how good retrieval is - it has no concept of
    meaning. This asks an LLM judge instead: "given this question and this
    reference answer, is the retrieved context relevant to / sufficient to
    answer it?" (ragas's ContextPrecision / ContextRecall metrics).

    Same retrieval path as score_pgvector (retrieve_hybrid_pgvector) - only
    the grading differs. Costs one Claude call per question per metric
    (2 metrics here), and unlike score_pgvector's scores, these aren't
    perfectly reproducible run to run since an LLM is doing the judging.

    Needs: pip install ragas (already installed).
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import ContextPrecision, ContextRecall

    samples = []
    for question, reference_answer in eval_set:
        top_k_indices, chunk_text_by_index = retrieve_hybrid_pgvector(conn, doc_name, model_name, question, dense_model, k=k, rrf_k=rrf_k)
        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=[chunk_text_by_index[i] for i in top_k_indices],
            reference=reference_answer,
        ))

    judge_llm = llm_factory(judge_model, provider="anthropic", client=anthropic.Anthropic())
    # ragas's InstructorModelArgs defaults to BOTH temperature and top_p set -
    # fine for OpenAI, but Anthropic's API rejects a request that specifies
    # both ("temperature and top_p cannot both be specified for this model").
    # llm_factory has no kwarg to omit one, so drop top_p from the already-
    # built dict directly.
    judge_llm.model_args.pop("top_p", None)
    metrics = [ContextPrecision(), ContextRecall()]
    result = evaluate(dataset=EvaluationDataset(samples=samples), metrics=metrics, llm=judge_llm)

    # Average each metric's per-question scores ourselves (same style as
    # score_pgvector above) instead of relying on EvaluationResult's private
    # _repr_dict.
    return {metric.name: sum(result[metric.name]) / len(result[metric.name]) for metric in metrics}


if __name__ == "__main__":
    # Cheap self-check - reuses models already downloaded by dense_embeddings.py
    # (MiniLM, tiny) and sparse_embeddings.py (BGE-m3, already cached if you've
    # run that file before). Not the production BGE-m3-dense combo from
    # rag-strategy.md - that's a deliberate simplification, this only needs to
    # prove the retrieve -> prompt -> real Haiku call plumbing works, not
    # reproduce the exact eval-winning config.
    from dense_embeddings import model as minilm_model
    from sparse_embeddings import embed_chunks_sparse

    demo_chunks = [
        "Vector Technics builds industrial sensors for manufacturing lines.",
        "Q3 revenue grew 40% year over year, driven by the new sensor line.",
    ]
    demo_dense_embeddings = minilm_model.encode(demo_chunks, normalize_embeddings=True, convert_to_tensor=True)
    demo_sparse_weights = embed_chunks_sparse(demo_chunks)

    result = generate_answer(
        "What does Vector Technics make?",
        demo_chunks,
        demo_dense_embeddings,
        demo_sparse_weights,
        dense_model=minilm_model,
        k=1,
    )
    print(result["answer"])

    assert result["source_indices"] == [0], f"expected the sensors chunk to be retrieved, got {result['source_indices']}"
    assert "sensor" in result["answer"].lower(), f"expected the answer to mention sensors, got: {result['answer']}"

    print("self-check passed (in-memory)")

    # Same check again, but round-tripped through Postgres first - proves
    # retrieve_hybrid_pgvector/generate_answer_pgvector work, not just the
    # in-memory path above. Needs `docker compose up -d` running first (same
    # requirement as pgvector_store.py's own self-check).
    conn = pgvector_store.get_connection()
    pgvector_store.init_db(conn)
    pgvector_store.store_embeddings(conn, "_selftest_doc", "minilm", demo_chunks, demo_dense_embeddings)
    pgvector_store.store_sparse_embeddings(conn, "_selftest_doc", demo_chunks, demo_sparse_weights)

    result_pgvector = generate_answer_pgvector(
        conn,
        "_selftest_doc",
        "minilm",
        "What does Vector Technics make?",
        dense_model=minilm_model,
        k=1,
    )
    print(result_pgvector["answer"])

    assert result_pgvector["source_indices"] == [0], f"expected the sensors chunk to be retrieved, got {result_pgvector['source_indices']}"
    assert "sensor" in result_pgvector["answer"].lower(), f"expected the answer to mention sensors, got: {result_pgvector['answer']}"

    print("self-check passed (pgvector)")

    # score_pgvector check - one question per demo chunk, each answer only
    # appears in its own chunk, so a working hybrid retrieval should get a
    # perfect score on all four metrics at k=1.
    demo_eval_set = [
        ("What does Vector Technics make?", "sensors"),
        ("How much did Q3 revenue grow?", "40%"),
    ]
    metrics = score_pgvector(conn, "_selftest_doc", "minilm", demo_eval_set, dense_model=minilm_model, k=1)
    for name in ("recall@k", "precision@k", "mrr", "ndcg@k"):
        assert metrics[name] == 1.0, f"expected a perfect {name} on this trivial eval set, got {metrics}"

    print("self-check passed (score_pgvector)")

    # score_ragas_pgvector check - same demo docs/questions, but graded by an
    # LLM judge instead of substring matching. Loose >= 0.5 threshold (not
    # == 1.0 like above) because an LLM judge's score isn't perfectly
    # deterministic - this only needs to catch the wiring being broken, not
    # pin an exact score. Costs a handful of real Haiku calls.
    ragas_metrics = score_ragas_pgvector(conn, "_selftest_doc", "minilm", demo_eval_set, dense_model=minilm_model, k=1)
    for name, value in ragas_metrics.items():
        assert value >= 0.5, f"expected a clearly-relevant-context score for {name} on this trivial eval set, got {ragas_metrics}"

    print("self-check passed (score_ragas_pgvector)")

    conn.execute("DELETE FROM chunk_embeddings WHERE doc_name = %s", ("_selftest_doc",))
    conn.execute("DELETE FROM chunk_sparse_weights WHERE doc_name = %s", ("_selftest_doc",))
