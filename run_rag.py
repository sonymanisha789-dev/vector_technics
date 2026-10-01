"""
Standalone command-line script for this RAG project - run this instead of
opening manisha.ipynb every time you want to add a document or ask it a
question.

It has two commands:

  store  - parse a PDF, chunk it (unstructured's chunk_by_title, same
           chunker the notebook uses), embed it (dense model of your choice
           + BGE-m3 sparse), and save the chunks + embeddings into the
           Postgres/pgvector database (pgvector_store.py).

  query  - ask a question about a document ALREADY stored in pgvector.
           Nothing gets re-parsed or re-embedded except your one question -
           this is the fast, "real" query path (rag_query.py's
           generate_answer_pgvector), the same function the notebook's very
           last cells call.

  score  - score retrieval quality (recall@k, precision@k, mrr, ndcg@k) for
           a document ALREADY stored in pgvector, against eval_set.py's
           EVAL_SET. Same retrieval path as `query` (rag_query.py's
           score_pgvector, built on retrieve_hybrid_pgvector) - no separate
           in-memory eval to keep in sync. Grading is exact substring match,
           so eval_set answers must be exact text copied from the source
           document, not paraphrased sentences - see eval_set.py's docstring.

  score-ragas - same idea as `score`, but graded by an LLM judge (ragas's
           ContextPrecision/ContextRecall) instead of substring matching.
           Use this when your eval_set answers are natural-language
           sentences instead of exact substrings - substring matching can't
           grade those correctly no matter how good retrieval is. Costs a
           Claude call per question per metric and scores aren't perfectly
           reproducible run to run.

Setup this script assumes is already done (same as the notebook needs):
  - `docker compose up -d`            (starts the Postgres/pgvector container)
  - ANTHROPIC_API_KEY set in .env     (used to generate the final answer)

Examples:
  python run_rag.py store --pdf 6b5330ba3c12ba40.pdf --doc-name my_doc
  python run_rag.py query --doc-name my_doc "How much did Raphe mPhibr raise in its Series B?"
  python run_rag.py score --doc-name my_doc
  python run_rag.py score-ragas --doc-name my_doc
"""
import argparse

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

import pgvector_store
from contextual_chunking import add_context_to_chunks
from eval_set import EVAL_SET
from rag_query import generate_answer_pgvector, score_pgvector, score_ragas_pgvector
from sparse_embeddings import embed_chunks_sparse

load_dotenv(override=True)  # override=True so this project's own .env key wins over a router like OmniRoute setting ANTHROPIC_API_KEY globally - see rag_query.py's identical call

# Best-scoring combo from rag-strategy.md's eval (recall@k=1.0) - used unless
# --model overrides it.
DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def embed_dense(chunks: list[str], model_name: str):
    """Load one dense model by name and embed every chunk with it - same
    normalize_embeddings/max_seq_length choices dense_embeddings.py uses,
    just parameterized to whichever model this run picked."""
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512
    embeddings = model.encode(chunks, normalize_embeddings=True, convert_to_tensor=True)
    return model, embeddings


def store(pdf_path: str, doc_name: str, model_name: str, contextualize: bool) -> None:
    print(f"parsing {pdf_path} ...")
    elements = partition_pdf(filename=pdf_path)
    chunks = [str(c) for c in chunk_by_title(elements, max_characters=2500, new_after_n_chars=1000, overlap=200)]
    print(f"{len(chunks)} chunks")

    if contextualize:
        full_text = "\n\n".join(str(el) for el in elements)
        print("adding Claude context blurbs to each chunk (one API call per chunk)...")
        chunks = add_context_to_chunks(full_text, chunks)

    print(f"embedding with {model_name} ...")
    _, dense_embeddings = embed_dense(chunks, model_name)
    sparse_weights = embed_chunks_sparse(chunks)

    conn = pgvector_store.get_connection()
    pgvector_store.init_db(conn)
    pgvector_store.store_embeddings(conn, doc_name, model_name, chunks, dense_embeddings)
    pgvector_store.store_sparse_embeddings(conn, doc_name, chunks, sparse_weights)
    print(f"stored {len(chunks)} chunks under doc_name={doc_name!r}, model_name={model_name!r}")


def query(doc_name: str, model_name: str, question: str, k: int) -> None:
    print(f"loading {model_name} to embed the question ...")
    dense_model = SentenceTransformer(model_name)
    dense_model.max_seq_length = 512

    conn = pgvector_store.get_connection()
    result = generate_answer_pgvector(conn, doc_name, model_name, question, dense_model=dense_model, k=k)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nRetrieved chunk indices: {result['source_indices']}")


def score(doc_name: str, model_name: str, k: int, eval_set: list[tuple[str, str]] = EVAL_SET) -> None:
    """eval_set defaults to eval_set.py's EVAL_SET, but any list of
    (question, answer_substring) pairs works - same format:
    [("question", "answer"), ...]."""
    print(f"loading {model_name} to embed {len(eval_set)} eval questions ...")
    dense_model = SentenceTransformer(model_name)
    dense_model.max_seq_length = 512

    conn = pgvector_store.get_connection()
    metrics = score_pgvector(conn, doc_name, model_name, eval_set, dense_model, k=k)

    print(f"\nScored {len(eval_set)} questions against doc_name={doc_name!r}, model_name={model_name!r}, k={k}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def score_ragas(doc_name: str, model_name: str, k: int, eval_set: list[tuple[str, str]] = EVAL_SET) -> None:
    """Same as score(), but LLM-judged (rag_query.py's score_ragas_pgvector)
    instead of substring-matched - use when eval_set's answers are
    natural-language sentences rather than exact substrings from the source
    document."""
    print(f"loading {model_name} to embed {len(eval_set)} eval questions ...")
    dense_model = SentenceTransformer(model_name)
    dense_model.max_seq_length = 512

    conn = pgvector_store.get_connection()
    metrics = score_ragas_pgvector(conn, doc_name, model_name, eval_set, dense_model, k=k)

    print(f"\nRagas-scored {len(eval_set)} questions against doc_name={doc_name!r}, model_name={model_name!r}, k={k}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_store = sub.add_parser("store", help="chunk + embed a PDF and save it into pgvector")
    p_store.add_argument("--pdf", required=True, help="path to the PDF to ingest")
    p_store.add_argument("--doc-name", required=True, help="name to store this document under (the lookup key you'll use at query time)")
    p_store.add_argument("--model", default=DEFAULT_MODEL_NAME, help=f"dense embedding model (default: {DEFAULT_MODEL_NAME})")
    p_store.add_argument("--contextualize", action="store_true", help="add a Claude-generated context blurb to each chunk before embedding (1 API call per chunk; this was the best-scoring setup in eval)")

    p_query = sub.add_parser("query", help="ask a question against a document already stored in pgvector")
    p_query.add_argument("question", help="the question to ask")
    p_query.add_argument("--doc-name", required=True, help="doc_name it was stored under")
    p_query.add_argument("--model", default=DEFAULT_MODEL_NAME, help=f"dense model used when storing - must match (default: {DEFAULT_MODEL_NAME})")
    p_query.add_argument("--k", type=int, default=5, help="how many chunks to retrieve (default: 5)")

    p_score = sub.add_parser("score", help="score retrieval quality (recall@k/precision@k/mrr/ndcg@k) for a document already stored in pgvector, against eval_set.py's EVAL_SET")
    p_score.add_argument("--doc-name", required=True, help="doc_name it was stored under")
    p_score.add_argument("--model", default=DEFAULT_MODEL_NAME, help=f"dense model used when storing - must match (default: {DEFAULT_MODEL_NAME})")
    p_score.add_argument("--k", type=int, default=5, help="how many chunks to retrieve per question (default: 5)")

    p_score_ragas = sub.add_parser("score-ragas", help="LLM-judged retrieval scoring (ragas ContextPrecision/ContextRecall) for a document already stored in pgvector - use for eval sets with natural-language answers instead of exact substrings")
    p_score_ragas.add_argument("--doc-name", required=True, help="doc_name it was stored under")
    p_score_ragas.add_argument("--model", default=DEFAULT_MODEL_NAME, help=f"dense model used when storing - must match (default: {DEFAULT_MODEL_NAME})")
    p_score_ragas.add_argument("--k", type=int, default=5, help="how many chunks to retrieve per question (default: 5)")

    args = parser.parse_args()

    if args.command == "store":
        store(args.pdf, args.doc_name, args.model, args.contextualize)
    elif args.command == "query":
        query(args.doc_name, args.model, args.question, args.k)
    elif args.command == "score":
        score(args.doc_name, args.model, args.k)
    elif args.command == "score-ragas":
        score_ragas(args.doc_name, args.model, args.k)


if __name__ == "__main__":
    main()
