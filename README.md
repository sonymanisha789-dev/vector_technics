# Vector Technics RAG POC

A proof-of-concept Retrieval-Augmented Generation (RAG) pipeline: it parses a PDF
into chunks, embeds those chunks (both a "dense" semantic embedding and a
"sparse" keyword-style embedding), stores them in a Postgres database with the
`pgvector` extension, and then answers questions about the PDF by retrieving
the most relevant chunks and asking Claude to answer using only that context.

The easiest way to see it work is [test_new.ipynb](test_new.ipynb) — a short
notebook that stores one PDF, asks it a question, and scores the retrieval
quality.

## 1. Prerequisites

You need three things installed on your machine before anything else:

1. **Python 3.9+** — check with `python3 --version`.
2. **Docker Desktop** — this project stores embeddings in a Postgres database
   that runs inside a Docker container, so you don't have to install Postgres
   yourself. Check with `docker --version`.
3. **An Anthropic API key** — get one at https://console.anthropic.com/. This
   is what pays for the Claude calls that generate answers and (optionally)
   grade retrieval quality.

## 2. Clone the repo and set up a virtual environment

```bash
git clone https://github.com/sonymanisha789-dev/vector_technics.git
cd vector_technics

python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

A virtual environment (`.venv`) is just an isolated folder of Python packages
for this project only, so it can't conflict with other Python projects on
your machine.

## 3. Install the Python packages

```bash
pip install -r requirements.txt
```

This installs everything the scripts and notebook import: the PDF parser
(`unstructured`), the embedding models (`sentence-transformers`,
`FlagEmbedding`), the database driver (`psycopg` + `pgvector`), the Claude
SDK (`anthropic`), the eval library (`ragas`), and Jupyter itself so you can
open the notebook.

The first time you run anything, the embedding models (several hundred MB)
will download automatically from Hugging Face and get cached locally — this
is the "Fetching N files" progress bar you'll see.

## 4. Add your API key

Create a file named `.env` in the project root (same folder as this README)
with:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

This file is already listed in `.gitignore`, so it will never get committed
or pushed to GitHub — keep it that way, never share this key.

## 5. Start the database

This project uses `docker-compose.yml` to spin up a Postgres database with
the `pgvector` extension already installed:

```bash
docker compose up -d
```

`-d` runs it in the background ("detached"). You can confirm it's running
with `docker ps` — you should see a container using the
`pgvector/pgvector:pg17` image. To stop it later, run `docker compose down`
(your data is kept in a Docker volume unless you run `docker compose down -v`).

## 6. Run the notebook

```bash
jupyter notebook test_new.ipynb
```

This opens Jupyter in your browser. Run the cells top to bottom with
Shift+Enter. Here's what each cell does:

1. **Imports** `store`, `query`, `score`, and `score_ragas` from
   [run_rag.py](run_rag.py) — the four functions that do all the real work.
2. **Picks a PDF to store.** It points at
   `rag_project/data/raw/competitors/pdfs/aerodrone.pdf` and sets a
   `doc_name` to identify it in the database. The actual `store(...)` call is
   commented out in this cell — uncomment it the first time you run the
   notebook against a new PDF (it only needs to be stored once; after that,
   it stays in the Postgres database and you can just query it).
3. **Asks a question** about the stored PDF with `query(...)` and prints
   Claude's answer, built only from the top 3 retrieved chunks.
4. **Scores retrieval quality** with `score_ragas(...)` against a small set
   of question/expected-answer pairs, using an LLM judge (`ragas`) to grade
   how well the retrieved chunks support the expected answer.

## 7. Try it with your own PDF (optional)

You don't need the notebook at all — [run_rag.py](run_rag.py) is a
command-line script that does the same thing:

```bash
python run_rag.py store --pdf path/to/your.pdf --doc-name my_doc
python run_rag.py query --doc-name my_doc "Your question about the PDF"
python run_rag.py score --doc-name my_doc
python run_rag.py score-ragas --doc-name my_doc
```

`store` only needs to be run once per document; `query` is fast afterwards
since it reuses what's already in the database.

## Project layout

- [run_rag.py](run_rag.py) — command-line entry point (store/query/score).
- [test_new.ipynb](test_new.ipynb) / [manisha.ipynb](manisha.ipynb) — notebooks for interactive exploration.
- [pgvector_store.py](pgvector_store.py) — Postgres/pgvector read/write layer.
- [dense_embeddings.py](dense_embeddings.py), [sparse_embeddings.py](sparse_embeddings.py), [hybrid_search.py](hybrid_search.py) — embedding and retrieval logic.
- [contextual_chunking.py](contextual_chunking.py) — adds surrounding context to each chunk before embedding.
- [rag_query.py](rag_query.py) — retrieval + Claude answer generation, plus eval scoring.
- [eval_set.py](eval_set.py) — the question/answer pairs used for scoring.
- [rag_project/data/raw/](rag_project/data/raw/) — sample PDFs/competitor research used for testing.
