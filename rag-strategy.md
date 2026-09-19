https://github.com/Danielskry/Awesome-RAG. --use this link to refer various strategy
https://www.reddit.com/r/Rag/comments/1gcf39v/comparative_analysis_of_chunking_strategies_which/#lightbox --chunking strategy

## Environment

- Venv: `python3 -m venv .venv` in project root. Activate: `source .venv/bin/activate`.
- Jupyter kernel: `pip install ipykernel && python -m ipykernel install --user --name=vector-rag-venv`. Select it in the notebook's kernel picker.
- `unstructured` needs extras for PDF/xlsx: `pip install "unstructured[pdf,xlsx]"`.
  - Requires system lib `libheif` for the `pi-heif` dependency: `brew install libheif`.
  - `pi-heif` 1.4.0 uses Python 3.10+ syntax and breaks on Python 3.9 — pin `pip install "pi-heif==0.22.0"` if on 3.9.
  - Also needs `poppler` (provides `pdftoppm`) — `brew install poppler` if missing.
- `nltk` + its `punkt_tab` tokenizer data come pre-installed as a transitive dep of `unstructured` — no extra install needed for sentence tokenization.
- `langchain-text-splitters` installed standalone (not full `langchain`) for `RecursiveCharacterTextSplitter` / `NLTKTextSplitter`.

## Chunking strategies (cheapest → most complex)

1. **Fixed-size** — split every N chars with overlap. Simple, ignores structure.
2. **Recursive character** (`RecursiveCharacterTextSplitter`) — tries separators `["\n\n","\n"," ",""]` in order, only falling back to a cruder split when a piece is still oversized. Good general default.
3. **Sentence-based** (`NLTKTextSplitter`, or `nltk.sent_tokenize` by hand) — never cuts mid-sentence. `chunk_size`/`chunk_overlap` are in **characters**, not words.
4. **Structure-aware** (`unstructured.chunking.title.chunk_by_title`) — groups elements until the next `Title` element, so boundaries align to real document sections (uses layout info `unstructured` already extracted — font size/position — not guessed from text). **Best fit here** since PDFs are already parsed into typed elements via `partition_pdf`.
5. **Semantic** (embed sentences, cut at similarity drops) — costs an embedding call per sentence just to chunk. Not set up yet.
6. **Agentic/LLM-based** — most accurate, most expensive. Not needed at POC stage.

## Chunk size / overlap guidance

- Size in **tokens**, not characters: ~4 chars/token, ~0.75 tokens/word in English.
- Sweet spot: **500–1024 tokens** (~2000–4000 characters) for granular retrieval. Below 500 loses context, above 1024 mixes topics and hurts retrieval precision.
- Overlap: **10–20% of chunk_size**, expressed as a percentage so it scales when chunk_size changes (a fixed absolute overlap doesn't).
- Can estimate a starting chunk size from the document itself: take the median word-count of its structural elements (paragraphs/sections from `unstructured`), multiply by ~4 (a few paragraphs per chunk), convert words→tokens (`/0.75`). This gives a *starting point*, not the final answer.
- **Watch units**: a token estimate from that formula is NOT a word count — plugging a token number into a `max_words` parameter inflates real chunk size by ~33%.
- Goal is not "fewer chunks" or "smaller chunk_size" as ends in themselves — both are just side effects of the chunk_size dial, in tension with each other (bigger chunks = cheaper but noisier retrieval; smaller = sharper retrieval but more storage/compute and less standalone context). The actual goal is retrieval quality (recall@k / answer correctness) on a real eval set, at acceptable cost. If two sizes tie on quality, prefer the cheaper (larger/fewer-chunks) one.

## Evaluating chunking strategies

No embedding model wired up yet, so use a stdlib-only word-overlap cosine retriever (`retrieve_top_k`/`recall_at_k`, see notebook) as a cheap stand-in for comparing strategies *relatively*. Build a small eval set of ~15-30 `(question, expected_answer_substring)` pairs from the real docs, sweep chunk_size/overlap combinations, measure recall@k, pick the winner. Swap in real embeddings (sentence-transformers/OpenAI) once retrieval quality needs to be exact, not just relatively ranked.

## Open TODOs

- No embedding model or vectorstore chosen yet.
- Eval set of real questions not yet built.
- Example `max_characters=3000 / new_after_n_chars=2500 / overlap=300` values for `chunk_by_title` are illustrative defaults from the guidance above, not tuned against this project's actual documents — run through `recall_at_k` before trusting them.
  - Of these, `max_characters` and `overlap` are derived from the token/percentage guidance above; `new_after_n_chars=2500` is a guess (~83% of `max_characters`, a soft buffer) with no principled derivation — don't treat all three as equally justified.

## More chunking strategies

- **Paragraph-based** — group `unstructured` elements (already split into paragraphs) until a `max_chars` ceiling, join with `"\n\n"`. Simpler than `chunk_by_title` since it ignores Title elements entirely — no section awareness, just paragraph boundaries.
- **Semantic chunking** (`langchain_experimental.text_splitter.SemanticChunker`) — embeds each sentence, computes cosine *distance* between consecutive sentence embeddings, and cuts a chunk boundary wherever that distance spikes above a threshold (default: 95th percentile of all distances in the doc — i.e. only the most dramatic topic shifts become boundaries). Requires an embeddings object (works fully offline/free with a local HuggingFace model, no API key).
  - Key params: `embeddings` (required), `breakpoint_threshold_type` (`percentile` default / `standard_deviation` / `interquartile` / `gradient`), `breakpoint_threshold_amount` (default 95 for percentile — lower = more boundaries = more/smaller chunks), `buffer_size` (sentences grouped before embedding, smooths noise), `min_chunk_size`, `number_of_chunks` (target count directly instead of a threshold).
  - On our real PDF (67 sentences via `(?<=[.?!])\s+`) the default 95th-percentile threshold produced 5 chunks — only the biggest topic-shift jumps in the distance sequence cleared that bar. Lowering `breakpoint_threshold_amount` (e.g. to 80) would produce more, smaller chunks.
  - Visualized this by plotting sentence-pair index vs. cosine distance, a horizontal line at the percentile threshold, and markers at the sentences that crossed it — the crossings are exactly the chunk boundaries. See `dataviz` skill for the plotting method if regenerating.

## Embedding models (< 1GB, local/offline via HuggingFace)

| Model | Size | Dims | Notes |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | ~80MB | 384 | Smallest, fastest, good default for a POC |
| `BAAI/bge-small-en-v1.5` | ~127MB | 384 | Slightly better quality than MiniLM at similar size |
| `BAAI/bge-base-en-v1.5` | ~416MB | 768 | Sweet spot for quality vs. size if MiniLM's recall is too low |
| `sentence-transformers/all-mpnet-base-v2` | ~420-438MB | 768 | Strong general-purpose alternative to bge-base |
| `BAAI/bge-m3` | ~1.1-2.2GB (fp16/fp32) | 1024 | Over the 1GB bar — multilingual, 8192-token context. **Not a chunking strategy** — it's an embedding model, used *after* chunking regardless of which chunking strategy is picked. |

## Context-aware chunking

Not a boundary-selection method — a cross-cutting technique layered on top of *any* chunker (fixed-size, sentence, paragraph, structure-aware). Two flavors:

1. **Prepend context to the chunk text before embedding** — cheapest version: prepend the section title (`chunk_with_title_context` — tracks the last-seen `Title` element from `unstructured` and glues `"Section: {title}\n\n{chunk text}"` before embedding). Most powerful version: have an LLM write a 1-2 sentence blurb situating the chunk in the whole document (Anthropic's "Contextual Retrieval" — see below).
2. **Parent-child retrieval** (`ParentDocumentRetriever`) — embed and search over small chunks for precision, but hand the LLM the larger parent chunk they came from for full context at generation time. Not yet implemented here.

Caveat found while building `chunk_with_title_context` on the real PDF: `unstructured`'s `"auto"` PDF strategy detects `Title` elements by font-size heuristics, and it missed at least one visually-obvious heading in our doc — so title-based context isn't 100% reliable and should be spot-checked, not trusted blindly.

## Contextual chunking with Claude Haiku (LLM-generated context)

Implemented in [contextual_chunking.py](contextual_chunking.py) — `add_context_to_chunks(full_document, chunks)`. For each chunk, calls `claude-haiku-4-5` once, asking it to write a short blurb situating that chunk within the whole document, then prepends the blurb to the chunk text before embedding.

- Requires `pip install anthropic` (done — venv has 0.125.0, the latest available for Python 3.9) and `ANTHROPIC_API_KEY` set in the environment (already set).
- **Prompt caching is essential here**: the whole document is sent as a `system` block with `cache_control: {"type": "ephemeral"}`. Since the document text is identical on every call in the loop, only the first call pays full price to write it to cache — every call after reads it back at ~10% of input price. Without this, contextualizing N chunks means re-sending the whole document N times at full price.
- `response.usage.cache_creation_input_tokens` / `cache_read_input_tokens` — printed per chunk to confirm the cache is actually hitting (chunk 0: cache_creation > 0, cache_read = 0; every chunk after: the reverse).
- Rough cost estimate for our real PDF (~7,300 tokens, ~22 chunks): ~$0.05–$0.10 total — back-of-envelope from the token/pricing math, not a measured run.
- Deliberately not run against the real PDF yet — it's a paid API call. The file's `if __name__ == "__main__"` block is a cheap self-check (fake 2-sentence doc, 2 chunks, fraction of a cent) to confirm the key + caching setup work before pointing it at the real document.

## Dense embeddings + retrieval (implemented)

`dense_embeddings.py` — uses `sentence-transformers` directly (no LangChain wrapper needed, it's just vectors + cosine similarity).

- `embed_chunks(chunks, passage_prefix="")` — encodes a list of chunks into unit-length vectors (`normalize_embeddings=True`, so cosine similarity = a plain dot product). Uses `str(chunk)` internally, not `chunk` directly — `chunk_by_title` and other `unstructured`-based chunkers return `Element` objects, not plain strings (unlike the LangChain splitters, which return `list[str]`); `str()` is a no-op on an already-plain string and pulls the text out of an `Element`.
- `retrieve_top_k(chunks, chunk_embeddings, query, k=3, query_prefix="")` — embeds the query the same way, ranks every chunk by `util.cos_sim`, returns the top-k chunk positions.
- `model.max_seq_length = 512` is set right after loading the model — caps how many tokens any single input can ever be. No-op for MiniLM (already truncates at 256), but see "Gotchas on Apple Silicon" below for why this matters once you swap in E5/BGE-m3.

## Retrieval evaluation metrics (implemented)

Also in `dense_embeddings.py`: `_metrics_from_relevance(relevant)` and `evaluate_retrieval(chunks, chunk_embeddings, eval_set, k)`.

- `_metrics_from_relevance` is the per-question calculator — given a list of 0/1 hit labels for one question's already-retrieved top-k chunks (e.g. `[0, 1, 0]`), it computes recall/precision/reciprocal_rank/ndcg for just that question. Pure math, no embedding model involved — lets the metric formulas be unit-tested with hand-picked lists instead of a live model.
- `evaluate_retrieval` is the runner for the whole eval set — it's the one that actually retrieves (the expensive part): for every `(question, answer)` pair it embeds the question, gets the top-k chunks, checks which ones contain the answer, hands that 0/1 list to `_metrics_from_relevance`, then **averages every metric across all questions** into the final `recall@k`/`precision@k`/`mrr`/`ndcg@k`.
- `recall_at_k` is now a thin wrapper reading `recall@k` back out of `evaluate_retrieval` — kept only so earlier notebook cells calling it don't break.
- All 4 metrics are 0–1, **higher is always better** for every one of them — none are an error-rate-style "lower is better" metric:
  - recall@k: was the right chunk anywhere in the top-k at all? The main signal — did retrieval find it, period.
  - precision@k: what fraction of the top-k chunks were actually relevant (a noise check).
  - mrr: how high did the first hit rank (1st place scores 1.0, 5th place scores 0.2).
  - ndcg@k: rank-discounted hit quality, normalized against the best possible ordering of the same hits.
  - Priority order for a POC: watch recall@k first. Only once it's already decent do mrr/precision@k/ndcg become useful — they're for telling apart two models or chunking strategies that both "find" the answer but rank or surround it differently.

## Eval set for the real document (built)

`eval_set.py` — `EVAL_SET`, a list of `(question, answer_substring)` tuples for `6b5330ba3c12ba40.pdf` (the Raphe mPhibr investment report), ready to pass straight into `evaluate_retrieval`/`recall_at_k`.

- Written as an outsider reviewer: every question came from reading the PDF's raw text (`pdftotext -layout`) directly, with no look at how the notebook actually chunks or embeds it — avoids biasing questions toward convenient chunk boundaries or a specific embedding model's quirks, which would make recall@k look better than retrieval actually is.
- 28 questions spread across every major section (exec summary, company/product specs, funding round, Indian competitors, global competitors, market figures, government policy, risks, conclusion) so no one section is over- or under-tested.
- Gotcha #1 — ligature mangling: this PDF's font drops "fi"/"fl"/"tf" on extraction ("profit" extracts as "prot", "flight" as "ight"). Every `answer_substring` was picked to dodge a mangled word, so a miss reflects a retrieval problem, not an extraction artifact.
- Gotcha #2 — line-wrap: `pdftotext -layout` sometimes splits a fact across a line break (e.g. "$6.26" and "billion" landed on separate lines, so "$6.26 billion" wasn't literally one string in the extracted text). Caught by the file's own self-check (`python eval_set.py`), which re-extracts the PDF and asserts every `answer_substring` is actually present — re-run it any time the list is edited.

## Using E5 and BGE-m3 (via dense_embeddings.py)

Both work through the same `embed_chunks`/`retrieve_top_k` functions above — just change `MODEL_NAME` and pass the right prefixes:

| Model | passage_prefix | query_prefix |
|---|---|---|
| MiniLM (current default) | `""` | `""` |
| BGE (small/base/large, not m3) | `""` | `"Represent this sentence for searching relevant passages: "` |
| E5 (any size, incl. multilingual-e5-*) | `"passage: "` | `"query: "` |
| BGE-m3 (dense mode only, via sentence-transformers) | `""` | `""` |

- E5's trap: unlike BGE, it needs the prefix on **both** sides. Skipping the passage side looks optional by analogy with BGE — it isn't, for E5, and recall degrades silently (no error, just worse matches).
- BGE-m3 is actually 3 embedding types in one model: dense (a single vector), sparse (BM25-style keyword weights), and multi-vector (ColBERT-style, one vector per token). Loading it through `SentenceTransformer` the way this file does only gets the dense vector — a fine MiniLM drop-in with zero code changes beyond `MODEL_NAME`. Getting the sparse or multi-vector outputs needs the separate `FlagEmbedding` library's `BGEM3FlagModel` class instead — a different call shape, not implemented here.
- BGE-m3 is ~2.2GB — the only model in the embedding-models table above over the "<1GB, local/offline" bar. Only reach for it if MiniLM's recall on the real eval set is too low.

## Sparse embeddings (implemented)

`sparse_embeddings.py` — uses `FlagEmbedding`'s `BGEM3FlagModel` directly (plain `sentence-transformers`, what `dense_embeddings.py` uses, only exposes BGE-m3's dense vectors — sparse needs this separate library/class). Needs `pip install FlagEmbedding` (not yet installed) and re-downloads BGE-m3's ~2.2GB weights separately from any copy already cached via `sentence-transformers`.

- A sparse embedding is a `{token_id: weight}` dict, not a fixed-length vector — most tokens get weight 0 and are simply absent. Closer to keyword search (BM25-like) than to the semantic similarity dense vectors capture: scores high mainly on shared literal words/subwords, weighted by learned importance.
- `embed_chunks(chunks)` — `model.encode(..., return_dense=False, return_sparse=True, return_colbert_vecs=False)`, returns the `lexical_weights` list. `retrieve_top_k(chunks, chunk_sparse_weights, query, k=3)` — scores every chunk against the query with `model.compute_lexical_matching_score` (dot product over shared token ids) and sorts. No query/passage prefix split unlike dense's E5/BGE table — sparse matching is symmetric.
- Reuses `_metrics_from_relevance` from `dense_embeddings.py` for the recall/precision/mrr/ndcg math (pure, model-agnostic); only the retrieval loop is duplicated since it calls a different `retrieve_top_k`.
- `use_fp16=False` — the usual `use_fp16=True` speedup is a CUDA GPU optimization; half-precision matmuls aren't reliably supported on Apple Silicon/CPU (same flavor of hardware trap as the MPS gotchas below).
- Run against the real eval set (`manisha.ipynb`, cell building `sparse_embeddings_dict`/`sparse_all_results`): recall@k ranged 0.86-0.96 across the 6 chunking strategies, best was `recursive_character_chunks_doc1` (recall@k 0.96, ndcg@k 0.87).

## Hybrid search (implemented)

`hybrid_search.py` — combines `dense_embeddings.py`'s MiniLM ranking and `sparse_embeddings.py`'s BGE-m3-sparse ranking into one ranking via **Reciprocal Rank Fusion (RRF)**, the production default (Elasticsearch/OpenSearch/Weaviate all use it) — not a weighted score average.

- Why RRF over averaging scores: dense's cosine similarity (0-1) and sparse's lexical dot product (unbounded) are on different scales — averaging them raw lets whichever happens to produce bigger numbers dominate, and fixing that needs per-query normalization plus a tuned weight. RRF only looks at each chunk's *rank* in each ranking (1st, 2nd, ...), never the raw score, so there's nothing to calibrate.
- `reciprocal_rank_fusion(rankings, k=3, rrf_k=60)` — the general fusion step, model-agnostic (takes any number of chunk-index rankings, not just dense+sparse). `rrf_k=60` is the constant from the original RRF paper (Cormack et al., 2009); it softens how much rank 1 beats rank 2 (without it rank 1 scores exactly double rank 2).
- `retrieve_top_k(chunks, chunk_dense_embeddings, chunk_sparse_weights, query, k=3, rrf_k=60)` — gets each retriever's **full** ranking (not just its own top-k) via `dense_embeddings.py`/`sparse_embeddings.py`'s own `retrieve_top_k` with `k=len(chunks)`, then fuses. RRF needs each chunk's rank across the whole list, not just whether it made a small top-k cutoff.
- `hybrid_search.py`'s own `evaluate_retrieval` is MiniLM-only (it imports `dense_embeddings.py`'s single module-level model, no `model_name` param). `manisha.ipynb` adds its own `evaluate_hybrid(..., model_name=...)` on top — reuses `reciprocal_rank_fusion` (model-agnostic, only looks at ranks) plus the notebook's own model-aware `retrieve_top_k` — to fuse sparse with all 3 dense models (MiniLM, E5-base-v2, BGE-m3), not just MiniLM.
- Run against the real eval set across all 6 chunk strategies × 3 dense models (18 combos, each fused with sparse): best was `recursive_character_chunks_doc1` + BGE-m3+sparse (recall@k 1.0, ndcg@k 0.88).

## Results dashboard (implemented)

`eval_results.json` + `dashboard.html` — a local, dependency-free way to actually look at the dense/sparse/hybrid numbers instead of reading printed dicts out of notebook cell output.

- `manisha.ipynb`'s eval loops (dense, sparse, and both hybrid loops) now store results into dicts (`dense_all_results`, `sparse_all_results`, `hybrid_all_results`) as they run, not just print them. A final cell nests those into `eval_results.json` as `{engine: {chunk_strategy: {model: metrics}}}` (sparse has no model dimension, so it's `{chunk_strategy: metrics}`). Re-run the notebook and the file refreshes.
- `dashboard.html` — plain HTML/CSS/JS, no npm/build step, no charting library (CSS width bars instead — nothing to install for a handful of flat bars). `fetch()`s `eval_results.json`, so it needs a local HTTP server, not `file://` (browsers block `fetch` of local files under `file://`): `python3 -m http.server` from this directory, then open `http://localhost:8000/dashboard.html`.
- Layout: 3 top cards showing the best chunk-strategy/model combo per engine (ranked by `ndcg@k` — rewards a hit landing near rank 1, not just present somewhere in top-k), plus a sidebar (chunk strategy / engine / model dropdowns — model auto-disables under Sparse, since sparse is always BGE-m3) driving a detail panel of the 4 metrics for whatever combo is selected.

## Gotchas found running this on Apple Silicon (MPS)

- **`RuntimeError: Invalid buffer size: X GiB`** inside `SentenceTransformer.encode()` — not a normal out-of-memory, it's Metal (Apple's GPU backend, used by default on Mac) refusing a single allocation past its hard ceiling. Root cause: self-attention's memory cost scales with sequence length *squared*, and `encode()` batches multiple chunks together (32 by default), padding every chunk in a batch up to the length of the *longest* one in it. MiniLM can never hit this — it truncates at 256 tokens no matter what you feed it — but E5 (512 tokens) and especially BGE-m3 (8192 tokens) don't truncate nearly as early, so one unusually long chunk sneaking into a batch is enough to blow past Metal's ceiling, especially with BGE-m3's much bigger architecture (24 layers, 1024-dim vs MiniLM's 6 layers, 384-dim).
  - Fixed once, at the shared model-loading point in `dense_embeddings.py`: `model.max_seq_length = 512` — a no-op for MiniLM, but caps E5/BGE-m3 so no chunking strategy or model swap can trigger this crash again.
- **`TypeError: can only concatenate str (not "CompositeElement") to str`** in `embed_chunks` — `chunk_by_title` (and other `unstructured`-based chunkers) return a list of `Element` objects, not plain strings, unlike the LangChain splitters which return `list[str]`. Fixed with `str(chunk)` instead of `chunk` in the prefix-concatenation line — a no-op on an already-plain string, and pulls the text out of an `Element` (same idiom the notebook already uses elsewhere, e.g. `str(el)` when building `text_doc1`).

## Open TODOs (updated)

- Eval set of real questions: **done** (`eval_set.py`, 28 questions, self-verified against the source PDF).
- Dense embeddings + all 4 retrieval metrics: **done** (`dense_embeddings.py`).
- Running `evaluate_retrieval` with `EVAL_SET` against real chunks from each chunking strategy x each embedding model (dense, sparse, and hybrid, 18+ combos): **done**, in `manisha.ipynb` — real recall@k/precision@k/mrr/ndcg@k numbers computed for all three engines, see `eval_results.json`/`dashboard.html`. `chunk_by_title_doc1` is still excluded from `chunks_dict` (commented out) — its own local `evaluate_retrieval` in cell `b71ea6cc` has an unfixed `.lower()`-on-non-string bug (same root cause fixed elsewhere in `dense_embeddings.py`/`sparse_embeddings.py`), so it's never been run through the comparison.
- Contextual chunking (`contextual_chunking.py`) — chunks generated and used in the eval loops above (`contextualized_chunks_doc1`), so it has run against the real document.
- BGE-m3's sparse mode: **done** (`sparse_embeddings.py`, via `FlagEmbedding`'s `BGEM3FlagModel`), run against the real eval set. Multi-vector (ColBERT-style) mode still not implemented.
- Hybrid (sparse + dense) scoring: **done** (`hybrid_search.py` + `manisha.ipynb`'s multi-model extension, Reciprocal Rank Fusion), run against the real eval set across all chunk/model combos.
- Results dashboard: **done** (`dashboard.html` + `eval_results.json`) — local, dependency-free viewer for dense/sparse/hybrid scores.
- API key in `manisha.ipynb` cell `7b35df08` (`API_KEY = "sk-ant-..."`) is hardcoded in the notebook, not read from an env var, despite the cell's own docstring saying `ANTHROPIC_API_KEY` is already set in the environment — should switch to `anthropic.Anthropic()` (reads the env var automatically) and delete the hardcoded key before this notebook is ever shared/committed anywhere.
