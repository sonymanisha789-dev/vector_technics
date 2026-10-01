# RAG POC Comparison

Living document comparing RAG proof-of-concepts built by different team members. Each POC gets a column in the table below and a detail section further down. Update both when adding a new POC — the table is the fast scan, the detail sections are the evidence backing each cell.

## How to add a new POC

1. Drop the new team member's write-up (PDF, doc, notes) in the project and tell me to add it here.
2. I'll add a column to the comparison table and a new `## POC: <name>` detail section following the same structure as the existing ones.
3. If a new POC introduces a criterion the table doesn't have yet (e.g. a new embedding architecture, a new eval method), I'll add a row — don't force-fit it into an existing one.

## Comparison table

| Criterion | POC: Vector Technics (ours) | POC: Kunal/Rithish (3-strategy benchmark) |
|---|---|---|
| Ingestion | `unstructured.partition_pdf()` | `unstructured.partition_pdf()` |
| Chunking strategies tried | 6 (fixed, recursive, sentence, structure-aware/title, semantic, LLM-contextual) | 1 (custom rule-based: RecursiveCharacterTextSplitter + title boundaries + intact tables) |
| Embedding architectures | 2 (dense, sparse) | 3 (dense+sparse hybrid, ColBERT-style late interaction, Jina asymmetric adapters) |
| Hybrid fusion method | Reciprocal Rank Fusion (rank-based, no score calibration needed) | alpha-weighted score average, alpha=0.7, **not tuned** |
| Eval metrics | recall@k, precision@k, MRR, NDCG, + ragas LLM-judged | Recall@5, MRR |
| Ground truth method | Substring match (+ LLM-judged fallback via ragas) | Keyword match — same class of weakness as substring match |
| Combos evaluated | 18+ (6 chunking × 3 dense models, each fused w/ sparse) | 1 complete (Modular Hybrid only; ColBERT/Jina implemented but never finished benchmarking) |
| Documents tested | 1 (Raphe mPhibr PDF, 28 questions) | 2 (13MB annual report, 4KB competitive intel doc) |
| Generation step (actual answers, not just retrieval) | Yes — `rag_query.py` via Claude Haiku, cited sources | Not present — pipeline stops at retrieval scoring |
| Persistence | pgvector, plus CLI (`run_rag.py`: store/query/score) | In-memory only (explicit known limitation) |
| Results viewer | `dashboard.html` + `eval_results.json` | Static tables/chart in write-up |
| Known unresolved issues | none blocking | SPLADE indexing unbatched (18-min index time on 6,169 chunks); Windows Application Control blocking native DLLs (unresolved) |

## Similarities

- Same ingestion library (`unstructured.partition_pdf()`) and same base chunker (`RecursiveCharacterTextSplitter`), with a title/structure-aware variant in both.
- Both ground truth sets use literal text matching (substring/keyword), which can't catch paraphrased answers — a shared, unaddressed weakness in both POCs' core eval, not just a footnote.
- Both explicitly flag their eval sample sizes as too small for statistical significance, only for validating pipeline correctness.

## Differences

- **Embedding architecture breadth**: their POC covers 3 distinct representation types (dense+sparse, late-interaction/ColBERT, asymmetric adapters/Jina); ours covers 2 (dense, sparse). This is their strongest advantage — ColBERT and Jina-style adapters don't exist in our codebase at all.
- **Everything downstream of embedding**: fusion method (RRF vs. untuned weighted average), chunking strategies swept (6 vs. 1), eval depth (4 metrics + LLM-judged vs. 2 metrics), combos actually benchmarked to completion (18+ vs. 1), generation, persistence, and a results dashboard are all present in ours and absent or incomplete in theirs.
- **Document coverage**: they stress-tested on two docs of very different size/difficulty; we've only run the full pipeline against one document.

## Which is better

Depends what "the POC" is supposed to prove:

- **If the goal is "which embedding architecture wins"**: their POC is more credible in intent (3 architectures) but incomplete in execution — only 1 of 3 has full benchmark numbers, so right now it doesn't actually answer its own question.
- **If the goal is "can we ship a working RAG system"**: ours is further along — it's the only one of the two with a generation step, persistent storage, and a CLI someone could actually run against a new document today.
- **Recommendation**: bring their ColBERT/Jina embedding code into our pipeline (it already has the chunking sweep + eval harness + RRF fusion to properly benchmark them) rather than trying to add our generation/persistence work into theirs. That gets both POCs' strengths into one system instead of picking a "winner" and discarding the other's work.

## Open questions for the team

- Do we need late-interaction (ColBERT) or asymmetric adapters (Jina) for our actual use case, or is dense+sparse+RRF sufficient? Their POC doesn't answer this since ColBERT/Jina numbers were never completed.
- Should ground truth move to an LLM-judged or human-verified set instead of substring/keyword matching, given both POCs share this weakness?
- Is our SPLADE/sparse embedding batched? Their POC's unbatched loop caused an 18-minute index time on ~6K chunks — worth checking ours doesn't have the same bottleneck.
