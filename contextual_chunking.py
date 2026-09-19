"""
Contextual chunking with Claude Haiku (Anthropic's "Contextual Retrieval" trick).

Takes chunks you already made with any chunker (fixed-size, sentence, paragraph,
chunk_by_title...) and asks Haiku to write a 1-2 sentence blurb that situates
each chunk inside the whole document. That blurb gets glued onto the front of
the chunk before you embed it, so a chunk that on its own reads like "Revenue
grew 40% year over year" becomes "This is from the Q3 2026 financial results
section discussing regional sales. Revenue grew 40% year over year" - much
easier for an embedding model (and a human) to retrieve correctly.

Needs: pip install anthropic (already installed in .venv)
Needs: ANTHROPIC_API_KEY set in your environment (already set - confirmed via
`echo $ANTHROPIC_API_KEY` before writing this).
"""
import json

import anthropic

MODEL = "claude-haiku-4-5"  # cheapest Claude model - fine for a short "where does this fit" blurb


def add_context_to_chunks(full_document: str, chunks: list[str], client: anthropic.Anthropic = None) -> list[str]:
    """
    For every chunk, call Haiku once and prepend its answer to the chunk.

    The whole document is sent as a *cached* system block. Cache_control tells
    Anthropic "remember this exact block of text for a few minutes." Because
    `full_document` is byte-identical on every call in the loop, only the
    FIRST call pays full price to write it to cache - every call after that
    reads the same ~7k-token document back at ~10% of the input price instead
    of ~100%. Without this, contextualizing 22 chunks would mean sending the
    whole document 22 times at full price.
    """
    client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment - never hardcode the key

    contextualized = []
    for i, chunk in enumerate(chunks):
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,  # a 1-2 sentence blurb never needs more than this
            system=[
                {
                    "type": "text",
                    "text": f"<document>\n{full_document}\n</document>",
                    "cache_control": {"type": "ephemeral"},  # cache this block; TTL 5 min by default
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Here is the chunk we want to situate within the whole document:\n"
                        f"<chunk>\n{chunk}\n</chunk>\n\n"
                        "Give a short, succinct context (1-2 sentences) that situates this "
                        "chunk within the overall document, to improve search retrieval of "
                        "the chunk. Answer with only the context, nothing else."
                    ),
                }
            ],
        )
        context_blurb = response.content[0].text.strip()
        contextualized.append(f"{context_blurb}\n\n{chunk}")

        # usage tells you whether the cache actually hit - cache_read should be
        # 0 on chunk 0 (nothing cached yet) and equal to the document's token
        # count on every chunk after that
        u = response.usage
        print(
            f"chunk {i}: cache_write={u.cache_creation_input_tokens} "
            f"cache_read={u.cache_read_input_tokens} output={u.output_tokens}"
        )

    return contextualized


def save_chunks(chunks: list[str], path: str = "contextualized_chunks_doc1.json") -> None:
    """
    Write the list of chunk strings to a JSON file. `json.dump` with no
    `indent` argument produces compact output - no pretty-printing, no
    extra whitespace - so the whole list lands on one line in the file,
    even though each chunk's own "\\n\\n" (blurb + chunk text) stays intact
    inside its string: JSON escapes real newlines *inside* a string as the
    two characters backslash-n, so they don't turn into actual line breaks
    in the file.
    """
    with open(path, "w") as f:
        json.dump(chunks, f)


def load_chunks(path: str = "contextualized_chunks_doc1.json") -> list[str]:
    """Read the JSON file back and hand you the exact same list of strings you saved."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Cheap self-check - a 2-sentence "document" and 2 one-word chunks costs a
    # fraction of a cent. Run this file directly to confirm your API key and
    # the caching setup both work before pointing it at the real 5,498-word PDF.
    demo_document = "Vector Technics builds industrial sensors. Q3 revenue grew 40%."
    demo_chunks = ["Q3 revenue grew 40%.", "Vector Technics builds industrial sensors."]

    result = add_context_to_chunks(demo_document, demo_chunks)
    for r in result:
        print("---")
        print(r)

    assert len(result) == len(demo_chunks)
    assert all(isinstance(r, str) and len(r) > 0 for r in result)

    # round-trip check: save to disk, load back, confirm it's identical
    save_chunks(result, "demo_chunks.json")
    reloaded = load_chunks("demo_chunks.json")
    assert reloaded == result

    print("\nself-check passed")
