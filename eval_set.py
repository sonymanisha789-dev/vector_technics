"""
Eval set for 6b5330ba3c12ba40.pdf (the Raphe mPhibr drone-investment report).

Written as an outsider reviewer: every question comes from reading the raw
PDF text (via `pdftotext -layout`) directly, with NO look at how the notebook
chunks or embeds it. That's deliberate - if a question were phrased to match
a convenient chunk boundary or a specific chunking/embedding choice, a good
recall@k score would just mean "the eval set was written to be easy," not
"retrieval actually works." Questions are spread across every major section
(executive summary, company profile, funding rationale, competitive
landscape - Indian and global peers, market figures, government policy,
risks, conclusion) so no one section is over- or under-tested.

Each pair is (question, answer_substring). A chunk "hits" for a question if
answer_substring appears in it, case-insensitive - see retrieve_top_k /
evaluate_retrieval in dense_embeddings.py.

Note on answer_substring choices: this PDF's font mangles "fi"/"fl"/"tf"
ligatures on extraction - "significant" comes out as "signi cant", "profit"
as "prot", "flight" as "ight", "platform" as "plaorm". Every answer below was
picked to dodge a mangled ligature (numbers, proper nouns, multi-word
phrases) so a miss reflects a retrieval problem, not an extraction artifact.
`pdftotext -layout` also sometimes wraps a number across a line break (e.g.
"$6.26" / "billion" landed on separate lines) - the self-check below caught
two of these during transcription; each was swapped for an equivalent fact
that lands on one line intact.
"""

EVAL_SET = [
    ("How much did Raphe mPhibr raise in its Series B round?", "$100 million"),
    ("Who led Raphe mPhibr's Series B funding round?", "General Catalyst"),
    ("What is Raphe mPhibr's valuation after the Series B round?", "$900 million"),
    ("What is Raphe mPhibr's total funding to date?", "$145 million"),
    ("In which city is Raphe mPhibr based?", "Noida"),
    ("Who founded Raphe mPhibr?", "Vivek Mishra and Vikash Mishra"),
    ("How big is Raphe mPhibr's manufacturing facility in Noida?", "650,000 square feet"),
    ("What was the size of Raphe mPhibr's initial facility before expansion?", "70,000 square feet"),
    ("How many UAVs can the mR10 drone swarm coordinate?", "100 UAVs"),
    ("What is the tracking precision of the mR10 drone swarm?", "99.5% tracking precision"),
    ("What is the maximum payload capacity of Raphe mPhibr's drones?", "100 kilograms"),
    ("What is the maximum flight time of Raphe mPhibr's drones?", "360 minutes"),
    ("How many larger drone units does Raphe mPhibr plan to manufacture annually?", "1,500-2,000"),
    ("How much did the Indian government plan to invest in UAVs over the next 12-24 months?", "$470 million"),
    ("How much did Indian drone makers collectively raise in the five years before this round?", "$151 million"),
    ("Who backs Garuda Aerospace?", "MS Dhoni"),
    ("By how much did IdeaForge's revenue contract in FY25?", "47.05%"),
    ("What was Tata Advanced Systems' revenue as of March 2024?", "$585 million"),
    ("How much total funding has Anduril Industries raised?", "$6.26B"),
    ("What is Anduril Industries' valuation?", "$12.5 billion"),
    ("How much total funding has Helsing raised?", "€1.36 billion"),
    ("How much total funding has Vannevar Labs raised?", "$91.5 million"),
    ("What share of the global military drone market do fixed-wing drones hold in 2025?", "85%"),
    ("What is the projected size of the global military drone market by 2032?", "34.4 billion"),
    ("At what CAGR is India's military drone market expected to grow from 2025?", "17.9%"),
    ("Where are most of India's imported drone components sourced from?", "China"),
    ("What is the total incentive amount under India's PLI scheme for drones?", "Rs 120 crore"),
    ("What real-world military operation validated Raphe mPhibr's drones?", "Operation Sindoor"),
]


if __name__ == "__main__":
    # Self-check: confirm every answer_substring was transcribed correctly by
    # re-extracting the PDF's text and checking each one is actually in there.
    # Catches a typo on my end before it silently costs you a point on every
    # future recall@k run.
    import subprocess

    pdf_path = "/Users/mjmacair15/Vector_Technics_RAG_POC/6b5330ba3c12ba40.pdf"
    result = subprocess.run(["pdftotext", "-layout", pdf_path, "-"], capture_output=True, text=True)
    document_text = result.stdout.lower()

    missing = [(q, a) for q, a in EVAL_SET if a.lower() not in document_text]
    assert not missing, f"these answer substrings were NOT found in the PDF text: {missing}"

    print(f"self-check passed - all {len(EVAL_SET)} answer substrings verified present in the source PDF")
