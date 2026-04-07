import os
import sys
import time
import logging
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder
from database import search_vault

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Add it to your .env file:\n"
        "GROQ_API_KEY=gsk_your_key_here"
    )

client = Groq(api_key=_api_key)

# ── Model config ───────────────────────────────────────────────────────────────
LIGHT_TEXT_MODEL = "llama-3.1-8b-instant"
ANSWER_MODEL     = "llama-3.1-8b-instant"

# FIX #4 — RETRIEVE-THEN-RERANK ARCHITECTURE
# Fetch a wide net of 30 chunks from the vector database to guarantee high Recall,
# but only process the Top 10 to protect Precision and save LLM tokens.
FETCH_K          = 30  
PROCESS_K        = 10  
CHUNK_WORD_LIMIT = 300

# ── CrossEncoder grader ────────────────────────────────────────────────────────
print("🧠 Loading Cross-Encoder Scoring Model (This takes a few seconds)...")
grader_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

CORRECT_THRESHOLD   = 4.5   
AMBIGUOUS_THRESHOLD = 0.5   


# ── Core LLM call with retry ───────────────────────────────────────────────────
def call_llm(
    prompt: str,
    system: str = "You are a helpful and accurate assistant.",
    model: str = ANSWER_MODEL,
    max_tokens: int = 1024,
    retries: int = 2,
) -> str:
    from groq import RateLimitError
    delay = 5

    for attempt in range(1, retries + 2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            if attempt > retries:
                raise
            wait = delay * attempt
            log.warning("Rate limit hit. Waiting %ds before retry…", wait)
            time.sleep(wait)


def _trim(text: str, max_words: int = CHUNK_WORD_LIMIT) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …[truncated]"


# ── Step 2a: Extract best sentence from AMBIGUOUS chunk ────────────────────────
def extract_best_sentence(query: str, chunk: str) -> str:
    prompt = f"""You are a precision fact extractor for a scientific research system.

The following text chunk is partially relevant to the user's question.
It contains mostly irrelevant content, but has 1 useful factual sentence buried inside.

User Question:
{query}

Text Chunk:
{_trim(chunk)}

Task: Extract and return ONLY the single most factual and relevant sentence
from the chunk above that directly helps answer the question.

Rules:
- Return ONLY that one sentence, word for word as it appears in the chunk.
- Do NOT paraphrase, summarise, or add any explanation.
- Do NOT return more than one sentence.
- If no single sentence is useful, return exactly: NO_USEFUL_SENTENCE
"""
    result = call_llm(prompt, model=ANSWER_MODEL, max_tokens=128).strip()
    if not result or result == "NO_USEFUL_SENTENCE":
        return ""
    return result


# ── Step 2b: Supplemental search for AMBIGUOUS chunks ─────────────────────────
def supplemental_search(query: str, anchor_sentence: str, n: int = 3) -> tuple[list, list]:
    sub_query_prompt = f"""You are a search query builder.

Given this user question:
"{query}"

And this key sentence extracted from a partially relevant document:
"{anchor_sentence}"

Write a short, keyword-focused search query (under 10 words) that combines
the topic of the question with the specific concept in the sentence.
Return ONLY the query, no explanation.
"""
    sub_query = call_llm(sub_query_prompt, model=LIGHT_TEXT_MODEL, max_tokens=32).strip()
    log.info("    Supplemental sub-query: %r", sub_query)

    raw = search_vault(sub_query, n_results=n)
    return raw["documents"][0], raw["metadatas"][0]


# ── Step 2c: Compare two candidates ───────────────────────────────────────────
def pick_more_relevant(query: str, candidate_a: str, candidate_b: str) -> str:
    score_a = grader_model.predict([query, candidate_a])
    score_b = grader_model.predict([query, candidate_b])
    return "A" if score_a >= score_b else "B"


# ── Step 2d: Grade all chunks and filter (RERANKING APPLIED HERE) ─────────────
def evaluate_chunks(query: str, raw_chunks: list, raw_metadatas: list) -> tuple[list, list]:
    verified_chunks  = []
    verified_sources = []

    print("━" * 50)
    print(f"  PHASE 1 — Reranking {len(raw_chunks)} fetched chunks...")
    
    # 1. Score ALL retrieved chunks
    scored_candidates = []
    for i, chunk in enumerate(raw_chunks):
        source = raw_metadatas[i].get("source", "Unknown") if raw_metadatas else "Unknown"
        score = float(grader_model.predict([query, _trim(chunk)]))
        scored_candidates.append((score, chunk, source))

    # 2. Sort descending (highest math score first)
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # 3. Slice the top K to process
    top_candidates = scored_candidates[:PROCESS_K]

    print(f"  Grading the Top {len(top_candidates)} chunks")
    print("━" * 50)

    grades  = []
    chunks  = []
    sources = []

    for i, (score, chunk, source) in enumerate(top_candidates):
        # Assign Grades based on thresholds
        if score >= CORRECT_THRESHOLD:
            grade = "CORRECT"
        elif score >= AMBIGUOUS_THRESHOLD:
            grade = "AMBIGUOUS"
        else:
            grade = "INCORRECT"

        grades.append(grade)
        chunks.append(chunk)
        sources.append(source)

        icon = {"CORRECT": "✅", "AMBIGUOUS": "🟡", "INCORRECT": "❌"}.get(grade, "❓")
        print(f"  Rank {i+1} → {icon} {grade:<10} score={score:+.2f}  [{source}]")

    print("━" * 50)
    print(
        f"  Summary: {grades.count('CORRECT')} CORRECT  |  "
        f"{grades.count('AMBIGUOUS')} AMBIGUOUS  |  "
        f"{grades.count('INCORRECT')} INCORRECT"
    )
    print("━" * 50)

    # Keep all CORRECT chunks
    for chunk, grade, source in zip(chunks, grades, sources):
        if grade == "CORRECT":
            verified_chunks.append(chunk)
            verified_sources.append(source)

    ambiguous_indices = [i for i, g in enumerate(grades) if g == "AMBIGUOUS"]

    if ambiguous_indices:
        print(f"  PHASE 2 — Resolving {len(ambiguous_indices)} AMBIGUOUS chunk(s)")
        print("━" * 50)

        for n, idx in enumerate(ambiguous_indices):
            chunk  = chunks[idx]
            source = sources[idx]

            print(f"  ┌─ Resolving Rank {idx+1} [{source}]"
                  f"  ({n+1} of {len(ambiguous_indices)} ambiguous)")

            best_sentence = extract_best_sentence(query, chunk)

            if not best_sentence:
                print("  └─ ⚠️  No useful sentence found → discarded.")
                if n < len(ambiguous_indices) - 1:
                    print("  │")
                continue

            print(f"  │  📌 Extracted: {best_sentence[:120]!r}")
            print("  │  🔍 Supplemental search…")
            supp_chunks, supp_metas = supplemental_search(query, best_sentence, n=3)

            print("  │  Grading supplemental chunks:")
            best_supp        = None
            best_supp_source = source

            for j, sc in enumerate(supp_chunks):
                sc_source = supp_metas[j].get("source", "Unknown") if supp_metas else "Unknown"
                sc_score  = float(grader_model.predict([query, _trim(sc)]))
                sc_grade  = "CORRECT" if sc_score >= CORRECT_THRESHOLD else ("AMBIGUOUS" if sc_score >= AMBIGUOUS_THRESHOLD else "INCORRECT")
                
                icon = {"CORRECT": "✅", "AMBIGUOUS": "🟡", "INCORRECT": "❌"}.get(sc_grade, "❓")
                print(f"  │    Supp {j+1}/3 → {icon} {sc_grade:<10} score={sc_score:+.2f}  [{sc_source}]")

                if sc_grade == "CORRECT" and best_supp is None:
                    best_supp        = sc
                    best_supp_source = sc_source

            if best_supp:
                winner = pick_more_relevant(query, best_sentence, _trim(best_supp, 150))
                if winner == "B":
                    print("  └─ ✅ RESOLVED → supplemental chunk is BETTER — kept.")
                    verified_chunks.append(best_supp)
                    verified_sources.append(best_supp_source)
                else:
                    print("  └─ ✅ RESOLVED → extracted sentence is BETTER — kept.")
                    verified_chunks.append(best_sentence)
                    verified_sources.append(source)
            else:
                rescue_score = float(grader_model.predict([query, best_sentence]))
                if rescue_score >= AMBIGUOUS_THRESHOLD:
                    print(f"  └─ 📎 Extracted sentence re-graded {rescue_score:+.2f} → kept.")
                    verified_chunks.append(best_sentence)
                    verified_sources.append(source)
                else:
                    print(f"  └─ ❌ Extracted sentence re-graded {rescue_score:+.2f} → discarded.")

            if n < len(ambiguous_indices) - 1:
                print("  │")

        print("━" * 50)

    print(f"  RESULT — {len(verified_chunks)} item(s) in Verified Context Array.")
    print("━" * 50)

    return verified_chunks, verified_sources


# ── Step 3: Rewrite query once if everything failed ────────────────────────────
def rewrite_query(original_query: str) -> str:
    prompt = f"""You are a search query optimizer for a scientific research database.

The following question failed to retrieve relevant results:
"{original_query}"

Rewrite it as a short, keyword-focused search query using different,
more specific scientific terminology. Return ONLY the rewritten query.
Keep it under 10 words.
"""
    return call_llm(prompt, model=LIGHT_TEXT_MODEL, max_tokens=32).strip()


# ── Step 4: Generate final answer ─────────────────────────────────────────────
def generate_final_answer(query: str, chunks: list, sources: list) -> str:
    if not chunks:
        return "No information detected from the papers for your query."

    context = "\n\n---\n\n".join(
        f"[Source: {sources[i]}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""You are a precise scientific research assistant.

Answer the user's question using ONLY the context provided below.

Rules:
- Be clear, structured, and concise.
- Do NOT include any information not present in the context.
- If the context only partially answers the question, clearly say so.
- Cite the source filenames where relevant (e.g. "According to paper.pdf...").

User Question:
{query}

Verified Context:
{context}
"""
    return call_llm(
        prompt,
        model=ANSWER_MODEL,
        max_tokens=900,
        system=(
            "You are a strict, citation-aware research assistant. "
            "Never hallucinate. Only use the provided context."
        ),
    )


# ── Step 4b: Generate soft / partial answer ────────────────────────────────────
def generate_soft_answer(query: str, chunks: list, sources: list) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {sources[i]}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )
    prompt = f"""You are a helpful research assistant.
I could not find a direct, perfect answer to the user's question in our database.
However, I found some related snippets from the following papers: {set(sources)}

User Question: {query}

Task:
1. Start by explicitly stating you couldn't find a direct answer.
2. Summarize the related information provided below.
3. Be honest about why this might only be partially relevant.

Related Context:
{context}
"""
    return call_llm(prompt, model=ANSWER_MODEL, max_tokens=900)


# ── Step 5: Master CRAG pipeline ──────────────────────────────────────────────
def run_logic_engine(user_query: str) -> dict:

    # ── Attempt 1: Fetch 30, Rerank, Evaluate Top 10 ──────────────────────────
    raw      = search_vault(user_query, n_results=FETCH_K)
    chunks   = raw["documents"][0]
    metadatas = raw["metadatas"][0]

    verified, sources = evaluate_chunks(user_query, chunks, metadatas)

    if verified:
        answer = generate_final_answer(user_query, verified, sources)
        return {
            "status":     "success",
            "answer":     answer,
            "facts":      verified,
            "sources":    list(set(sources)),
            "query_used": user_query,
        }

    # ── Attempt 2: rewrite query ──────────────────────────────────────────────
    print()
    print("━" * 50)
    print("  ⚠️  ADAPTIVE FALLBACK — all chunks failed")
    print("━" * 50)

    new_query = rewrite_query(user_query)
    print(f"  Rewritten query: {new_query!r}")
    print("━" * 50)
    print()

    raw2      = search_vault(new_query, n_results=FETCH_K)
    chunks2   = raw2["documents"][0]
    metadatas2 = raw2["metadatas"][0]

    verified2, sources2 = evaluate_chunks(new_query, chunks2, metadatas2)

    if verified2:
        answer = generate_final_answer(user_query, verified2, sources2)
        return {
            "status":     "fallback_success",
            "answer":     answer,
            "facts":      verified2,
            "sources":    list(set(sources2)),
            "query_used": new_query,
        }

    # ── Attempt 3: Bendy / partial match ──────────────────────────────────────
    print()
    print("━" * 50)
    print("  🔗 BENDY MODE — Finding related snippets...")
    print("━" * 50)

    BENDY_THRESHOLD = 1.5   

    scored_bendy = []
    for i, chunk in enumerate(chunks2):
        score = float(grader_model.predict([user_query, chunk]))
        if score >= BENDY_THRESHOLD:
            src = metadatas2[i].get("source", "Unknown") if metadatas2 else "Unknown"
            scored_bendy.append((score, chunk, src))
            print(f"  Fetched Chunk {i+1} → score={score:+.2f} passes BENDY_THRESHOLD")

    # Sort descending by score and keep only the 2 most relevant
    scored_bendy.sort(key=lambda x: x[0], reverse=True)
    scored_bendy = scored_bendy[:2]

    related_chunks  = [x[1] for x in scored_bendy]
    related_sources = [x[2] for x in scored_bendy]

    if related_chunks:
        bendy_answer = generate_soft_answer(user_query, related_chunks, related_sources)
        return {
            "status":     "partial_match",
            "answer":     bendy_answer,
            "facts":      related_chunks,
            "sources":    list(set(related_sources)),
            "query_used": new_query,
        }

    return {
        "status":     "failed",
        "answer":     "No information detected from the papers for your query.",
        "facts":      [],
        "sources":    [],
        "query_used": new_query,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    question = (
        " ".join(sys.argv[1:]).strip()
        if len(sys.argv) > 1
        else "What is the criterion for turbulent flow?"
    )

    print()
    print("━" * 50)
    print("  QUESTION")
    print("━" * 50)
    print(f"  {question}")
    print("━" * 50)
    print()

    result = run_logic_engine(question)

    print()
    print("━" * 50)
    print("  FINAL ANSWER")
    print("━" * 50)
    print()
    print(result["answer"])
    print()
    print("━" * 50)
    print("  PIPELINE SUMMARY")
    print("━" * 50)
    status_icon = {
        "success":          "✅",
        "fallback_success": "⚠️",
        "partial_match":    "🔗",
        "failed":           "❌",
    }.get(result["status"], "❓")
    print(f"  Status      : {status_icon} {result['status'].upper()}")
    print(f"  Query used  : {result['query_used']}")
    print(f"  Facts kept  : {len(result['facts'])} / min({FETCH_K}, {PROCESS_K})")
    print(f"  Sources     : {', '.join(result['sources']) if result['sources'] else 'None'}")
    print("━" * 50)
    print()