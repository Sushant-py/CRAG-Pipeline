import fitz
import chromadb
import os
import re
from sentence_transformers import SentenceTransformer

# ── Setup ChromaDB ─────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./vector_vault")
collection    = chroma_client.get_or_create_collection(name="ieee_papers")

# BGE-Small: current open-source standard for dense retrieval
print("Loading BGE-Small Embedding Model (High-Accuracy)...")
model = SentenceTransformer('BAAI/bge-small-en-v1.5')


# ── Chunking ───────────────────────────────────────────────────────────────────
def split_into_semantic_chunks(text, max_words=350, overlap_words=75):
    """
    Native Pure-Python Semantic Chunker.
    Splits text by sentence boundaries so context is never destroyed.
    """
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\s+', ' ', text)

    sentences = re.split(r'(?<=[.!?]) +', text.strip())

    chunks        = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())

        if current_words + words_in_sentence > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))

            overlap_chunk = []
            overlap_count = 0
            for prev_sentence in reversed(current_chunk):
                prev_words = len(prev_sentence.split())
                if overlap_count + prev_words <= overlap_words:
                    overlap_chunk.insert(0, prev_sentence)
                    overlap_count += prev_words
                else:
                    break

            current_chunk = overlap_chunk + [sentence]
            current_words = overlap_count + words_in_sentence
        else:
            current_chunk.append(sentence)
            current_words += words_in_sentence

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ── Ingestion ──────────────────────────────────────────────────────────────────
def process_and_save_pdf(pdf_path):
    file_name = os.path.basename(pdf_path)

    existing = collection.get(where={"source": file_name})
    if len(existing['ids']) > 0:
        print(f" Skipping {file_name}: Already in the vault.")
        return

    doc       = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text(sort=True) + "\n"

    chunks    = split_into_semantic_chunks(full_text)
    metadatas = []
    ids       = []

    for i, chunk in enumerate(chunks):
        metadatas.append({"source": file_name})
        ids.append(f"{file_name}_{i}")

    print(f"Translating {len(chunks)} semantic chunks from {file_name}...")
    vectors = model.encode(chunks).tolist()

    collection.add(
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )
    print(f" Saved {file_name}")


# ── Retrieval ──────────────────────────────────────────────────────────────────
def search_vault(query, n_results=10):
    """
    Retrieve top-n_results chunks and pre-filter by cosine distance.

    FIX — Distance pre-filter added:
      ChromaDB returns cosine distances (0 = identical, 2 = opposite).
      A distance > 0.60 means the chunk is semantically dissimilar enough
      that the CrossEncoder is unlikely to save it; dropping these early
      reduces the number of marginal chunks the logic engine has to grade,
      which directly lowers false positives.

    The default n_results is now 10 (was 5) so we cast a wider net before
    filtering — this preserves recall while the stricter logic-engine
    thresholds protect precision.
    """
    # BGE models require an "Instruct" prefix for retrieval queries
    search_query  = f"Represent this sentence for searching relevant passages: {query}"
    query_vector  = model.encode([search_query]).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    raw_docs      = results["documents"][0]
    raw_metas     = results["metadatas"][0]
    raw_distances = results["distances"][0]

    # Pre-filter: discard chunks whose cosine distance exceeds the threshold.
    # Lower distance  = more similar.  0.60 is a conservative cut-off that
    # removes clearly off-topic chunks while keeping borderline ones for the
    # CrossEncoder to judge.
    DISTANCE_THRESHOLD = 0.60

    filtered_docs   = []
    filtered_metas  = []

    for doc, meta, dist in zip(raw_docs, raw_metas, raw_distances):
        if dist <= DISTANCE_THRESHOLD:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
        else:
            src = meta.get("source", "Unknown")
            print(f"  [DB pre-filter] Dropped chunk from '{src}' — distance={dist:.3f} > {DISTANCE_THRESHOLD}")

    results["documents"][0] = filtered_docs
    results["metadatas"][0] = filtered_metas

    print(f"  [DB] Retrieved {len(raw_docs)} chunks → {len(filtered_docs)} passed distance pre-filter")
    return results


# ── CLI / batch ingestion ──────────────────────────────────────────────────────
if __name__ == "__main__":
    paper_folder = "papers"
    if not os.path.exists(paper_folder):
        os.makedirs(paper_folder)
        print(f"Please put your PDFs in the '{paper_folder}' folder.")

    pdf_files = [f for f in os.listdir(paper_folder) if f.endswith('.pdf')]
    print(f" Found {len(pdf_files)} papers. Starting upload...")

    for filename in pdf_files:
        path = os.path.join(paper_folder, filename)
        try:
            process_and_save_pdf(path)
        except Exception as e:
            print(f" Error with {filename}: {e}")

    print("\n DATABASE READY.")