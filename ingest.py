import re
import pdfplumber
import chromadb
from pathlib import Path

PDF_PATH = Path("laws/2026en-laws-of-the-game-compressed.pdf")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rugby_laws"


def extract_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def chunk_by_law(text: str) -> list[dict]:
    """Split text into chunks by law number (e.g. Law 1, Law 11.4)."""
    # Match "Law N" or "LAW N" at the start of a line as a top-level boundary
    pattern = re.compile(r"(?=^LAW\s+\d+[\s\S]|^Law\s+\d+[\s\S])", re.MULTILINE | re.IGNORECASE)
    splits = pattern.split(text)

    chunks = []
    for split in splits:
        split = split.strip()
        if not split:
            continue

        # Try to extract a law reference from the start of the chunk
        law_match = re.match(r"(?:Law|LAW)\s+(\d+)", split)
        law_num = law_match.group(1) if law_match else "unknown"

        # Further split long chunks by sub-law (e.g. 11.1, 11.2)
        sub_pattern = re.compile(r"(?=^\d+\.\d+\s)", re.MULTILINE)
        sub_splits = sub_pattern.split(split)

        for sub in sub_splits:
            sub = sub.strip()
            if len(sub) < 50:  # skip very short fragments
                continue

            sub_match = re.match(r"^(\d+\.\d+)", sub)
            if sub_match:
                label = f"Law {sub_match.group(1)}"
            else:
                label = f"Law {law_num}"

            chunks.append({"id": label, "text": sub, "law": law_num})

    return chunks


def load_into_chromadb(chunks: list[dict]):
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate to allow re-running ingest cleanly
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{chunk['id']}_{i}")
        documents.append(chunk["text"])
        metadatas.append({"law": chunk["law"], "label": chunk["id"]})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Loaded {len(chunks)} chunks into ChromaDB")


if __name__ == "__main__":
    print("Extracting text from PDF...")
    text = extract_text(PDF_PATH)
    print(f"Extracted {len(text)} characters")

    print("Chunking by law...")
    chunks = chunk_by_law(text)
    print(f"Found {len(chunks)} chunks")

    if chunks:
        print("Sample chunk:")
        print(f"  Label: {chunks[0]['id']}")
        print(f"  Text: {chunks[0]['text'][:200]}")

    print("Loading into ChromaDB...")
    load_into_chromadb(chunks)
    print("Done!")
