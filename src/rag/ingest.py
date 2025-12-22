import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import cassio
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .builder import build_vectorstore


CHUNKS_DIR = Path(__file__).resolve().parents[1] / "data" / "out_chunks"
DEFAULT_FILES: Sequence[str] = [
    "weather_global_guide.structured.jsonl",
    "weather_database.bullets.jsonl",
]


def _load_documents(paths: Iterable[Path], limit: int | None = None) -> List[Document]:
    docs: List[Document] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk file: {path}")

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = obj.get("page_content") or ""
                meta = obj.get("metadata") or {}
                if content.strip():
                    docs.append(Document(page_content=content, metadata=meta))

                if limit and len(docs) >= limit:
                    return docs
    return docs


def seed_vectorstore(
    *,
    table_name: str = "weather_data",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    chunk_files: Sequence[str] = DEFAULT_FILES,
    dry_run: bool = False,
    limit: int | None = None,
    test_query: str | None = None,
    k: int = 5,
) -> None:
    """
    Load chunked documents from disk and insert them into the Cassandra/Astra vector store.
    Use dry_run=True to only report counts.
    """
    load_dotenv()
    db_id = (os.getenv("CASSIO_DB_ID") or "").strip()
    token = (os.getenv("CASSIO_TOKEN") or "").strip()
    if not db_id or not token:
        raise ValueError("CASSIO_DB_ID and CASSIO_TOKEN must be set in the environment/.env")

    cassio.init(database_id=db_id, token=token)

    paths = [CHUNKS_DIR / name for name in chunk_files]
    docs = _load_documents(paths, limit=limit)
    print(f"[vectorstore] Loaded {len(docs)} documents from {len(paths)} file(s).")

    if dry_run:
        return

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = build_vectorstore(embeddings=embeddings, table_name=table_name)

    vectorstore.add_documents(docs)
    print(f"[vectorstore] Upserted {len(docs)} documents into table '{table_name}'.")

    if test_query:
        results = vectorstore.similarity_search(test_query, k=k)
        print(f"[vectorstore] Test query='{test_query}' -> {len(results)} result(s)")
        for idx, doc in enumerate(results, 1):
            meta = getattr(doc, "metadata", {}) or {}
            preview = (doc.page_content or "").replace("\n", " ")[:160]
            print(f"  {idx}. {meta.get('country', 'n/a')} | {meta.get('weather', 'n/a')} | {meta.get('section', 'n/a')}")
            print(f"     {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Cassandra/Astra vector store with weather chunks.")
    parser.add_argument("--table", default="weather_data", help="Target Cassandra table name.")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2", help="Embedding model name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for number of docs to load.")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts, do not upsert.")
    parser.add_argument("--test-query", default=None, help="Optional query to run after ingest.")
    parser.add_argument("--k", type=int, default=5, help="Top-K results to fetch for the test query.")
    args = parser.parse_args()

    seed_vectorstore(
        table_name=args.table,
        embedding_model=args.model,
        dry_run=args.dry_run,
        limit=args.limit,
        test_query=args.test_query,
        k=args.k,
    )


if __name__ == "__main__":
    main()
