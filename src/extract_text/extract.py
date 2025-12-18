from pathlib import Path
import json
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

from src.chunker.first_pdf_chuncker import StructuredWeatherClothingChunker
from src.chunker.second_pdf_chuncker import WeatherDatabaseBulletChunker


SRC_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SRC_DIR / "data"
OUT_DIR = DATA_DIR / "out_chunks"

PDF_01 = DATA_DIR / "Weather Activity Clothing Database - A Comprehensive Global Guide.pdf"
PDF_02 = DATA_DIR / "Weather Activity Clothing Database.pdf"


def load_pdf_text(pdf_path: Path) -> str:
    loader = PyMuPDF4LLMLoader(str(pdf_path))
    docs = loader.load()
    return "\n".join((d.page_content or "") for d in docs)


def save_jsonl(chunks, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in chunks:
            f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata or {}}, ensure_ascii=False) + "\n")
    print(f"[OK] Saved JSONL: {out_path} | chunks={len(chunks)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # PDF_01
    raw_01 = load_pdf_text(PDF_01)
    chunker_01 = StructuredWeatherClothingChunker()
    chunks_01 = chunker_01.split_text(raw_01)
    print(f"PDF_01 chunks: {len(chunks_01)}")
    save_jsonl(chunks_01, OUT_DIR / "weather_global_guide.structured.jsonl")

    # PDF_02
    raw_02 = load_pdf_text(PDF_02)
    chunker_02 = WeatherDatabaseBulletChunker()
    chunks_02 = chunker_02.split_text(raw_02)
    print(f"PDF_02 chunks: {len(chunks_02)}")
    save_jsonl(chunks_02, OUT_DIR / "weather_database.bullets.jsonl")


if __name__ == "__main__":
    main()
