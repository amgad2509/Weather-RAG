streamlit run app.py

uvicorn src.api.main:app --reload --host localhost --port 8000

uv run --active python -m src.extract_text.extract