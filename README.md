
---

# Weather Activity & Clothing Assistant

An LLM-powered assistant that provides **current weather**, **clothing recommendations**, and **outdoor activity suggestions** based on a user’s location. The system combines real-time weather data, retrieval-augmented generation (RAG) over a curated knowledge base, and tool-calling orchestration using LangGraph.

---

## Table of Contents

* [Overview](#overview)
* [Tooling Model](#tooling-model)
* [Key Features](#key-features)
* [Architecture](#architecture)
* [Core Components](#core-components)
* [API](#api)
* [Streamlit UI](#streamlit-ui)
* [Knowledge Base & RAG Pipeline](#knowledge-base--rag-pipeline)
* [Configuration](#configuration)
* [Installation](#installation)
* [Run](#run)
* [Testing](#testing)
* [Troubleshooting](#troubleshooting)
* [Project Structure](#project-structure)
* [License--credits](#license--credits)

---

## Overview

This project answers user queries such as:

* “What’s the weather now in Cairo?”
* “What should I wear today in Egypt?”
* “What outdoor activities are suitable in London right now?”

It uses:

* **OpenWeatherMap** for live weather conditions
* **Groq (ChatGroq)** for LLM + tool calling
* **Cassandra / Astra DB via Cassio** for vector storage
* **HuggingFace embeddings** for semantic indexing
* **Cohere Rerank** for reranking / contextual compression
* **LangGraph** to orchestrate tool execution
* **FastAPI** for backend endpoints (**non-stream + true streaming SSE**)
* **Streamlit** for an interactive chat UI

---

## Tooling Model

The assistant is designed around **exactly three tools**. The LLM must select between them based on intent.

### Tool 1 — `weather_query(location: str)`

* **Purpose:** Fetch real-time weather data for a given country/city.
* **Source:** OpenWeatherMap API.
* **Used when:** The user asks about weather, temperature, forecast, “what’s it like now”, or anything that needs current weather context.

### Tool 2 — `retrieve_weather_activity_clothing_info(query: str)`

* **Purpose:** Retrieve clothing and activity recommendations from the vector knowledge base.
* **Source:** Cassandra/Astra vector store (Cassio) with embeddings + Cohere reranking.
* **Used when:** The user asks what to wear and/or what activities are suitable **based on weather**.

### Tool 3 — `internet_search(query: str)`

* **Purpose:** Answer general informational questions unrelated to weather/clothing/activity.
* **Source:** DuckDuckGo Instant Answer API (lightweight lookup).
* **Used when:** The user asks about topics unrelated to weather/clothing/activities.

**Critical Routing Rule**

* Weather/clothing/activity requests must **never** call `internet_search`.
* If the user request is weather/clothing/activity-based and includes a valid location, the assistant must call `weather_query(location)` immediately.

---

## Key Features

* **Tool-first weather flow:** If a valid location is provided and the request is weather/clothing/activity-related, the assistant calls `weather_query` immediately.
* **RAG-powered recommendations:** Clothing/activity guidance is retrieved from a vectorized guide and reranked for relevance.
* **Safe location handling:** Prevents tool calls with invalid placeholder locations (`"unknown"`, `"?"`, `"n/a"`, empty strings).
* **True streaming support:** Token/event streaming via LangGraph `astream_events`, exposed through a FastAPI SSE endpoint.
* **Clean UI:** Streamlit chat bubbles, optional reasoning expander, and stable session history.

---

## Architecture

High-level flow:

1. User sends a message (Streamlit UI or API client).

2. The agent routes the request:

   * Weather-only → `weather_query(location)` → “Weather Snapshot”
   * Clothing/activities → `weather_query(location)` → build weather context → `retrieve_weather_activity_clothing_info(query)`
   * Non-weather informational → `internet_search(query)`

3. Tools return results.

4. Agent formats and returns a structured response.

---

## Core Components

### 1) Agent (`WeatherActivityClothingAgent`)

* Loads environment variables from `.env`
* Initializes:

  * OpenWeatherMap wrapper
  * Groq LLM with tool binding
  * Cassandra vector store
  * Retriever tool with Cohere reranking
* Builds a LangGraph state machine:

  * `ai_agent` node (LLM call)
  * `tools` node (executes tool calls)
  * Conditional routing via `tools_condition`

### 2) Tools

* `weather_query(location: str)`
  Fetches current weather data for a given location.

* `retrieve_weather_activity_clothing_info(query: str)`
  Retrieves relevant clothing/activity guidance from the knowledge base via vector search + reranking.

* `internet_search(query: str)`
  Lightweight web lookup for general questions outside the weather/clothing/activity scope.

---

## API

Base prefix: `/api/v1`

### `POST /api/v1/chat` (Non-stream)

Returns a full response as JSON.

**Request**

```json
{ "message": "What should I wear today in Egypt?" }
```

**Response**

```json
{ "answer": "<reasoning>...</reasoning>\nWeather Snapshot...\n..." }
```

### `POST /api/v1/chat/stream` (SSE Streaming)

True streaming endpoint (Server-Sent Events). Emits incremental data frames:

* `data: {"type":"status","value":"started"}`
* `data: {"type":"delta","value":"..."}`  (token/partial text)
* `data: {"type":"done"}`
* `data: {"type":"error","message":"..."}`

**Example**

```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"What should I wear today in Egypt?"}'
```

---

## Streamlit UI

The Streamlit app is a chat interface that supports:

* Session-based history
* Streaming output in the assistant chat bubble
* Optional reasoning expander (when `<reasoning>...</reasoning>` exists)

### Supported Modes

1. **API mode (recommended)**
   Streamlit calls FastAPI endpoints:

* `/api/v1/chat` (non-stream)
* `/api/v1/chat/stream` (SSE streaming)

2. **Local agent mode (optional)**
   Streamlit imports and runs the agent directly (useful for local experiments).

---

## Knowledge Base & RAG Pipeline

### Data Sources

The knowledge base is built from two PDF guides that provide clothing/activity recommendations across multiple weather scenarios.

### Extraction & Chunking (Current Implementation)

We use a dedicated extraction module that:

* Loads PDFs via `PyMuPDF4LLMLoader`
* Applies file-specific chunkers
* Saves output as **JSONL** ready for ingestion/indexing

**Chunkers**

* `src/chunker/first_pdf_chuncker.py`
  For the “Comprehensive Global Guide” format (`##` weather → `###` country → temperature blocks).

* `src/chunker/second_pdf_chuncker.py`
  For the bullet-style database format (`**1. Weather**` → `Country:` → `Outdoor Activities / Appropriate Clothing`).

**Outputs**

* Saved under: `src/data/out_chunks/*.jsonl`

### Indexing Pipeline (Conceptual)

* Load JSONL chunks
* Embeddings: `sentence-transformers/all-mpnet-base-v2`
* Vector store: Cassandra/Astra via Cassio
* Retrieval:

  * KNN retrieval (`k = retriever_k`)
  * Cohere reranking (`top_n = rerank_top_n`)
  * Contextual compression retriever

---

## Configuration

Create a `.env` file in the project root.

Required variables:

```env
GROQ_API_KEY=your_key
OPENWEATHERMAP_API_KEY=your_key
COHERE_API_KEY=your_key
CASSIO_DB_ID=your_db_id
CASSIO_TOKEN=your_token
```

Notes:

* `OPENAI_API_KEY` is **not required** for the current implementation unless you add OpenAI-dependent components later.
* Free tiers may impose rate limits.

---

## Installation

### Using `uv` (recommended)

```bash
uv venv
uv pip install -r requirements.txt
```

### Using `pip`

```bash
python -m venv weather_env
# Linux/macOS
source weather_env/bin/activate
# Windows
weather_env\Scripts\activate

pip install -r requirements.txt
```

---

## Run

### 1) Run the API

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

### 2) Run Streamlit

```bash
streamlit run app.py
```

API mode environment variable:

```bash
# Linux/macOS
API_BASE_URL="http://localhost:8000" streamlit run app.py

# Windows PowerShell
$env:API_BASE_URL="http://localhost:8000"
streamlit run app.py
```

---

## Testing

Quick functional checks:

1. Weather query:

* “What’s the weather now in Doha?”

2. Clothing + activities:

* “What should I wear today in Cairo? What can I do outside?”

3. Non-weather informational question:

* “What is machine learning?”

Streaming check:

```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"What should I wear today in Egypt?"}'
```

---

## Troubleshooting

### `Agent not initialized`

* Ensure FastAPI startup ran and the agent was created successfully.
* Verify `.env` variables exist and are loaded.

### Missing API keys

The agent will raise a `ValueError` if any required key is missing:

* `CASSIO_DB_ID`, `CASSIO_TOKEN`
* `GROQ_API_KEY`
* `OPENWEATHERMAP_API_KEY`
* `COHERE_API_KEY`

### Streamlit shows no streaming / delayed tokens

* Verify streaming with `curl -N` to confirm SSE works.
* Ensure no proxy buffers responses (reverse proxies can buffer SSE).
* Ensure the Streamlit client reads the response incrementally (`stream=True`, `iter_lines()`).

### JSONL extraction produces `0 chunks`

* This usually indicates a **format mismatch** between the PDF extraction text and the chunker regex.
* Validate the extracted raw text and confirm the chunker pattern matches the real headings/markers.

---

## Project Structure

```text
Weather-RAG/
├─ app.py
├─ agent.py                
├─ src/
│  ├─ agent/
│  │  └─ weather_agent.py
│  ├─ prompts/
│  │  └─ system_prompt.py
│  ├─ tools/
│  │  ├─ weather.py
│  │  └─ search.py
│  ├─ rag/
│  │  └─ builder.py
│  ├─ chunker/
│  │  ├─ first_pdf_chuncker.py
│  │  └─ second_pdf_chuncker.py
│  ├─ extract_text/
│  │  └─ extract.py         
│  └─ api/
│     ├─ main.py
│     └─ routes/
│        ├─ base_route.py
│        ├─ chat.py
│        └─ module/
│           └─ schema.py
└─ pyproject.toml
```

---

## License & Credits

* Built by Amgad Shalaby
* Powered by LangChain/LangGraph, OpenWeatherMap, Groq, HuggingFace, Cassio (Astra DB), and Cohere

---