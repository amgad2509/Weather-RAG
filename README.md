# Weather Activity & Clothing Assistant

An LLM-powered assistant that provides **current weather**, **clothing recommendations**, and **outdoor activity suggestions** based on a user’s location. The system combines real-time weather data, retrieval-augmented generation (RAG) over a curated knowledge base, and tool-calling orchestration.

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
* [Roadmap](#roadmap)
* [License & Credits](#license--credits)

---

## Overview

This project answers user queries such as:

* “What’s the weather now in Cairo?”
* “What should I wear today in Egypt?”
* “What outdoor activities are suitable in London right now?”

It uses:

* **OpenWeatherMap** for live weather conditions
* **Groq (ChatGroq)** for LLM reasoning + tool calling
* **Cassandra / Astra DB via Cassio** for vector storage
* **HuggingFace embeddings** for semantic indexing
* **Cohere Rerank** for contextual compression / reranking
* **LangGraph** to orchestrate tool execution
* **FastAPI** for backend endpoints (including true streaming)
* **Streamlit** for interactive chat UI

---

## Tooling Model

The assistant is designed around **exactly three tools**. The LLM is required to choose between them based on the user’s intent.

### Tool 1 — `weather_query(location: str)`

* **Purpose:** Fetch real-time weather data for a country/city.
* **Source:** OpenWeatherMap API.
* **Used when:** The user asks about weather, temperature, forecast, or anything that requires current weather context.

### Tool 2 — `retrieve_weather_activity_clothing_info(query: str)`

* **Purpose:** Retrieve clothing and activity recommendations from the vector knowledge base.
* **Source:** Cassandra/Astra vector store (Cassio) with embeddings + Cohere reranking.
* **Used when:** The user asks what to wear and/or what activities are suitable **based on weather**.

### Tool 3 — `internet_search(query: str)`

* **Purpose:** Answer general, non-weather informational questions.
* **Source:** DuckDuckGo Instant Answer API (lightweight lookup).
* **Used when:** The user asks about topics unrelated to weather/clothing/activities.

> Important behavior rule: **Weather/clothing/activity requests must never use `internet_search`.**

---

## Key Features

* **Tool-first weather flow:** When a valid location is provided and the request is weather/clothing/activity-related, the assistant calls `weather_query` immediately.
* **RAG-powered recommendations:** Clothing/activity guidance is retrieved from a vectorized guide and reranked for relevance.
* **Safe location handling:** Prevents calling weather tools with placeholder locations (e.g., `"unknown"`, `"?"`, empty strings).
* **True streaming support:** Token/event streaming via `LangGraph` `astream_events` exposed through an SSE endpoint.
* **Clean UI:** Streamlit chat bubbles, optional reasoning expander, and stable session history.

---

## Architecture

High-level flow:

1. User sends a message (via Streamlit or API).
2. Agent decides which tool(s) to call:

   * Weather-related requests → `weather_query(location)`
   * Clothing/activity requests → `weather_query` → build context → `retrieve_weather_activity_clothing_info`
   * Non-weather informational requests → `internet_search(query)`
3. Tools return results.
4. Agent formats and returns a structured final response.

---

## Core Components

### 1) Agent (`WeatherActivityClothingAgent`)

* Loads environment variables from `.env`
* Initializes:

  * OpenWeatherMap wrapper
  * Groq LLM with tool binding
  * Cassandra vector store
  * Retrieval tool with Cohere reranking
* Builds a LangGraph state machine:

  * `ai_agent` node (LLM call)
  * `tools` node (executes tool calls)
  * Conditional routing via `tools_condition`

### 2) Tools

* `weather_query(location: str)`
  Fetches real-time weather data for a given country/city.

* `retrieve_weather_activity_clothing_info(query: str)`
  Retrieves clothing/activity recommendations from the knowledge base (vector search + rerank + compression).

* `internet_search(query: str)`
  Performs lightweight web lookup for **non-weather** informational questions.

---

## API

Base prefix: `/api/v1`

### `POST /api/v1/chat`

Returns a full response (non-streaming).

**Request**

```json
{ "message": "What should I wear today in Egypt?" }
```

**Response**

```json
{ "answer": "<reasoning>...</reasoning>\nWeather Snapshot...\n..." }
```

### `POST /api/v1/chat/stream` (SSE)

True streaming endpoint. Returns events as **Server-Sent Events**:

* `data: {"type":"status","value":"started"}`
* `data: {"type":"delta","value":"..."}`
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
* Streaming output in the chat bubble
* Optional reasoning expander (when `<reasoning>...</reasoning>` exists)

You can run Streamlit in two modes:

1. **Local agent mode** (directly imports and runs the agent)
2. **API mode** (recommended for deployment)

   * Calls FastAPI `/api/v1/chat/stream` for true streaming

---

## Knowledge Base & RAG Pipeline

### Data Source

A guide/document (PDF) containing clothing/activity recommendations across diverse weather scenarios.

### Indexing Pipeline (conceptual)

* Load document content
* Chunking strategy (weather sections)
* Embeddings: `sentence-transformers/all-mpnet-base-v2`
* Vector store: Cassandra / Astra via Cassio
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

* `OPENAI_API_KEY` is **not required** for the current code path unless you add OpenAI-based components later.
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

If Streamlit uses the API server:

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

* Test streaming via `curl -N` to confirm server emits SSE correctly.
* Ensure no proxy buffers responses (common with reverse proxies).
* Ensure the Streamlit client reads `iter_lines()` with `stream=True`.

---

## Project Structure

```text
weather-chatbot-rag/
├─ app.py
├─ agent.py                      # legacy/standalone agent (optional)
├─ src/
│  ├─ agent/
│  │  └─ weather_agent.py
│  ├─ rag/
│  │  └─ builder.py
│  ├─ tools/
│  │  ├─ weather.py
│  │  └─ search.py
│  ├─ prompts/
│  │  └─ system_prompt.py
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
