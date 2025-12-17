import os
import json
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv
import inspect

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool, create_retriever_tool

from langchain_groq import ChatGroq
from langchain_community.utilities import OpenWeatherMapAPIWrapper

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END, MessagesState

from langchain_community.vectorstores import Cassandra
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import cassio
from cassio.table.cql import STANDARD_ANALYZER

from langchain_cohere import CohereRerank

try:
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
except Exception:
    from langchain_classic.retrievers import ContextualCompressionRetriever

load_dotenv()


class WeatherActivityClothingAgent:
    def __init__(
        self,
        table_name: str = "weather_data",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        retriever_k: int = 8,
        rerank_top_n: int = 4,
        rerank_model: str = "rerank-english-v3.0",
        groq_model: str = "openai/gpt-oss-20b",
    ):
        # -------------------------
        # ENV / KEYS
        # -------------------------
        os.environ["GROQ_API_KEY"] = (os.getenv("GROQ_API_KEY") or "").strip()
        os.environ["OPENWEATHERMAP_API_KEY"] = (os.getenv("OPENWEATHERMAP_API_KEY") or "").strip()
        os.environ["COHERE_API_KEY"] = (os.getenv("COHERE_API_KEY") or "").strip()

        db_id = (os.getenv("CASSIO_DB_ID") or "").strip()
        token = (os.getenv("CASSIO_TOKEN") or "").strip()

        if not db_id:
            raise ValueError("Missing CASSIO_DB_ID in .env")
        if not token:
            raise ValueError("Missing CASSIO_TOKEN in .env")
        if not os.environ["GROQ_API_KEY"]:
            raise ValueError("Missing GROQ_API_KEY in .env")
        if not os.environ["OPENWEATHERMAP_API_KEY"]:
            raise ValueError("Missing OPENWEATHERMAP_API_KEY in .env")
        if not os.environ["COHERE_API_KEY"]:
            raise ValueError("Missing COHERE_API_KEY in .env")

        cassio.init(database_id=db_id, token=token)

        # -------------------------
        # PROMPT
        # -------------------------
        self.prompt = """
        You are a helpful assistant with access to three tools.

        CRITICAL RULES (Do NOT violate):
        - Do NOT ask the user for travel dates, season, specific cities/regions, or planned activities.
        - The ONLY clarification you may ask is the location (country/city) IF AND ONLY IF the user did not provide any location.
        - If the user provided a location (e.g., "Russia", "Moscow"), call weather_query immediately. No extra questions.
        - NEVER invent a location and NEVER call weather_query with placeholder/unknown values like: "?", "unknown", "n/a", "", or null.
          If no valid location is provided, ask exactly ONE question: "Which location (country/city)?"
        - Do NOT show internal tool calls, step numbers, logs, or debugging info in the user-facing answer. User should only see the final answer.

        -INTERNET SEARCH RULE (Mandatory):
            - If the user asks for ANY informational question (any topic/field), you MUST call internet_search immediately.
            - This applies even if you believe you already know the answer.
            - After internet_search returns results, answer the user based on those results.
            - EXCEPTION: If the user question is about current weather OR clothing/activities based on weather + location,
              follow the weather_query (+ retrieve_weather_activity_clothing_info when needed) flow instead (do NOT use internet_search).

        TOOLS:

        1) weather_query(location: str)
        - Fetches current weather details for the given location (country/city).
        - Requires ONLY the location string; do NOT request anything else.

        2) retrieve_weather_activity_clothing_info(query: str)
        - Retrieves recommended outdoor activities and appropriate clothing from the knowledge base.
        - Always pass location + weather context in the query you send.

        3) internet_search(query: str)
        - Use only when the user asks for information that likely requires internet lookup for any topic or any field like what is a machine learning ....  etc .
        
        TASK FLOW:

        A) If the user asks about clothing and/or activities AND provides a location:
        1) Call weather_query(location).
        2) Build a compact weather context (condition, temperature, feels-like, wind, precipitation if present).
        3) Call retrieve_weather_activity_clothing_info with a query that includes:
            - location
            - weather condition
            - temperature/feels-like
            Example query:
            "Russia | overcast | temp=-20C feels=-25C wind=1.7m/s precip=snow | clothing + activities recommendations"

        B) If the user asks about clothing/activities but NO location is provided:
        - Ask exactly ONE question: "Which location (country/city)?"
        - After receiving it, follow flow (A). Do NOT ask anything else.

        D) If the user asks about current weather (e.g., "What’s the weather now?"):
        - If a valid location is provided: call weather_query(location) and answer with a "Weather Snapshot".
        - If NO valid location is provided: ask exactly ONE question: "Which location (country/city)?"
        - Do NOT call retrieve_weather_activity_clothing_info unless the user also asks what to wear / activities.

        C) If the user asks a general question NOT related to weather/clothing/activities:
        - Answer from your own knowledge if possible.
        - Use internet_search only if the user asked for lookup or the info must be current.

        OUTPUT FORMAT (user-facing):
        - First, include a reasoning block in this exact format:
          <reasoning>
          (briefly explain your decision: which tool(s) you used and why)
          </reasoning>

        - Then provide:
          - A short "Weather Snapshot" (key values).
          - "What to Wear" as layered guidance + shoes + accessories.
          - "Activities" as suggestions (based on KB + current weather).
          - A "Quick Checklist".
        - Be clear, accurate, and concise.
        """

        # -------------------------
        # LLM (ENABLE STREAMING SAFELY)
        # -------------------------
        llm_kwargs = dict(
            model=groq_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Try to enable streaming without breaking if param name differs
        try:
            sig = inspect.signature(ChatGroq.__init__)
            if "streaming" in sig.parameters:
                llm_kwargs["streaming"] = True
            elif "stream" in sig.parameters:
                llm_kwargs["stream"] = True
        except Exception:
            # best-effort
            llm_kwargs["streaming"] = True

        self.llm = ChatGroq(**llm_kwargs)

        # -------------------------
        # WEATHER + EMBEDDINGS + VECTORSTORE
        # -------------------------
        self.weather = OpenWeatherMapAPIWrapper()
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.vectorstore = Cassandra(
            embedding=self.embeddings,
            table_name=table_name,
            body_index_options=[STANDARD_ANALYZER],
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": retriever_k})
        compressor = CohereRerank(model=rerank_model, top_n=rerank_top_n)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever,
        )

        self.retriever_tool = create_retriever_tool(
            compression_retriever,
            name="retrieve_weather_activity_clothing_info",
            description=(
                "Retrieves contextually relevant and compressed info about recommended outdoor activities "
                "and appropriate clothing based on weather conditions and location."
            ),
        )

        # -------------------------
        # TOOLS
        # -------------------------
        @tool
        def weather_query(location: str) -> str:
            """
            Fetches real-time weather data for a specified location using OpenWeatherMap API.
            """
            loc = (location or "").strip()
            bad = {"?", "unknown", "n/a", "na", "none", "null", ""}
            if loc.lower() in bad:
                return "ERROR: invalid location. Ask the user: Which location (country/city)?"
            return self.weather.run(loc)

        @tool
        def internet_search(query: str, max_related: int = 6) -> str:
            """
            Lightweight web lookup via DuckDuckGo Instant Answer API.
            Returns a compact summary + a few source links when available.
            """
            if not query or not query.strip():
                return "Error: empty query."

            base_url = "https://api.duckduckgo.com/"
            params = {
                "q": query.strip(),
                "format": "json",
                "no_html": 1,
                "no_redirect": 1,
                "skip_disambig": 1,
            }

            try:
                url = f"{base_url}?{urlencode(params)}"
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                data = r.json()
            except requests.RequestException as e:
                return f"Internet lookup failed (network/http): {e}"
            except ValueError:
                return "Internet lookup failed: response was not valid JSON."

            heading = (data.get("Heading") or "").strip()
            abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
            answer = (data.get("Answer") or "").strip()
            definition = (data.get("Definition") or "").strip()
            abstract_url = (data.get("AbstractURL") or "").strip()

            related_texts = []
            related = data.get("RelatedTopics") or []
            for item in related:
                if isinstance(item, dict) and "Topics" in item and isinstance(item["Topics"], list):
                    for t in item["Topics"]:
                        txt = (t.get("Text") or "").strip()
                        u = (t.get("FirstURL") or "").strip()
                        if txt:
                            related_texts.append((txt, u))
                elif isinstance(item, dict):
                    txt = (item.get("Text") or "").strip()
                    u = (item.get("FirstURL") or "").strip()
                    if txt:
                        related_texts.append((txt, u))

            lines = []
            title = heading if heading else query.strip()
            lines.append(f"Title: {title}")

            if answer:
                lines.append(f"Answer: {answer}")
            if definition:
                lines.append(f"Definition: {definition}")
            if abstract:
                lines.append(f"Abstract: {abstract}")
            if abstract_url:
                lines.append(f"Source: {abstract_url}")

            if related_texts:
                lines.append("Related:")
                for txt, u in related_texts[:max_related]:
                    lines.append(f"- {txt}" + (f" ({u})" if u else ""))

            if len(lines) <= 1:
                return "No instant-answer content found for this query. Try a more specific query."

            return "\n".join(lines)

        self.weather_query = weather_query
        self.internet_search = internet_search

        self.tools = [self.weather_query, self.retriever_tool, self.internet_search]
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)

        # -------------------------
        # LANGGRAPH
        # -------------------------
        def ai_agent(state: MessagesState) -> MessagesState:
            messages = [SystemMessage(content=self.prompt)] + state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        self.graph = StateGraph(MessagesState)
        self.graph.add_node("ai_agent", ai_agent)
        self.graph.add_node("tools", ToolNode(self.tools))

        self.graph.set_entry_point("ai_agent")
        self.graph.add_conditional_edges("ai_agent", tools_condition)
        self.graph.add_edge("tools", "ai_agent")
        self.graph.add_edge("ai_agent", END)

        self.app = self.graph.compile()

    def invoke(self, user_input: str) -> str:
        state = {"messages": [HumanMessage(content=user_input)]}
        out = self.app.invoke(state)
        msgs = out.get("messages", [])
        for m in reversed(msgs):
            if getattr(m, "type", "") == "ai":
                return m.content
        return msgs[-1].content if msgs else ""

    def __call__(self, user_input: str) -> str:
        return self.invoke(user_input)
