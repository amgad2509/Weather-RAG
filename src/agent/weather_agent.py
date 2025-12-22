import os
import inspect
from typing import Any, List, Tuple, Dict
from src.utils.telemetry import Stopwatch
from dotenv import load_dotenv

import cassio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END, MessagesState

from ..prompts import PROMPT
from ..tools import make_weather_query_tool, internet_search, dummy_weather
from ..rag import build_vectorstore, build_retriever_tool
from src.utils.source_parsers import (
    parse_sources_from_internet_output,
    parse_sources_from_retriever_output,
)
from src.utils.telemetry import Stopwatch

load_dotenv()


class WeatherActivityClothingAgent:
    def __init__(
        self,
        table_name: str = "weather_data",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        retriever_k: int = 8,
        rerank_top_n: int = 4,
        rerank_model: str = "rerank-english-v3.0",
        groq_model: str = "openai/gpt-oss-120b",
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
        self.prompt = PROMPT

        # -------------------------
        # LLM (best-effort enable streaming flag)
        # -------------------------
        llm_kwargs = dict(
            model=groq_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        try:
            sig = inspect.signature(ChatGroq.__init__)
            if "streaming" in sig.parameters:
                llm_kwargs["streaming"] = True
            elif "stream" in sig.parameters:
                llm_kwargs["stream"] = True
        except Exception:
            llm_kwargs["streaming"] = True

        self.llm = ChatGroq(**llm_kwargs)

        # -------------------------
        # WEATHER + EMBEDDINGS + VECTORSTORE
        # -------------------------
        self.weather = OpenWeatherMapAPIWrapper()
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.vectorstore = build_vectorstore(embeddings=self.embeddings, table_name=table_name)

        self.retriever_tool = build_retriever_tool(
            vectorstore=self.vectorstore,
            retriever_k=retriever_k,
            rerank_model=rerank_model,
            rerank_top_n=rerank_top_n,
        )

        # -------------------------
        # TOOLS
        # -------------------------
        self.weather_query = make_weather_query_tool(self.weather)
        self.dummy_weather = dummy_weather
        self.internet_search = internet_search

        self.tools = [self.weather_query, self.dummy_weather, self.retriever_tool, self.internet_search]
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)

        # -------------------------
        # LANGGRAPH (SYNC)
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

    def _find_last_human(self, msgs: List[Any]) -> Tuple[int, str]:
        last_human = None
        last_human_idx = -1

        for idx in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[idx], HumanMessage):
                last_human = msgs[idx].content
                last_human_idx = idx
                break

        return last_human_idx, last_human

    def _collect_internet_search_outputs(self, msgs: List[Any], start_idx: int) -> List[str]:
        outputs: List[str] = []

        for msg in msgs[start_idx:]:
            name = getattr(msg, "name", None)
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", None)

            is_tool_msg = msg_type == "tool" or name
            if not is_tool_msg:
                continue

            if name == "internet_search":
                if isinstance(content, list):
                    content = "".join(str(c) for c in content)
                if content:
                    outputs.append(str(content))

        return outputs

    def _collect_retriever_outputs(self, msgs: List[Any], start_idx: int) -> List[Any]:
        outputs: List[Any] = []

        for msg in msgs[start_idx:]:
            name = getattr(msg, "name", None)
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", None)

            is_tool_msg = msg_type == "tool" or name
            if not is_tool_msg:
                continue

            if name == "retrieve_weather_activity_clothing_info":
                outputs.append(content)

        return outputs

    def _extract_retriever_queries_from_tool_calls(self, tool_calls: Any) -> List[str]:
        queries: List[str] = []
        if not tool_calls:
            return queries

        calls = tool_calls if isinstance(tool_calls, list) else []
        for call in calls:
            try:
                name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else None)
                if name != "retrieve_weather_activity_clothing_info":
                    continue
                args = getattr(call, "args", None) or (call.get("args") if isinstance(call, dict) else {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if isinstance(args, dict):
                    q = args.get("query")
                    if q:
                        queries.append(str(q))
            except Exception:
                continue
        return queries

    def _vectorstore_sources_from_queries(self, queries: List[str], k: int = 5) -> List[Dict[str, str]]:
        sources: List[Dict[str, str]] = []
        seen = set()

        for q in queries:
            if len(sources) >= k:
                break
            try:
                docs = self.vectorstore.similarity_search(q, k=k)
            except Exception:
                continue

            for doc in docs:
                if len(sources) >= k:
                    break
                meta = getattr(doc, "metadata", {}) or {}
                url = (
                    meta.get("url")
                    or meta.get("source")
                    or meta.get("link")
                    or meta.get("path")
                    or meta.get("document_id")
                    or meta.get("row_id")
                )
                name = meta.get("title") or meta.get("file_name") or meta.get("filename") or url
                if not url:
                    url = f"chunk:{len(sources)+1}"
                if str(url) in seen:
                    continue
                sources.append({"name": name or url, "url": str(url)})
                seen.add(str(url))

        return sources

    def invoke_with_sources(self, user_input: str) -> Tuple[str, List[dict], Dict[str, int]]:
        sw_total = Stopwatch()
        metrics: Dict[str, int] = {"total_ms": 0, "llm_ms": 0, "retrieve_ms": 0}

        state = {"messages": [HumanMessage(content=user_input)]}
        sw_llm = Stopwatch()
        out = self.app.invoke(state)
        metrics["llm_ms"] = sw_llm.ms()
        msgs = out.get("messages", [])

        last_human_idx, last_human = self._find_last_human(msgs)

        last_ai = None
        last_tool_calls = None
        for idx in range(len(msgs) - 1, last_human_idx, -1):
            msg = msgs[idx]
            if last_ai is None and getattr(msg, "type", "") == "ai":
                last_ai = msg.content
            if last_tool_calls is None and isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    last_tool_calls = tool_calls
            if last_ai and last_tool_calls:
                break

        internet_outputs = self._collect_internet_search_outputs(msgs, last_human_idx + 1)
        internet_sources: List[dict] = []
        for output in internet_outputs:
            internet_sources.extend(parse_sources_from_internet_output(output))

        # For now, we only surface sources from internet_search to avoid noisy RAG/source placeholders.
        merged_sources = internet_sources

        if last_tool_calls:
            print(f"[debug] tool_calls: {last_tool_calls}")
        if last_human:
            print(f"[debug] last_human: {last_human}")
        if merged_sources:
            print(f"[debug] sources: {merged_sources}")

        metrics["total_ms"] = sw_total.ms()

        if last_ai is not None:
            return last_ai, merged_sources, metrics
        if last_tool_calls is not None:
            return str(last_tool_calls), merged_sources, metrics
        if last_human is not None:
            return last_human, merged_sources, metrics
        return (msgs[-1].content if msgs else ""), merged_sources, metrics

    def invoke(self, user_input: str) -> str:
        answer, _, _ = self.invoke_with_sources(user_input)
        return answer

    def __call__(self, user_input: str) -> str:
        return self.invoke(user_input)
