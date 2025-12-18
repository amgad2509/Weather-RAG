import os
import inspect
from dotenv import load_dotenv

import cassio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END, MessagesState

from ..prompts import PROMPT
from ..tools import make_weather_query_tool, internet_search
from ..rag import build_vectorstore, build_retriever_tool

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
        self.internet_search = internet_search

        self.tools = [self.weather_query, self.retriever_tool, self.internet_search]
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
