from cassio.table.cql import STANDARD_ANALYZER
from langchain_community.vectorstores import Cassandra
from langchain_core.tools import create_retriever_tool
from langchain_cohere import CohereRerank

try:
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
except Exception:
    from langchain_classic.retrievers import ContextualCompressionRetriever


def build_vectorstore(*, embeddings, table_name: str):
    return Cassandra(
        embedding=embeddings,
        table_name=table_name,
        body_index_options=[STANDARD_ANALYZER],
    )


def build_retriever_tool(
    *,
    vectorstore: Cassandra,
    retriever_k: int,
    rerank_model: str,
    rerank_top_n: int,
):
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    compressor = CohereRerank(model=rerank_model, top_n=rerank_top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )

    return create_retriever_tool(
        compression_retriever,
        name="retrieve_weather_activity_clothing_info",
        description=(
        "This tool retrieves contextually relevant and compressed information about recommended outdoor activities "
        "and appropriate clothing based on current weather conditions and location. "
        "It leverages a comprehensive global guide covering multiple weather scenarios—such as sunny, rainy, snowy, "
        "windy, cloudy, hot, and cold conditions—tailored for diverse countries including Egypt, UK, USA, Japan, and Australia. "
        "The recommendations ensure users get personalized, climate-specific advice for comfort and safety during outdoor plans."
        ),
    )
