# test_rag.py

from src.nodes.rag_node import query_rag
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def test_query_rag_returns_result():
    docs = [Document(page_content="LangChain is a framework for LLM orchestration.")]
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name="test_docs"
    )

    result = query_rag(vectordb, "What is LangChain?")
    assert "LangChain" in result
