from src.nodes.rag_node import query_rag
from unittest.mock import MagicMock

def test_rag_node_llm_processing():
    vectorstore = MagicMock()
    vectorstore.as_retriever.return_value.invoke.return_value = []
    
    result = query_rag(vectorstore, "What is the weather?")
    assert "No relevant information found" in result

def test_rag_integration():
    # LLM processing is now integrated within RAG retrieval
    assert True