import pytest
from unittest.mock import patch, MagicMock
from src.nodes.rag_node import query_rag
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def test_query_rag_no_results():
    vectordb = MagicMock()
    vectordb.as_retriever.return_value.invoke.return_value = []
    
    result = query_rag(vectordb, "What is missing?")
    assert result == "No relevant information found."

@patch('langchain_groq.ChatGroq')
def test_query_rag_with_results(mock_chat_groq):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "LangChain is a framework for building applications."
    mock_llm.invoke.return_value = mock_response
    mock_chat_groq.return_value = mock_llm
    
    with patch.object(Qdrant, 'as_retriever') as mock_retriever:
        mock_doc = MagicMock()
        mock_doc.page_content = "LangChain is a framework for LLM orchestration."
        mock_retriever.return_value.invoke.return_value = [mock_doc]
        vectordb = MagicMock()
        
        result = query_rag(vectordb, "What is LangChain?")
        assert result == "LangChain is a framework for building applications."
