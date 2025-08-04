from src.nodes.decision_node import decide_query_type

def test_weather_query_detection():
    weather_queries = [
        "What is the weather in Delhi?",
        "What is the temperature today?",
        "How humid is it in Mumbai?",
        "Is it sunny outside?",
        "What's the wind speed?"
    ]
    
    for query in weather_queries:
        result = decide_query_type(query)
        assert result == "weather", f"Query '{query}' should be classified as weather"

def test_document_query_detection():
    document_queries = [
        "What are the key points in the document?",
        "Summarize the PDF content",
        "What does the document say about policies?",
        "List the main topics covered"
    ]
    
    for query in document_queries:
        result = decide_query_type(query)
        assert result == "rag", f"Query '{query}' should be classified as rag"

def test_edge_cases():
    assert decide_query_type("") == "rag"
    assert decide_query_type("Hello") == "rag"
    assert decide_query_type("WEATHER") == "weather"