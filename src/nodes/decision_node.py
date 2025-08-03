def decide_query_type(query):
    if "weather" in query.lower():
        return "weather"
    return "rag"


if __name__ == "__main__":
    queries = [
        "What's the weather in Mumbai?",
        "Summarize the PDF content"
    ]
    for q in queries:
        print(f"{q} → {decide_query_type(q)}")
