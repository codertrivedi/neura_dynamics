from src.utils.pdf_parser import load_pdf_text
from src.utils.embedding_utils import generate_embeddings
from src.langgraph_pipeline import build_graph
from src.evaluation.langsmith_eval import evaluate_output
from src.utils.llm_2 import process_weather_data, process_rag_response, summarize_content

def simulate_pipeline(query: str):
    print(f"\n➡️  Query: {query}")

    # Step 1: Load PDF & Embed
    texts = load_pdf_text()
    vectordb = generate_embeddings(texts)

    # Step 2: Build LangGraph pipeline with decision node
    graph = build_graph(vectordb)

    # Step 3: Run the graph
    result = graph.invoke({"query": query})
    raw_response = result["response"]
    
    # Step 4: Process response based on query type
    if "weather" in query.lower():
        # Weather response - format if it's raw data
        if isinstance(raw_response, dict):
            city = query.split()[-1].rstrip("?")
            processed_response = process_weather_data(raw_response, city)
        else:
            processed_response = raw_response
    else:
        # RAG response - process with LLM for better summarization
        processed_response = process_rag_response(raw_response, query)
    
    print(f"✅ Raw Response: {raw_response}")
    print(f"🤖 LLM Processed Response: {processed_response}")

    # # Step 4: Evaluate via LangSmith
    # print("📊 Evaluating with LangSmith...")
    # run_url = evaluate_output(query, response)
    # print(f"🔗 LangSmith Evaluation: {run_url}")

if __name__ == "__main__":
    # Test Case 1: Weather query
    simulate_pipeline("What is the weather in Delhi?")

    # Test Case 2: PDF-based question
    simulate_pipeline("What are the key points in the document?")