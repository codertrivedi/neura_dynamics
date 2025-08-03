from dotenv import load_dotenv
from langsmith import Client
import os
from src.config import LANGSMITH_API_KEY

load_dotenv()

def evaluate_output(query, response):
    """
    Simple evaluation that logs to LangSmith for tracking
    """
    try:
        # Check if LangSmith API key is available
        if not os.getenv("LANGSMITH_API_KEY"):
            print("LangSmith API key not found. Skipping evaluation.")
            return None
        
        # Calculate local score for immediate feedback
        score = _calculate_response_score(query, response)
        
        # Log evaluation locally (LangSmith will auto-trace if LANGCHAIN_TRACING_V2=true)
        print(f"📊 Response Quality Score: {score:.2f}/1.0")
        print(f"📊 Query Type: {'weather' if 'weather' in query.lower() else 'document'}")
        
        # The automatic tracing (enabled in app.py) will handle LangSmith logging
        return score
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None

def create_evaluation_dataset():
    """
    Create a LangSmith dataset for proper evaluation (run this once)
    """
    try:
        client = Client()
        
        # Create dataset for weather and document queries
        dataset = client.create_dataset(
            dataset_name="neura-dynamics-test-cases",
            description="Test cases for Neura Dynamics AI - weather and document queries"
        )
        
        # Sample test cases
        examples = [
            {
                "inputs": {"query": "What is the weather in Delhi?"},
                "outputs": {"expected_type": "weather", "should_contain": ["temperature", "delhi"]}
            },
            {
                "inputs": {"query": "What are the key points in the document?"},
                "outputs": {"expected_type": "document", "should_contain": ["key points", "document"]}
            },
            {
                "inputs": {"query": "How's the weather in Mumbai today?"},
                "outputs": {"expected_type": "weather", "should_contain": ["temperature", "mumbai"]}
            }
        ]
        
        # Add examples to dataset
        client.create_examples(dataset_id=dataset.id, examples=examples)
        print(f"✅ Created evaluation dataset: {dataset.id}")
        return dataset.id
        
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return None

def _calculate_response_score(query, response):
    """Calculate automatic response quality score (0.0 to 1.0)"""
    score = 0.5  # baseline
    
    # Check for error responses
    if "error" in response.lower() or "sorry" in response.lower():
        score -= 0.3
    
    # Check response length (too short or too long might be bad)
    if 50 < len(response) < 500:
        score += 0.2
    
    # Weather-specific checks
    if "weather" in query.lower():
        if any(word in response.lower() for word in ["temperature", "°c", "°f", "humidity", "wind"]):
            score += 0.3
    
    # Document-specific checks  
    if any(word in query.lower() for word in ["key points", "document", "summarize"]):
        if "•" in response or len(response.split('.')) > 2:
            score += 0.3
    
    return max(0.0, min(1.0, score))

if __name__ == "__main__":
    # Test evaluation
    test_response = "The current temperature in Mumbai is 27°C with clear skies. Humidity is 65% and wind speed is 10 km/h."
    score = evaluate_output("What is the weather in Mumbai?", test_response)
    print(f"Evaluation Score: {score}")
    
    # Uncomment to create evaluation dataset (run once)
    # dataset_id = create_evaluation_dataset()
    # print(f"Dataset created: {dataset_id}")
