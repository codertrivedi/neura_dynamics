from dotenv import load_dotenv
from langsmith import Client
import os
from src.config import LANGSMITH_API_KEY
from langsmith.run_helpers import get_current_run_tree
from langsmith import traceable

load_dotenv()

@traceable(run_type="chain", name="evaluate_response")
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
        # Classify query type based on weather-related keywords
        weather_keywords = ['weather', 'temperature', 'temp', 'humidity', 'wind', 'rain', 'sunny', 'cloudy', 'forecast']
        query_type = 'weather' if any(keyword in query.lower() for keyword in weather_keywords) else 'document'
        
        # Log evaluation locally
        print(f"📊 Response Quality Score: {score:.2f}/1.0")
        print(f"📊 Query Type: {query_type}")
        
        # Send evaluation data to LangSmith
        client = Client()
        
        # Try to get the current run to attach feedback
        try:
            current_run = get_current_run_tree()
            if current_run:
                # Create feedback for the current run
                client.create_feedback(
                    run_id=current_run.id,
                    key="response_quality_score",
                    score=score,
                    comment=f"Query Type: {query_type}, Response Length: {len(response)} chars"
                )
                
                # Create additional feedback for query type
                client.create_feedback(
                    run_id=current_run.id,
                    key="query_type",
                    value=query_type,
                    comment=f"Automatically classified query type based on content"
                )
                
                print(f"✅ Evaluation data sent to LangSmith (Run ID: {current_run.id})")
            else:
                print("⚠️ No current run found, evaluation data not sent to LangSmith")
        except Exception as feedback_error:
            print(f"⚠️ Failed to send feedback to LangSmith: {feedback_error}")
        
        return score
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
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
