from dotenv import load_dotenv
from langsmith import Client
import os
from src.config import LANGSMITH_API_KEY

load_dotenv()

def evaluate_output(query, response):
    try:
        # Check if LangSmith API key is available
        if not os.getenv("LANGSMITH_API_KEY"):
            print("LangSmith API key not found. Skipping evaluation.")
            return None
        
        client = Client()
        
        # Use the simpler log_feedback approach instead
        feedback = client.create_feedback(
            run_id=None,
            key="user_satisfaction",
            score=1.0,
            comment=f"Query: {query}\nResponse: {response}"
        )
        
        print(f"✅ LangSmith feedback logged")
        return feedback.id
        
    except Exception as e:
        print(f"LangSmith evaluation failed: {str(e)}")
        print("API key needs to be set. ")
        return None

if __name__ == "__main__":
    url = evaluate_output("What is the weather in Mumbai?", "It is 27°C with clear sky.")
    print(f"Evaluation URL: {url}")
