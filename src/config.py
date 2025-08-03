import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OWM_API_KEY = os.getenv("OWM_API_KEY")
LANGCHAIN_PROJECT = "ai-pipeline"
LANGSMITH_API_KEY= os.getenv("LANGSMITH_API_KEY")