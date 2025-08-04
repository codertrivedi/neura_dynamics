import streamlit as st
import sys
import os
from pathlib import Path
from PIL import Image

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"Could not load .env file: {e}")

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "neura-dynamics-ai"

# Check if LangSmith is configured
if 'LANGSMITH_API_KEY' in os.environ:
    print("✅ LangSmith tracing enabled")
else:
    print("⚠️ LangSmith API key not found")

# Debug: Check if API key is available
if 'OWM_API_KEY' in os.environ:
    print(f"✅ Weather API key loaded (length: {len(os.environ['OWM_API_KEY'])})")
else:
    print("❌ Weather API key NOT found in environment")

from src.utils.pdf_parser import load_pdf_text
from src.utils.embedding_utils import generate_embeddings
from src.langgraph_pipeline import build_graph
from src.utils.llm_2 import process_weather_data, process_rag_response
from src.evaluation.langsmith_eval import evaluate_output

im = Image.open(r"S:\project\neura_dynamics\neura_dynamics\neura.png")

def initialize_pipeline():
    """Initialize the pipeline components"""
    if 'pipeline_initialized' not in st.session_state:
        with st.spinner("Initializing..."):
            try:
                texts = load_pdf_text()
                vectordb = generate_embeddings(texts)
                graph = build_graph(vectordb)
                
                st.session_state.pipeline_initialized = True
                st.session_state.graph = graph
                
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                st.session_state.pipeline_initialized = False
                return False
    
    return st.session_state.pipeline_initialized

def process_query(query: str):
    """Process a user query through the pipeline"""
    try:
        result = st.session_state.graph.invoke({"query": query})
        raw_response = result.get("response", "No response generated")
        
        print(f"🔍 Raw response: {raw_response}")
        print(f"🔍 Response type: {type(raw_response)}")
        
        # Process response based on query type
        if "weather" in query.lower():
            if isinstance(raw_response, dict):
                city = query.split()[-1].rstrip("?")
                processed_response = process_weather_data(raw_response, city)
                print(f"🌤️ Processed weather data for {city}")
            else:
                processed_response = raw_response
                print(f"⚠️ Weather response was not dict: {raw_response}")
        else:
            processed_response = process_rag_response(raw_response, query)
            print(f"📄 Processed RAG response")
        
        # Evaluate within the traced context
        try:
            evaluation_score = evaluate_output(query, processed_response)
            if evaluation_score:
                print(f"📊 LangSmith evaluation completed (Score: {evaluation_score:.2f})")
                # Store evaluation score for Streamlit display
                st.session_state.last_evaluation_score = evaluation_score
        except Exception as eval_error:
            print(f"⚠️ LangSmith evaluation failed: {eval_error}")
            st.session_state.last_evaluation_score = None
        
        return processed_response
        
    except Exception as e:
        print(f"❌ Error in process_query: {e}")
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Neura Dynamics",
        page_icon=im,
        layout="centered"
    )
    
    st.title("🤖 AI Assistant")
    
    # Initialize pipeline
    if not initialize_pipeline():
        st.stop()
    
    # Query input
    query = st.text_input(
        "Ask me anything:",
        placeholder="What's the weather in Delhi? or What are the key points in the document?"
    )
    
    if st.button("Submit", type="primary") and query:
        with st.spinner("Processing..."):
            response = process_query(query)
            st.markdown(f"**Answer:** {response}")
            
            # Display evaluation score if available
            if hasattr(st.session_state, 'last_evaluation_score') and st.session_state.last_evaluation_score:
                st.info(f"📊 Response Quality Score: {st.session_state.last_evaluation_score:.2f}/1.0")

if __name__ == "__main__":
    main()