import streamlit as st
import sys
from pathlib import Path
from PIL import Image

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from src.utils.pdf_parser import load_pdf_text
from src.utils.embedding_utils import generate_embeddings
from src.langgraph_pipeline import build_graph

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
        return result.get("response", "No response generated")
        
    except Exception as e:
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

if __name__ == "__main__":
    main()