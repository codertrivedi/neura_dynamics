from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import os

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vector_store = Qdrant(
            embeddings=embeddings,
            path=":memory:",
            collection_name="neura_dynamics_data"
        )
    return _vector_store

def initialize_with_documents(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    global _vector_store
    _vector_store = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        path=":memory:",
        collection_name="neura_dynamics_data"
    )
    
    return _vector_store

def store_weather_data(weather_data, city):
    try:
        weather_text = f"Weather data for {city} on {datetime.now().strftime('%Y-%m-%d %H:%M')}: Temperature {weather_data.get('main', {}).get('temp')}°C, feels like {weather_data.get('main', {}).get('feels_like')}°C, humidity {weather_data.get('main', {}).get('humidity')}%, weather conditions: {weather_data.get('weather', [{}])[0].get('description', 'unknown')}, wind speed {weather_data.get('wind', {}).get('speed')} m/s"
        
        vectorstore = get_vector_store()
        vectorstore.add_texts(
            [weather_text], 
            metadatas=[{"type": "weather", "city": city, "timestamp": datetime.now().isoformat()}]
        )
        
        print(f"Weather data stored in vector DB for {city}")
        return True
    except Exception as e:
        print(f"Failed to store weather data: {e}")
        return False

def search_similar(query, k=5):
    try:
        vectorstore = get_vector_store()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        return results
    except Exception as e:
        print(f"Failed to search vector store: {e}")
        return []

def clear_vector_store():
    global _vector_store
    _vector_store = None