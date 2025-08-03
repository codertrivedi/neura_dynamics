from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from src.config import GROQ_API_KEY

def process_weather_data(weather_data, city):
    """Process raw weather data using LLM to generate a natural response"""
    try:
        llm = ChatGroq(
            model="llama3-8b-8192",
            api_key=GROQ_API_KEY,
            temperature=0.7
        )
        
        if isinstance(weather_data, dict):
            weather_desc = weather_data.get('weather', [{}])[0].get('description', 'unknown')
            temp = weather_data.get('main', {}).get('temp', 'unknown')
            feels_like = weather_data.get('main', {}).get('feels_like', temp)
            humidity = weather_data.get('main', {}).get('humidity', 'unknown')
            
            prompt = f"""
            Generate a natural, conversational weather response for {city} based on this data:
            - Description: {weather_desc}
            - Temperature: {temp}°C
            - Feels like: {feels_like}°C
            - Humidity: {humidity}%
            
            Make it friendly and informative, like a weather forecaster would say.
            """
        else:
            # Handle error cases where weather_data is a string
            return str(weather_data)
        
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        return f"The weather in {city} is {weather_desc} with temperature {temp}°C."

def process_rag_response(retrieved_content, query):
    """Process retrieved PDF content using LLM to generate a summarized response"""
    try:
        llm = ChatGroq(
            model="llama3-8b-8192",
            api_key=GROQ_API_KEY,
            temperature=0.5
        )
        
        prompt = f"""
        Based on the following retrieved document content, answer the user's question: "{query}"
        
        Document content:
        {retrieved_content}
        
        Please provide a clear, concise answer based only on the information in the document. 
        If the document doesn't contain relevant information, say so.
        """
        
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        return f"Error processing document content: {str(e)}"

def summarize_content(content, content_type="text"):
    """General purpose content summarization using LLM"""
    try:
        llm = ChatGroq(
            model="llama3-8b-8192",
            api_key=GROQ_API_KEY,
            temperature=0.5
        )
        
        prompt = f"""
        Please summarize the following {content_type} content in a clear and concise way:
        
        {content}
        
        Focus on the key points and main information.
        """
        
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        return f"Error summarizing content: {str(e)}"

if __name__ == "__main__":
    # Test the functions
    sample_weather = {
        'weather': [{'description': 'clear sky'}],
        'main': {'temp': 25.5, 'feels_like': 27.0, 'humidity': 65}
    }
    
    print("Weather processing test:")
    print(process_weather_data(sample_weather, "Delhi"))
    
    print("\nRAG processing test:")
    print(process_rag_response("This document discusses AI and machine learning applications.", "What is this document about?"))