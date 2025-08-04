from src.utils.llm_2 import process_weather_data, process_rag_response

def test_process_weather_data():
    weather_data = {
        "weather": [{"description": "clear sky"}],
        "main": {"temp": 25, "feels_like": 27, "humidity": 60},
        "wind": {"speed": 3.5}
    }
    
    result = process_weather_data(weather_data, "Delhi")
    assert "Delhi" in result
    assert "25" in result or "25°" in result
    assert len(result) > 50

def test_process_weather_data_string_input():
    result = process_weather_data("Weather data unavailable", "Mumbai")
    assert "Weather data unavailable" in result

def test_process_rag_response():
    rag_content = "The document discusses AI frameworks and their applications in modern software development."
    query = "What does the document say about AI?"
    
    result = process_rag_response(rag_content, query)
    assert len(result) > 20
    assert isinstance(result, str)