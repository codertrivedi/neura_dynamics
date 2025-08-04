def decide_query_type(query):
    import re
    
    # Core weather keywords with partial matching
    weather_roots = ['weather', 'temp', 'humid', 'wind', 'rain', 'sun', 'cloud', 'forecast', 'climat']
    
    query_lower = query.lower()
    
    # Check for partial matches using word boundaries
    for root in weather_roots:
        if re.search(rf'\b{root}\w*', query_lower):
            return "weather"
    
    # Additional weather indicators
    weather_patterns = [
        r'\b\d+°[cf]?\b',  # Temperature patterns like "25°C", "77°F", "30°"
        r'\bhow (hot|cold|warm|cool)\b',  # "how hot", "how cold"
        r'\bis it (raining|sunny|cloudy)\b'  # "is it raining"
    ]
    
    for pattern in weather_patterns:
        if re.search(pattern, query_lower):
            return "weather"
    
    return "rag"


if __name__ == "__main__":
    queries = [
        "What's the weather in Mumbai?",
        "What is the temperature today in Delhi?",
        "How humid is it in Bangalore?",
        "Is it sunny outside?",
        "Summarize the PDF content"
    ]
    for q in queries:
        print(f"{q} → {decide_query_type(q)}")
