import re

# Try to load spaCy - graceful fallback if not available
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully")
    except OSError:
        print("❌ spaCy model not found. Install with: pip install spacy && python -m spacy download en_core_web_sm")
        nlp = None
except ImportError:
    print("❌ spaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")
    nlp = None

def extract_city_name(query: str) -> str:
    """
    Extract city name from natural language query using spaCy NER.
    
    Args:
        query (str): Natural language query about weather
        
    Returns:
        str: Extracted city name or fallback extraction
    """
    if not nlp:
        # Fallback to original logic if spaCy not available
        return _fallback_city_extraction(query)
    
    try:
        # Process query with spaCy
        doc = nlp(query)
        
        # Extract GPE (Geopolitical entities) which include cities
        cities = []
        for ent in doc.ents:
            if ent.label_ == "GPE":
                city_name = ent.text.strip()
                cities.append(city_name)
                print(f"🏙️ Found GPE entity: '{city_name}' (confidence: {ent.label_})")
        
        # Return first city found
        if cities:
            city = cities[0]
            print(f"🏙️ Selected city: '{city}'")
            return city
        
        # If no GPE entities found, try fallback
        print("🏙️ No GPE entities found, using fallback extraction")
        return _fallback_city_extraction(query)
        
    except Exception as e:
        print(f"❌ Error in spaCy city extraction: {e}")
        return _fallback_city_extraction(query)

def _fallback_city_extraction(query: str) -> str:
    """
    Fallback city extraction logic (improved version of original).
    """
    query_lower = query.lower()
    
    # Common time/weather words to filter out
    time_words = {
        "today", "tomorrow", "now", "currently", "right", "now", 
        "this", "morning", "tonight", "weather", "temperature",
        "forecast", "what", "how", "is", "the", "in", "about",
        "tell", "me", "please", "current", "s"
    }
    
    # Known cities that spaCy might miss (lowercase)
    known_cities = {
        "mumbai", "delhi", "bangalore", "chennai", "kolkata",
        "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow"
    }
    
    if " in " in query_lower:
        # Extract everything after "in"
        city_part = query.split(" in ")[-1].strip().rstrip("?!.")
        # Split into words and filter out time/weather words
        words = city_part.split()
        city_words = [word for word in words if word.lower() not in time_words]
        
        if city_words:
            city = " ".join(city_words)
            print(f"🏙️ Fallback extraction using 'in': '{city}'")
            return city
    
    # Check for known cities in the query
    words = query.lower().split()
    for word in words:
        clean_word = word.rstrip("?!.,").strip()
        if clean_word in known_cities:
            # Return with proper capitalization
            proper_city = clean_word.title()
            print(f"🏙️ Fallback found known city: '{proper_city}'")
            return proper_city
    
    # Last resort: take last meaningful word
    words = query.split()
    for word in reversed(words):
        clean_word = word.rstrip("?!.").strip()
        if clean_word.lower() not in time_words and len(clean_word) > 1:
            print(f"🏙️ Fallback extraction using last valid word: '{clean_word}'")
            return clean_word
    
    # Ultimate fallback
    print("🏙️ Using ultimate fallback")
    return query.split()[-1].rstrip("?!.")

if __name__ == "__main__":
    # Test cases
    test_queries = [
        "What is the weather in delhi today?",
        "How's mumbai weather right now?",
        "Tell me about New York weather",
        "Weather in London tomorrow",
        "What is the current temperature in Tokyo?",
        "Delhi weather please",
        "What's the weather in San Francisco?",
        "How is the weather in Los Angeles today?"
    ]
    
    print("🧪 Testing city extraction:")
    print("=" * 60)
    for query in test_queries:
        city = extract_city_name(query)
        print(f"Query: '{query}'")
        print(f"→ City: '{city}'")
        print("-" * 60)