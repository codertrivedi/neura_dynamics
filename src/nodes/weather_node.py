import requests
from src.config import OWM_API_KEY

def fetch_weather(city):
    # Debug: Print what city we're looking for
    print(f"🌍 Fetching weather for city: '{city}' (length: {len(city)})")
    
    # Clean and format city name
    city_clean = city.strip().title()
    print(f"🌍 Cleaned city name: '{city_clean}'")
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_clean}&appid={OWM_API_KEY}&units=metric"
    print(f"🌍 API URL: {url[:80]}...")
    
    try:
        response = requests.get(url)
        data = response.json()
        print(f"🌍 API Response status: {response.status_code}")
        print(f"🌍 API Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

        if response.status_code != 200 or data.get('cod') != 200:
            return f"Sorry, couldn't fetch weather for {city}. API error: {data.get('message', response.status_code)}"

        weather = data.get('weather', [])
        main = data.get('main', {})

        if not weather or 'description' not in weather[0] or 'temp' not in main:
            return f"Sorry, received incomplete weather data for {city}."

        return data
    
    except requests.RequestException as e:
        return f"Sorry, network error while fetching weather for {city}: {e}"
    except Exception as e:
        return f"Sorry, unexpected error while fetching weather for {city}: {e}"

if __name__ == "__main__":
    for city in ["Bangalore", "Delhi"]:
        print(fetch_weather(city))

