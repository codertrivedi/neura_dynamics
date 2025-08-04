# test_weather_api.py

from src.nodes.weather_node import fetch_weather

def test_fetch_weather_success(monkeypatch):
    def mock_get(url):
        class MockResponse:
            status_code = 200
            def json(self):
                return {
                    "cod": 200,
                    "weather": [{"description": "clear sky"}],
                    "main": {"temp": 27}
                }
        return MockResponse()
    
    monkeypatch.setattr("requests.get", mock_get)
    
    result = fetch_weather("Delhi")
    assert isinstance(result, dict)
    assert result["weather"][0]["description"] == "clear sky"
    assert result["main"]["temp"] == 27
