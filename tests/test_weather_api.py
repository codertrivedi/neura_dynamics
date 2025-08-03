# test_weather_api.py

from src.nodes.weather_node import fetch_weather

def test_fetch_weather_success(monkeypatch):
    def mock_get(url):
        class MockResponse:
            def json(self):
                return {
                    "weather": [{"description": "clear sky"}],
                    "main": {"temp": 27}
                }
        return MockResponse()
    
    monkeypatch.setattr("requests.get", mock_get)
    
    result = fetch_weather("Delhi")
    assert "clear sky" in result
    assert "27" in result
