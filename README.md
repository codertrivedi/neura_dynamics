# Neura Dynamics AI Assistant

An AI-powered assistant that combines weather data retrieval and document Q&A using RAG (Retrieval-Augmented Generation) with LangGraph pipeline.

## Features

- Weather information retrieval
- PDF document Q&A using RAG
- Interactive Streamlit web interface
- LangSmith evaluation and tracing

## Setup

### Prerequisites

- Python 3.8+
- OpenWeatherMap API key
- OpenAI API key
- LangSmith API key (optional, for evaluation)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neura_dynamics
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key
OWM_API_KEY=your_openweathermap_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

## Running the Application

### Web Interface (Streamlit)
```bash
streamlit run app.py
```

### Command Line Interface
```bash
python main.py
```

## Usage

The AI assistant can handle two types of queries:

1. **Weather queries**: "What's the weather in Delhi?"
2. **Document Q&A**: "What are the key points in the document?"

## Project Structure

- `src/` - Core application code
- `data/` - Document storage
- `tests/` - Unit tests
- `qdrant_data/` - Vector database storage