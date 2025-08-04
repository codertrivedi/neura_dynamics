from langgraph.graph import StateGraph
from typing import TypedDict
from src.nodes.weather_node import fetch_weather
from src.nodes.rag_node import query_rag
from src.nodes.decision_node import decide_query_type
from src.utils.city_parser import extract_city_name
from langgraph.graph import START, END

class GraphState(TypedDict):
    query: str
    branch: str
    response: str

def build_graph(vectorstore):
    graph = StateGraph(GraphState)

    # Decision node - this should return the state with updated branch
    def decision_node(state: GraphState) -> GraphState:
        branch = decide_query_type(state["query"])
        return {**state, "branch": branch}
    
    # Weather node - fetch weather data and store in vector DB
    def weather_node(state: GraphState) -> GraphState:
        try:
            print("Weather node called")
            print(f"🔍 Original query: '{state['query']}'")
            
            # Use spaCy-based city extraction
            city = extract_city_name(state["query"])
            print(f"🔍 Final city to fetch: '{city}'")
            
            # Fetch weather data
            weather_data = fetch_weather(city)
            
            # If successful, store in vector DB
            if isinstance(weather_data, dict):
                from src.utils.qdrant_utils import store_weather_data
                store_weather_data(weather_data, city)
            
            # Now use RAG to answer the query
            response = query_rag(vectorstore, state["query"])
            return {**state, "response": response}
            
        except Exception as e:
            error_response = f"Sorry, I couldn't fetch weather information for the requested location. Error: {str(e)}"
            return {**state, "response": error_response}
    
    # RAG node - should return updated state with response
    def rag_node(state: GraphState) -> GraphState:
        print("RAG node called")
        response = query_rag(vectorstore, state["query"])
        return {**state, "response": response}

    # Register nodes
    graph.add_node("decide", decision_node)
    graph.add_node("weather", weather_node)
    graph.add_node("rag", rag_node)

    # Set entry point
    graph.add_edge(START, "decide")

    # Add conditional edges - the function should take state and return the branch value
    def route_query(state: GraphState) -> str:
        return state["branch"]
    
    graph.add_conditional_edges(
        "decide",
        route_query,
        {
            "weather": "weather",
            "rag": "rag"
        }
    )

    # Add terminal edges
    graph.add_edge("weather", END)
    graph.add_edge("rag", END)

    return graph.compile()