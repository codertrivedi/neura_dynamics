from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from src.config import GROQ_API_KEY


def query_rag(vectorstore, query):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        results = retriever.invoke(query)
    except Exception as e:
        print(f"Failed to search vector store: {e}")
        results = []
    
    if not results:
        return "No relevant information found."
    
    # Get the most relevant result
    relevant_content = results[0].page_content
    print(f"🔍 Retrieved content: {relevant_content[:200]}...")
    
    # Use LLM to process the retrieved content and answer the query
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.4,
        max_tokens=150
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer the user's question in a natural conversational way based on the provided context."),
        HumanMessage(content=f"Context: {relevant_content}\n\nQuestion: {query}\n\nAnswer:")
    ]
    
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    from src.utils.pdf_parser import load_pdf_text
    from src.utils.qdrant_utils import initialize_with_documents as generate_embeddings
    vectordb = generate_embeddings(load_pdf_text())
    print(query_rag(vectordb, "key concepts"))
