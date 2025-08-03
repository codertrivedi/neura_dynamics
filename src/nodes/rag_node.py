def query_rag(vectorstore, query):
    retriever = vectorstore.as_retriever()
    results = retriever.invoke(query)
    return results[0].page_content if results else "No relevant info found."


if __name__ == "__main__":
    from src.utils.pdf_parser import load_pdf_text
    from src.utils.embedding_utils import generate_embeddings
    vectordb = generate_embeddings(load_pdf_text())
    print(query_rag(vectordb, "key concepts"))
