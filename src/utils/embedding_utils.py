from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

def generate_embeddings(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name="pdf_data"
    )

    return vectordb

if __name__ == "__main__":
    from pdf_parser import load_pdf_text
    texts = load_pdf_text()
    vectordb = generate_embeddings(texts)
    print(vectordb.similarity_search("summary", k=1))
