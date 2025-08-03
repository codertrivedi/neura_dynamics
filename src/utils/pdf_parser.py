from llama_index.core import SimpleDirectoryReader

def load_pdf_text(path="data/"):
    reader = SimpleDirectoryReader(input_dir=path)
    docs = reader.load_data()
    return [doc.text for doc in docs]


if __name__ == "__main__":
    text = load_pdf_text()
    print(text[0][:300])  
