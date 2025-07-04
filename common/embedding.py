from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def embed_and_store(documents: list[Document], persist_dir: str):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_dir
    )
    
    db.persist()
    return db