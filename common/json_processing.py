from pathlib import Path
import json
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

import time
import requests

PATH_FILMS = "./datasets"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

def load_and_embed_jsons(path):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = JSONLoader(file_path="datasets/DFF/Advance to the Piave.json", jq_schema=".films[]", text_content=False)
    documents = loader.load()

    db = Chroma.from_documents(documents, embedding_function)
    #query = "What year did albert einstein win the nobel prize?"
    #docs = db.similarity_search(query)
    #print(docs[0].page_content)
    return db


def load_jsons(path):
    data = []
    for file in Path(path).rglob("*.json"):
        with open(file, "r") as f:
            film = json.load(f)
            data.append(film)
    return data

    # TODO replace in load_or_create_chroma
    """for item in items:
    if isinstance(item, dict):
        # Extract relevant fields only
        title = item.get("title", "")
        description = item.get("description", "")
        year = item.get("year", "")
        country = item.get("country", "")
        
        # Customize formatting according to data used
        content = f"{title} ({year}, {country})\n\n{description}"
        
        doc = Document(
            page_content=content.strip(),
            metadata={"source": str(file)}
        )
        documents.append(doc)
    """

def load_or_create_chroma(path, persist_dir="chroma_db"):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if Path(persist_dir).exists():
        print("Loading existing Chroma DB...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
    
    print("Building new Chroma DB from JSONs...")
    documents = []

    for file in Path(path).rglob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = json.load(f)
                
                # Handle both list and dict roots
                if isinstance(content, list):
                    items = content
                elif isinstance(content, dict):
                    items = content.values()
                else:
                    continue  # Skip invalid structures

                for item in items:
                    if isinstance(item, dict):
                        doc = Document(
                            page_content=json.dumps(item, ensure_ascii=False),
                            metadata={"source": str(file)}
                        )
                        documents.append(doc)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if not documents:
        raise ValueError("No valid documents were found for embedding.")

    db = Chroma.from_documents(documents, embedding_function, persist_directory=persist_dir)
    db.persist()
    
    return db

def vector_embedding_json(path, embedding_function):
    pass


def query_ollama(prompt, model=MODEL_NAME):     # check which models available in command line: ollama list
    start_time = time.time()
    url = OLLAMA_URL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        #"options": {"num_predict": 128}
    }

    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    result = response.json()
    if "response" not in result:
        raise KeyError(f"Missing 'response' in: {result}")

    duration = time.time() - start_time
    print(f"Query time {duration:.5f}s")

    return result["response"]


def ask_with_context(question, db, model=MODEL_NAME, context_limit=3):
    # 1. Retrieve relevant documents from Chroma
    results = db.similarity_search(question, k=context_limit)
    
    # 2. Concatenate context into a single string
    context = "\n\n".join(doc.page_content for doc in results)

    # 3. Compose a prompt that includes the context
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question:
{question}
"""

    # 4. Query Ollama with the composed prompt
    answer = query_ollama(prompt, model=model)
    return answer


#film_jsons = load_and_embed_jsons(PATH_FILMS)
film_jsons = load_jsons(PATH_FILMS)
print(f"Loaded {len(film_jsons)} films.")
print(film_jsons[0])

db = load_or_create_chroma(PATH_FILMS)

# Example query
query = "What year did the Austro-Italian battle take place?"

# Retrieve top 4 most relevant documents
docs = db.similarity_search(query, k=4)

# Print the results
for i, doc in enumerate(docs, 1):
    print(f"\n--- Result {i} ---")
    print("Score (approx): not available in basic API")
    print("Source:", doc.metadata.get("source"))
    print("Content:", doc.page_content)


response = ask_with_context("Which films are from the first world war?", db)
print("🧠 Answer:", response)