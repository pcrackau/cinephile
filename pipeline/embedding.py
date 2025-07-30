from pathlib import Path
import json
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


def load_chroma(persist_dir="chroma_db"):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)


def embed_and_store(documents: list[Document], persist_dir: str):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_dir
    )
    
    db.persist()
    return db


def _load_film_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)


def generate_vectorization_text(film_metadata, keyframe_data, language):
    parts = [
        f"Film Title: {film_metadata.get('title', 'Unknown')}",
        #f"Film Title: {film_metadata.get('dcTitleLangAware', {}).get(language, ['Unknown'])[0]}",
        f"Year: {film_metadata.get('year', 'Unknown')}",
        f"Country: {film_metadata.get('country', 'Unknown')}",
        f"Description {film_metadata.get('dcDescription', {})}",
        f"Keyframe: {keyframe_data.get('filename', 'N/A')}"
    ]

    objects = keyframe_data.get('objects', [])
    if objects:
        object_lines = [f" - {obj['label']} ({obj['confidence']:.2%})" for obj in objects]
        parts.append("Detected Objects:\n" + "\n".join(object_lines))
    
    shot_type = keyframe_data.get('shot_type')
    if shot_type:
        parts.append(f"Shot Type: {shot_type}")
    return "\n".join(parts)


def generate_docs(metadata_path, keyframes_dir):
    film_metadata = _load_film_metadata(metadata_path)
    docs = []

    for keyframe_json in sorted(Path(keyframes_dir).glob("*.json")):
        with open(keyframe_json, 'r') as f:
            keyframe_data = json.load(f)
        
        #keyframe_data.setdefault("keyframe_filename", keyframe_json.name)
        
        text = generate_vectorization_text(film_metadata, keyframe_data)
        doc = Document(
            page_content=text,
            metadata={
                "keyframe": keyframe_json.name,
                "Film Title": film_metadata.get('title', 'Unknown'),
                #f"Film Title: {film_metadata.get('dcTitleLangAware', {}).get(language, ['Unknown'])[0]}",
                "Year": film_metadata.get('year', 'Unknown'),
                "Country": film_metadata.get('country', 'Unknown'),
                "Description": film_metadata.get('dcDescription', {})
            }
        )
        docs.append(doc)
    
    return docs

# TODO refactor into load chroma, create chroma from path
def load_or_create_chroma(path, persist_dir="chroma_db"):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if Path(persist_dir).exists():
        print("Loading existing Chroma DB...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
    
    print("Building new Chroma DB from JSONs...")
    documents = []

    for film_dir in Path(path).iterdir():
        if not film_dir.is_dir():
            continue
        
        film_id = film_dir.name
        metadata_path = Path("metadata") / f"{film_id}.json"
        keyframe_jsons = film_dir / "keyframes"

        if not metadata_path.exists() or not keyframe_jsons.exists():
            print(f"Skipping {film_id} — missing metadata or keyframes.")
            continue

        film_metadata = _load_film_metadata(metadata_path)

        for json_file in keyframe_jsons.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    keyframe_data = json.load(f)
                
                text = generate_vectorization_text(film_metadata, keyframe_data)
                doc = Document(
                    page_content=text,
                    metadata={
                        "film_title": film_metadata.get('title', 'Unknown'),
                        "year": film_metadata.get('year', 'Unknown'),
                        "country": film_metadata.get('country', 'Unknown'),
                        "keyframe": keyframe_data.get('filename', 'N/A'),
                    }
                )
                documents.append(doc)

            except Exception as e:
                print(f"Skipping {json_file} due to error: {e}")

    if not documents:
        raise ValueError("No valid documents found to embed.")

    db = Chroma.from_documents(documents, embedding_function, persist_directory=persist_dir)
    db.persist()
    return db
