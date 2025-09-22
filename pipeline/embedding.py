from pathlib import Path
import json
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


def load_chroma(persist_dir="chroma_db"):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)


def query_chroma(db, query: str, k: int = 5):
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def _load_film_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)


def generate_vectorization_text(film_metadata, keyframe_data):
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


def generate_docs(metadata_path, keyframes_dir: Path):
    """
    Given a film's metadata and the keyframes directory for one cut,
    load all keyframe JSONs and return LangChain Documents.
    """
    film_metadata = _load_film_metadata(metadata_path)
    docs = []

    for keyframe_json in sorted(Path(keyframes_dir).rglob("*.json")):
        with open(keyframe_json, 'r') as f:
            keyframe_data = json.load(f)

        text = generate_vectorization_text(film_metadata, keyframe_data)

        doc = Document(
            page_content=text,
            metadata={
                "film_title": film_metadata.get('title', 'Unknown'),
                "year": film_metadata.get('year', 'Unknown'),
                "country": film_metadata.get('country', 'Unknown'),
                "description": film_metadata.get('dcDescription', {}),
                "cut_id": keyframe_json.parent.parent.name,
                "keyframe": keyframe_json.name,
            }
        )
        docs.append(doc)

    return docs


def load_or_create_chroma(path="processing", persist_dir="chroma_db"):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if Path(persist_dir).exists():
        print("Loading existing Chroma DB...")
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
    
    print("Building new Chroma DB from JSONs...")
    documents = []

    for film_dir in Path(path).iterdir():
        if not film_dir.is_dir():
            continue

        metadata_path = Path("metadata") / f"{film_dir.name}.json"
        if not metadata_path.exists():
            print(f"Skipping {film_dir} — missing metadata.")
            continue

        # go through all cut keyframe dirs
        for cut_dir in (film_dir / "key_frames").iterdir():
            keyframes_dir = cut_dir / "keyframes"
            if not keyframes_dir.exists():
                continue

            docs = generate_docs(metadata_path, keyframes_dir)
            documents.extend(docs)

    if not documents:
        raise ValueError("No valid documents found to embed.")

    db = Chroma.from_documents(documents, embedding_function, persist_directory=persist_dir)
    return db
