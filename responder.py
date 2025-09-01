import argparse
import json
import sys
import requests
from typing import List, Dict, Any, Optional
import time

from pipeline.embedding import load_chroma, query_chroma

class OllamaRAGResponder:
    def __init__(self, model_name: str, ollama_host: str = "http://localhost:11434", persist_dir="chroma_db"):
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip('/')
        self.session = requests.Session()
        self.db = load_chroma(persist_dir)


    def check_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama"""
        try:
            response = self.session.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                return self.model_name in available_models
            return False
        except requests.RequestException as e:
            print(f"Error checking model availability: {e}")
            return False

    def generate_response(self, query: str, context_docs: list = None, 
                        k: int = 5, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        # Retrieve context from Chroma
        context_docs = query_chroma(self.db, query, k=k)
        
        # Combine all context docs into one prompt
        if context_docs:
            context_text = "\n\n".join(context_docs)
            full_prompt = f"Based on the following information, answer the question:\n\n{context_text}\n\nQuestion: {query}\nAnswer:"
        else:
            full_prompt = f"No relevant context found. Answer the question based on your general knowledge:\n\nQuestion: {query}\nAnswer:"

        # Call your LLM (Ollama) with the full prompt
        return self._call_llm(full_prompt, temperature=temperature, max_tokens=max_tokens)

    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        try:
            response = self.session.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: HTTP {response.status_code} - {response.text}"
        except requests.RequestException as e:
            return f"Error communicating with Ollama: {e}"


    def chat_mode(self, db, k: int = 5, temperature: float = 0.7, max_tokens: int = 2048):
        """Interactive RAG chat mode with explicit handling of no matches"""
        print(f"\n🤖 RAG Responder using model: {self.model_name}")
        print("Type 'quit' or 'exit' to end the session")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                # 1. Retrieve docs from Chroma
                results = db.similarity_search(user_input, k=k)

                if not results:
                    print("🤖 No relevant films found in the database.")
                    continue

                # Flatten retrieved docs into context text
                context = []
                for doc in results:
                    try:
                        parsed = json.loads(doc.page_content)
                        if isinstance(parsed, dict):
                            flat = " ".join(
                                " ".join(v) if isinstance(v, list) else str(v)
                                for v in parsed.values()
                            )
                            context.append(flat)
                        else:
                            context.append(str(parsed))
                    except json.JSONDecodeError:
                        context.append(doc.page_content)

                # 2. Generate LLM response
                print("🔎 Retrieved context from Chroma...")
                response = self.generate_response(
                    user_input,
                    context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # 3. Print response
                print(f"\n🤖 {response}")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Responder for Ollama")
    parser.add_argument("-model", "-m", default="llama3:latest")
    parser.add_argument("-host", default="http://localhost:11434")
    parser.add_argument("-persist-dir", default="chroma_db")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-temperature", "-t", type=float, default=0.7)
    parser.add_argument("-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    from pipeline.embedding import load_chroma
    db = load_chroma(args.persist_dir)

    responder = OllamaRAGResponder(args.model, args.host)

    if not responder.check_model_availability():
        print(f"Model '{args.model}' not found locally. Run `ollama pull {args.model}` first.")
        return

    responder.chat_mode(db, k=args.k, temperature=args.temperature, max_tokens=args.max_tokens)



if __name__ == "__main__":
    main()