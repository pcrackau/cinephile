import os
import sys
sys.path.append(os.getcwd())

from ultralytics import YOLO

from pathlib import Path

# TODO yolo only detects "person", expand with fairface or deepface
def detect_objects(image, model):
    results = model(image)
    objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]
            confidence = float(box.conf[0])
            print(f"Detected: {label} ({confidence:.2%})")
            print(result.boxes)
            bbox = [float(x) for x in box.xyxy[0].tolist()]
            objects.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox
            })
    
    return objects


# finetune

import torch
from transformers import (
BlipProcessor, BlipForConditionalGeneration,
CLIPProcessor, CLIPModel,
AutoProcessor, LlavaForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Dict
import json
from pathlib import Path

class LocalMovieAnalyzer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
            # Load models
        self.load_models()
        
    def load_models(self):
        """Load all required models locally"""
        print("Loading models...")
        
        # 1. CLIP alternative - SentenceTransformer with vision
        # This is smaller and faster than OpenAI CLIP
        try:
            self.clip_model = SentenceTransformer('clip-ViT-B-32')
            print("✓ Loaded CLIP alternative")
        except:
            # Fallback to HuggingFace CLIP
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✓ Loaded HuggingFace CLIP")
        
        # 2. BLIP2 for detailed captions
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.to(self.device)
        print("✓ Loaded BLIP2")
        
        # 3. Text embeddings for RAG
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Loaded text embedder")
        
        # Optional: LLaVA for more detailed analysis (larger model)
        # Uncomment if you have enough VRAM (>8GB)
        """
        self.llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            torch_dtype=torch.float16
        ).to(self.device)
        print("✓ Loaded LLaVA")
        """

    def analyze_keyframe(self, image_path: str) -> Dict:
        """Analyze a single keyframe"""
        image = Image.open(image_path).convert("RGB")
        
        results = {
            "image_path": image_path,
            "basic_caption": self.generate_basic_caption(image),
            "detailed_caption": self.generate_detailed_caption(image),
            "scene_embedding": self.get_scene_embedding(image),
            "timestamp": None  # You can add this from your keyframe detector
        }
        
        return results

    def generate_basic_caption(self, image: Image) -> str:
        """Generate basic caption with BLIP"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def generate_detailed_caption(self, image: Image) -> str:
        """Generate detailed caption with conditional generation"""
        # Use BLIP with specific prompts for movie analysis
        prompts = [
            "a detailed description of",
            "in this scene there is",
            "the setting shows",
            "people in this image are"
        ]
        
        captions = []
        for prompt in prompts:
            inputs = self.blip_processor(image, prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(
                    **inputs, 
                    max_length=100,
                    num_beams=5,
                    temperature=0.7
                )
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            # Remove the prompt from the beginning
            caption = caption.replace(prompt, "").strip()
            if caption and len(caption) > 10:
                captions.append(caption)
        
        return " | ".join(captions[:2])  # Combine best captions

    def get_scene_embedding(self, image: Image) -> np.ndarray:
        """Get scene embedding for similarity search"""
        if hasattr(self, 'clip_processor'):
            # Using HuggingFace CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.cpu().numpy().flatten()
        else:
            # Using SentenceTransformer CLIP
            return self.clip_model.encode(image)

    def analyze_movie_batch(self, keyframes_dir: str, movie_name: str) -> Dict:
        """Analyze all keyframes from a movie"""
        keyframe_paths = list(Path(keyframes_dir).glob("*.jpg")) + \
                        list(Path(keyframes_dir).glob("*.png"))
        
        print(f"Analyzing {len(keyframe_paths)} keyframes for {movie_name}...")
        
        all_results = []
        embeddings = []
        
        for i, frame_path in enumerate(keyframe_paths):
            try:
                result = self.analyze_keyframe(str(frame_path))
                all_results.append(result)
                embeddings.append(result["scene_embedding"])
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(keyframe_paths)} frames")
                    
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                continue
        
        # Create text embeddings for all descriptions
        all_text = []
        for result in all_results:
            combined_text = f"{result['basic_caption']} {result['detailed_caption']}"
            all_text.append(combined_text)
        
        text_embeddings = self.text_embedder.encode(all_text)
        
        # Add text embeddings to results
        for i, result in enumerate(all_results):
            result["text_embedding"] = text_embeddings[i]
        
        movie_analysis = {
            "movie_name": movie_name,
            "total_keyframes": len(all_results),
            "keyframe_analyses": all_results,
            "scene_embeddings": np.array(embeddings),
            "summary": self.generate_movie_summary(all_results)
        }
        
        return movie_analysis

    def generate_movie_summary(self, analyses: List[Dict]) -> str:
        """Generate overall movie summary from keyframe analyses"""
        all_captions = [a["basic_caption"] + " " + a["detailed_caption"] 
                    for a in analyses]
        
        # Simple extractive summary - you could use a proper summarization model here
        unique_scenes = list(set([cap.split('.')[0] for cap in all_captions if cap]))
        
        return f"Movie contains {len(analyses)} scenes including: " + \
            "; ".join(unique_scenes[:10]) + "..."

    def save_analysis(self, analysis: Dict, output_path: str):
        """Save analysis to JSON (embeddings as lists for JSON compatibility)"""
        # Convert numpy arrays to lists for JSON serialization
        json_analysis = analysis.copy()
        json_analysis["scene_embeddings"] = analysis["scene_embeddings"].tolist()
        
        for keyframe in json_analysis["keyframe_analyses"]:
            keyframe["scene_embedding"] = keyframe["scene_embedding"].tolist()
            keyframe["text_embedding"] = keyframe["text_embedding"].tolist()
        
        with open(output_path, 'w') as f:
            json.dump(json_analysis, f, indent=2)
        
        print(f"Analysis saved to {output_path}")

    def find_similar_scenes(self, query_embedding: np.ndarray, 
                        all_embeddings: np.ndarray, top_k: int = 5):
        """Find most similar scenes using cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices, similarities[top_indices]


# Usage example

def main():
    # Initialize analyzer
    analyzer = LocalMovieAnalyzer()


    # Analyze a single movie
    movie_name = "vintage_war_movie"
    keyframes_dir = "./keyframes/vintage_war_movie/"  # Your keyframes directory

    # Run analysis
    analysis = analyzer.analyze_movie_batch(keyframes_dir, movie_name)

    # Save results
    output_path = f"./analysis_{movie_name}.json"
    analyzer.save_analysis(analysis, output_path)

    # Example: Find similar scenes
    if len(analysis["keyframe_analyses"]) > 1:
        query_embedding = analysis["keyframe_analyses"][0]["scene_embedding"]
        all_embeddings = analysis["scene_embeddings"]
        
        similar_indices, similarities = analyzer.find_similar_scenes(
            query_embedding, all_embeddings, top_k=3
        )
        
        print("\nMost similar scenes:")
        for idx, sim in zip(similar_indices, similarities):
            print(f"Frame {idx}: {analysis['keyframe_analyses'][idx]['basic_caption']} "
                f"(similarity: {sim:.3f})")


if __name__ == "__main__":
    main()