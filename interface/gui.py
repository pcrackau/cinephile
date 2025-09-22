
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
#from tkinter.dnd import DND_FILES
import tkinterdnd2 as tkdnd
from PIL import Image, ImageTk
import numpy as np
import requests
import json
import os
from pathlib import Path
import threading
from datetime import datetime
import hashlib
import pickle

class RAGChatGUI:
    def __init__(self, llm_model):
        self.root = tkdnd.Tk()
        self.root.title("Cinephile Search Engine")
        self.root.geometry("1200x800")

        # Konfiguration
        self.ollama_url = "http://localhost:11434/api/generate"
        if llm_model:
            self.model_name = str(llm_model)
        else:
            self.model_name = "llama2:latest"

        
        # Image embeddings cache
        self.image_embeddings = {}
        self.cache_file = "image_embeddings_cache.pkl"
        self.load_embeddings_cache()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left - Chat Interface
        left_frame = ttk.LabelFrame(main_frame, text="Chat", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(
            left_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED,
            height=25,
            font=("Arial", 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Chat input
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.chat_input = tk.Text(input_frame, height=3, wrap=tk.WORD)
        self.chat_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT)
        
        # Control-Return to send
        self.chat_input.bind("<Return>", lambda e: self.send_message())
        
        # Model selection
        model_frame = ttk.Frame(left_frame)
        model_frame.pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self.model_name)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_var, width=15)
        model_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Right side
        right_frame = ttk.LabelFrame(main_frame, text="Films similar to image", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Drag and Drop Area
        self.drop_area = tk.Label(
            right_frame,
            text="Drag and drop image\nor click to upload",
            bg="lightgray",
            relief=tk.SUNKEN,
            height=8,
            font=("Arial", 12)
        )
        self.drop_area.pack(fill=tk.X, pady=(0, 10))
        self.drop_area.bind("<Button-1>", self.select_images)
        
        # Drag and Drop konfigurieren
        self.drop_area.drop_target_register(tkdnd.DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.handle_drop)
        
        # Ähnlichkeits-Threshold
        threshold_frame = ttk.Frame(right_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Similarity-threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_scale = ttk.Scale(
            threshold_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.threshold_var,
            orient=tk.HORIZONTAL
        )
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.80")
        self.threshold_label.pack(side=tk.RIGHT)
        threshold_scale.configure(command=self.update_threshold_label)
        
        # Bildvorschau und Ergebnisse
        self.image_frame = ttk.Frame(right_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Ergebnisse anzeigen
        self.results_text = scrolledtext.ScrolledText(
            self.image_frame,
            height=15,
            wrap=tk.WORD,
            font=("Arial", 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Chat initialisieren
        self.add_chat_message("System", "RAG-pipeline ready!")
        
    def load_embeddings_cache(self):
        """Load image-embeddings"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.image_embeddings = pickle.load(f)
                print(f"Loaded: {len(self.image_embeddings)} Image-embeddings from cache.")
            except Exception as e:
                print(f"Error loading from cache: {e}")
                self.image_embeddings = {}

    def save_embeddings_cache(self):
        """Save image-embeddings"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.image_embeddings, f)
        except Exception as e:
            print(f"Error saving of cache: {e}")

    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"{float(value):.2f}")

    def add_chat_message(self, sender, message):
        """Add message"""
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Formatierung je nach Sender
        if sender == "You":
            self.chat_display.insert(tk.END, f"[{timestamp}] You: ", "user")
        elif sender == "System":
            self.chat_display.insert(tk.END, f"[{timestamp}] System: ", "system")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] Bot: ", "bot")
        
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        # Styling
        self.chat_display.tag_config("user", foreground="blue")
        self.chat_display.tag_config("bot", foreground="green")
        self.chat_display.tag_config("system", foreground="orange")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def send_message(self):
        """Send text to ollama"""
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Show message in chat
        self.add_chat_message("You", message)
        self.chat_input.delete("1.0", tk.END)
        
        # Status
        self.status_var.set("Generate response...")
        
        # Query in separate thread
        thread = threading.Thread(target=self.get_ollama_response, args=(message,))
        thread.daemon = True
        thread.start()

    def get_ollama_response(self, message):
        """Retrieve ollama response"""
        try:
            payload = {
                "model": self.model_var.get(),
                "prompt": message,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result.get("response", "No answer")
                
                # Antwort im Chat anzeigen
                self.root.after(0, lambda: self.add_chat_message("Bot", bot_response))
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                self.root.after(0, lambda: self.add_chat_message("System", error_msg))
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            self.root.after(0, lambda: self.add_chat_message("System", error_msg))
        
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))

    def select_images(self, event=None):
        """Select images"""
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if files:
            self.process_images(files)

    def handle_drop(self, event):
        """Handle Drag and Drop"""
        files = self.root.tk.splitlist(event.data)
        image_files = []
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                image_files.append(file)
        
        if image_files:
            self.process_images(image_files)
        else:
            messagebox.showwarning("Warning", "No valid images found!")

    def process_images(self, image_paths):
        """Process images"""
        self.status_var.set("Processing images...")
        
        thread = threading.Thread(target=self.analyze_images, args=(image_paths,))
        thread.daemon = True
        thread.start()

    def get_image_hash(self, image_path):
        """Create hash for image (for caching)"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_image_embedding(self, image_path):
        """Create simple embedding"""
        try:
            # Check cache first
            img_hash = self.get_image_hash(image_path)
            if img_hash in self.image_embeddings:
                return self.image_embeddings[img_hash]
            
            # Lade und verarbeite Bild
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((256, 256))  # Standardgröße
            
            # Erstelle einfaches Farbhistogramm als Embedding
            img_array = np.array(img)
            
            # RGB Histogramme
            hist_r = np.histogram(img_array[:,:,0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=32, range=(0, 256))[0]
            
            # Kombiniere zu einem Embedding
            embedding = np.concatenate([hist_r, hist_g, hist_b])
            embedding = embedding / np.linalg.norm(embedding)  # Normalisiere
            
            # Cache speichern
            self.image_embeddings[img_hash] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def calculate_similarity(self, emb1, emb2):
        """Cosine similarity between two images"""
        return np.dot(emb1, emb2)

    def analyze_images(self, image_paths):
        """Find similarities"""
        try:
            # Erstelle Embeddings für alle Bilder
            embeddings = {}
            valid_images = []
            
            for img_path in image_paths:
                embedding = self.get_image_embedding(img_path)
                if embedding is not None:
                    embeddings[img_path] = embedding
                    valid_images.append(img_path)
            
            if len(valid_images) < 2:
                self.root.after(0, lambda: self.show_results("At least two images required."))
                return
            
            # Berechne Ähnlichkeiten
            similarities = []
            threshold = self.threshold_var.get()
            
            for i, img1 in enumerate(valid_images):
                for j, img2 in enumerate(valid_images[i+1:], i+1):
                    similarity = self.calculate_similarity(embeddings[img1], embeddings[img2])
                    
                    if similarity >= threshold:
                        similarities.append({
                            'img1': img1,
                            'img2': img2,
                            'similarity': similarity
                        })
            
            # Sortiere nach Ähnlichkeit
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Zeige Ergebnisse
            self.root.after(0, lambda: self.show_similarity_results(similarities, len(valid_images)))
            
            # Speichere Cache
            self.save_embeddings_cache()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, lambda: self.show_results(error_msg))
        
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))

    def show_similarity_results(self, similarities, total_images):
        """Show results"""
        self.results_text.delete(1.0, tk.END)
        
        result_text = f"Finished!\n"
        result_text += f"Analysed images: {total_images}\n"
        result_text += f"Threshold: {self.threshold_var.get():.2f}\n"
        result_text += f"Similarity pairs: {len(similarities)}\n\n"
        
        if similarities:
            result_text += "Similar pairs:\n"
            result_text += "-" * 50 + "\n"
            
            for i, sim in enumerate(similarities[:10]):  # Zeige top 10
                img1_name = os.path.basename(sim['img1'])
                img2_name = os.path.basename(sim['img2'])
                result_text += f"{i+1}. {img1_name} ↔ {img2_name}\n"
                result_text += f"   Similarity: {sim['similarity']:.3f}\n\n"
            
            if len(similarities) > 10:
                result_text += f"... and {len(similarities) - 10} additional pairs\n"
        else:
            result_text += "No images found.\n"
            result_text += "Try a lower threshold.\n"
        
        self.results_text.insert(1.0, result_text)

    def show_results(self, message):
        """Show results"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, message)

    def run(self):
        """Start GUI"""
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = RAGChatGUI("llama3.2:latest")
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure tkinterdnd2 and PIL are installed:")
        print("pip install tkinterdnd2 pillow numpy requests")