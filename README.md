# cinephile


## Challenge
Work in progress for a RAG challenge


- Searching the metadata for information
- Detecting as many shot types as possible (e.g., long shot, medium shot, American shot, close-up, extreme close-up)
- Detecting shot transitions (cuts)
- Identifying objects, locations, people, actions, etc. in single shots
- Accepting an input image and retrieving all scenes with similar visual composition
- Also: Any additional search feature that you find appropriate or particularly relevant to the dataset
- And ideally: combining all the above into a chatbot that can process complex queries, such as (but of course not limited to):
- “Find a scene from 1940s Germany in which a woman is working with a machine in a medium shot.”
- “Find a film where a medium shot of a soldier cuts to a close-up of his face.”
- “Find all films made after 1945 that contain shots resembling the input image.”


## Pipeline
- parse JSONs
TODOs:
- RAG with Ollama and Langchain: https://www.datacamp.com/tutorial/llama-3-1-rag?dc_referrer=https%3A%2F%2Fduckduckgo.com%2F
- Cut detection using (try) screen detect: https://www.scenedetect.com/docs/latest/api.html
- Shot type detection (try) shot type classifier: https://github.com/rsomani95/shot-type-classifier/
- Implement tensorflow for Mac: https://developer.apple.com/metal/tensorflow-plugin/
- Object detection tensorflow: https://www.tensorflow.org/hub/tutorials/object_detection
- Reverse image search: https://lantern.dev/tutorials/python/image
- Process all information for Ollama to use:
  * process JSON with langchain: https://how.wtf/how-to-use-json-files-in-vector-stores-with-langchain.html
  
Other potential search: 
