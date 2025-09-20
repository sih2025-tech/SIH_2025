# qa_system.py
# Load prebuilt FAISS index + metadata, then run queries with Gemma

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import os

# -------------------------------
# Load FAISS + metadata
# -------------------------------
def load_index(index_path="agro_index.faiss", meta_path="agro_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    chunks = meta["chunks"]
    embed_model = SentenceTransformer(meta["model_name"])
    return index, chunks, embed_model

# -------------------------------
# Load Gemma model
# -------------------------------
model_name = "google/gemma-3-12b"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

# -------------------------------
# Query function
# -------------------------------
def query_knowledge_base(question, index, embed_model, chunks, image_path=None, top_k=3):
    q_vec = embed_model.encode([question]).astype("float32")
    D, I = index.search(q_vec, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are an agricultural assistant.
Use the following context to answer clearly and concisely.
If not enough info is present, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""

    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
    answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# -------------------------------
# Example query
# -------------------------------
if __name__ == "__main__":
    index, chunks, embed_model = load_index("agro_index.faiss", "agro_chunks.pkl")

    question = "What are the best practices for organic paddy farming?"
    answer = query_knowledge_base(question, index, embed_model, chunks)
    print("\n❓ Question:", question)
    print("\n✅ Answer:", answer)
