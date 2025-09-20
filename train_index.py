# train_index.py
# Build or update FAISS index with multiple CSV files

import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

INDEX_PATH = "agro_index.faiss"
META_PATH = "agro_chunks.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------
# Load CSVs from folder
# -------------------------------
def load_text_from_csv_folder(csv_folder, text_column=None):
    all_text = ""
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            csv_file = os.path.join(csv_folder, file)
            try:
                df = pd.read_csv(csv_file)

                # If text_column is given, use it, else join all columns
                if text_column and text_column in df.columns:
                    text_data = df[text_column].dropna().astype(str)
                else:
                    text_data = df.astype(str).agg(" ".join, axis=1)

                all_text += "\n".join(text_data) + "\n"
                print(f"📥 Loaded {len(text_data)} rows from {file}")

            except Exception as e:
                print(f"⚠️ Failed to read {csv_file}: {e}")
    return all_text

# -------------------------------
# Chunking
# -------------------------------
def make_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# -------------------------------
# Build or Update FAISS index
# -------------------------------
def build_or_update_index(new_chunks):
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    new_vectors = embed_model.encode(new_chunks).astype("float32")

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("📂 Loading existing index...")
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        chunks = meta["chunks"]
    else:
        print("🆕 Creating new index...")
        index = faiss.IndexFlatL2(new_vectors.shape[1])
        chunks = []

    # Add new data
    index.add(new_vectors)
    chunks.extend(new_chunks)

    # Save updated index + metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "model_name": EMBED_MODEL_NAME}, f)

    print(f"✅ Index updated with {len(new_chunks)} new chunks")
    print(f"📦 Total chunks stored: {len(chunks)}")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    csv_folder = "/Users/vineetyelleswarapu/AgroGPT/CSVs"  # path to your CSV folder
    text_column = None  # 👈 set this if you only want one column, e.g., "text"

    all_text = load_text_from_csv_folder(csv_folder, text_column=text_column)
    new_chunks = make_chunks(all_text)
    build_or_update_index(new_chunks)
