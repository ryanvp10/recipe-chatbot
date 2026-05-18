#!/usr/bin/env python3
"""
Import pre-embedded ChromaDB data from Colab export JSON into a local ChromaDB instance.

Usage:
    python3 import_chroma_json.py <path_to_chroma_export.json> [chroma_data_dir]

Example:
    python3 import_chroma_json.py chroma_export.json /home/ubuntu/recipe-chatbot/backend/chroma_data
"""

import json
import sys
import os
import time
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 import_chroma_json.py <export.json> [chroma_data_dir]")
        sys.exit(1)

    json_path = sys.argv[1]
    chroma_dir = sys.argv[2] if len(sys.argv) > 2 else "./chroma_data_imported"

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print(f"Loading export from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    embeddings = data["embeddings"]

    total = len(ids)
    print(f"Loaded {total} records.")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")

    # Import chromadb
    import chromadb
    from chromadb.config import Settings

    print(f"\nConnecting to ChromaDB at: {chroma_dir}")
    client = chromadb.PersistentClient(path=chroma_dir)

    COLLECTION_NAME = "recipes"

    # Delete existing
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created collection: {COLLECTION_NAME}")

    # Import in batches
    BATCH_SIZE = 500
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]
        batch_emb = embeddings[i:i+BATCH_SIZE]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=batch_emb,
        )

        imported = min(i + BATCH_SIZE, total)
        pct = imported / total * 100
        elapsed = time.time() - start
        rate = imported / elapsed if elapsed > 0 else 0
        eta = (total - imported) / rate if rate > 0 else 0
        print(f"  {imported}/{total} ({pct:.0f}%) — {rate:.0f} rec/s — ETA: {eta:.0f}s")

    elapsed = time.time() - start
    count = collection.count()

    print(f"\n✅ Import complete!")
    print(f"   Total: {count} recipes")
    print(f"   Time: {elapsed:.1f}s")
    print(f"   ChromaDB data: {chroma_dir}")
    print(f"\nYour backend can now use this ChromaDB data.")

if __name__ == "__main__":
    main()
