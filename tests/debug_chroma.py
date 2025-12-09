from pathlib import Path

import chromadb


def inspect():
    path = Path("./chroma_data").resolve()
    print(f"Inspecting Chroma at: {path}")

    client = chromadb.PersistentClient(path=str(path))
    print(f"Collections: {[c.name for c in client.list_collections()]}")

    try:
        col = client.get_collection("thought_nodes")
        print("\nCollection: thought_nodes")
        print(f"Count: {col.count()}")
        print(f"Metadata: {col.metadata}")

        if col.count() > 0:
            peek = col.get(limit=1, include=["embeddings"])
            if len(peek["embeddings"]) > 0:
                print(f"Actual Dimension: {len(peek['embeddings'][0])}")
            else:
                print("No embeddings found in peek")
    except Exception as e:
        print(f"Error getting collection: {e}")


if __name__ == "__main__":
    inspect()
