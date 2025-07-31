# utils/vector_store.py

import faiss
import numpy as np
import json
import os
import uuid
from typing import List, Dict, Tuple, Any, Optional

class VectorStore:
    """
    Manages a FAISS-based vector store for efficient similarity search of text embeddings.
    It stores both the embeddings and their corresponding original text chunks with metadata.
    Supports adding chunks, performing searches, and local persistence.
    """
    def __init__(self, embedding_dim: int, embedding_model: Any):
        """
        Initializes the VectorStore.

        Args:
            embedding_dim (int): The dimension of the embeddings (e.g., 768 for text-embedding-004).
            embedding_model: An instance of an embedding model (e.g., utils.embeddings.EmbeddingModel)
                             with `embed_documents` and `embed_query` methods.
        """
        self.embedding_dim = embedding_dim
        # Use IndexFlatL2 for Euclidean distance, suitable for many embedding models
        self.index: faiss.Index = faiss.IndexFlatL2(embedding_dim)
        # Stores original chunk data: List[{"id": str, "content": str, "metadata": dict}]
        self.chunks: List[Dict[str, Any]] = [] 
        self.embedding_model = embedding_model
        print(f"[VectorStore] Initialized with embedding dimension: {embedding_dim}")

    def add_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[str]:
        """
        Adds a list of structured chunk data to the vector store.
        Each chunk dictionary is expected to have at least 'content' and 'id',
        and optionally 'metadata'.

        Args:
            chunks_data (List[Dict[str, Any]]): A list of dictionaries, where each dict represents
                                                 a chunk (e.g., {"id": "...", "content": "...", "metadata": {...}}).

        Returns:
            List[str]: A list of IDs for the chunks that were successfully added to the FAISS index.
        """
        if not chunks_data:
            print("[VectorStore] No chunks provided to add.")
            return []

        # Extract content for embedding, filtering out empty content
        valid_chunks = []
        contents_to_embed = []
        
        for chunk in chunks_data:
            content = chunk.get("content", "").strip()
            if content:  # Only process chunks with non-empty content
                valid_chunks.append(chunk)
                contents_to_embed.append(content)
            else:
                print(f"[VectorStore] Skipping chunk with empty content: {chunk.get('id', 'N/A')}")

        if not contents_to_embed:
            print("[VectorStore] No valid chunks with content to embed.")
            return []
        
        # Generate embeddings using the provided embedding model
        embeddings = self.embedding_model.embed_documents(contents_to_embed)

        if not embeddings or len(embeddings) != len(contents_to_embed):
            print(f"[VectorStore] Failed to generate embeddings for all {len(contents_to_embed)} chunks. Only {len(embeddings)} generated.")
            return []

        # Convert to numpy array and ensure correct data type
        try:
            embeddings_np = np.array(embeddings, dtype='float32')
            if embeddings_np.shape[1] != self.embedding_dim:
                print(f"[VectorStore] Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings_np.shape[1]}")
                return []
        except Exception as e:
            print(f"[VectorStore] Error converting embeddings to numpy array: {e}")
            return []

        added_chunk_ids = []
        chunks_to_add_to_store = []

        # Process valid chunks
        for i, chunk in enumerate(valid_chunks):
            # Ensure chunk has an ID, generate one if missing
            if "id" not in chunk or chunk["id"] is None:
                chunk["id"] = str(uuid.uuid4())
                print(f"[VectorStore] Generated ID for chunk: {chunk['id']}")
            
            # Ensure metadata exists
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            
            chunks_to_add_to_store.append(chunk)
            added_chunk_ids.append(chunk["id"])

        # Add the embeddings to the FAISS index
        self.index.add(embeddings_np)
        # Store the associated chunk data
        self.chunks.extend(chunks_to_add_to_store)

        print(f"[VectorStore] Successfully added {len(added_chunk_ids)} embeddings and chunks.")
        return added_chunk_ids

    def similarity_search_with_score(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a similarity search using a query embedding and returns the top k
        most similar chunks along with their original data (content, ID, metadata)
        and the similarity score (distance). Lower score indicates higher similarity for L2 distance.

        Args:
            query_embedding (List[float]): The embedding vector of the query.
            k (int): The number of top similar chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
                                  - 'id': The ID of the retrieved chunk.
                                  - 'content': The text content of the retrieved chunk.
                                  - 'metadata': The metadata associated with the chunk.
                                  - 'score': The L2 distance (similarity score) of the chunk to the query.
                                            Lower score means higher similarity.
        """
        if not query_embedding:
            print("[VectorStore] Empty query embedding provided for search.")
            return []

        if self.index.ntotal == 0:
            print("[VectorStore] Vector store is empty. No search performed.")
            return []
        
        try:
            # Ensure the query embedding is a 2D numpy array with correct shape and dtype
            query_embedding_np = np.asarray(query_embedding, dtype='float32').reshape(1, -1)
            
            if query_embedding_np.shape[1] != self.embedding_dim:
                print(f"[VectorStore] Query embedding dimension mismatch: expected {self.embedding_dim}, got {query_embedding_np.shape[1]}")
                return []

            # Ensure k does not exceed the number of vectors in the index
            k = min(k, self.index.ntotal)
            if k == 0:
                return []

            distances, indices = self.index.search(query_embedding_np, k)
            
        except Exception as e:
            print(f"[VectorStore] Error during FAISS search: {e}")
            return []

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for indices that couldn't be filled
                continue
            
            if idx >= len(self.chunks):
                print(f"[VectorStore] Index {idx} out of range for chunks list of length {len(self.chunks)}")
                continue
            
            # Retrieve the full original chunk data using the index
            chunk_data = self.chunks[idx]
            
            # Extract the score (distance) and convert to standard float
            score = float(distances[0][i])
            
            # Create the result dictionary including all original chunk data and the score
            result_chunk = {
                "id": chunk_data.get("id", f"chunk_{idx}"),
                "content": chunk_data.get("content", ""),
                "metadata": chunk_data.get("metadata", {}),
                "score": score
            }
            results.append(result_chunk)
            
        print(f"[VectorStore] Found {len(results)} similar chunks for query.")
        return results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Returns all stored original chunk data.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a stored chunk.
        """
        return self.chunks

    def clear(self):
        """
        Clears all data (FAISS index and stored chunks) from the vector store.
        """
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunks = []
        print("[VectorStore] Cleared all data.")

    def save_local(self, path: str):
        """
        Saves the FAISS index and the associated chunk data to a local directory.

        Args:
            path (str): The directory path where the data should be saved.
        """
        os.makedirs(path, exist_ok=True)
        index_file = os.path.join(path, "faiss_index.bin")
        chunk_data_file = os.path.join(path, "chunk_data.json")

        try:
            faiss.write_index(self.index, index_file)
            with open(chunk_data_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            print(f"[VectorStore] Saved FAISS index and chunk data to {path}.")
        except Exception as e:
            print(f"[VectorStore] Error saving VectorStore to {path}: {e}")
            raise

    @classmethod
    def load_local(cls, path: str, embedding_model: Any):
        """
        Loads the FAISS index and chunk data from a local directory to recreate a VectorStore instance.

        Args:
            path (str): The directory path from which the data should be loaded.
            embedding_model: An instance of an embedding model (e.g., utils.embeddings.EmbeddingModel)
                             needed for the VectorStore's operations.

        Returns:
            VectorStore: A new VectorStore instance populated with the loaded data.

        Raises:
            FileNotFoundError: If the specified directory or required files are not found.
            Exception: For other errors during loading (e.g., file corruption).
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found for loading VectorStore: {path}")

        index_path = os.path.join(path, "faiss_index.bin")
        chunk_data_path = os.path.join(path, "chunk_data.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not os.path.exists(chunk_data_path):
            raise FileNotFoundError(f"Chunk data file not found: {chunk_data_path}")

        try:
            index = faiss.read_index(index_path)
            with open(chunk_data_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            # Reconstruct VectorStore instance
            # index.d holds the embedding dimension
            instance = cls(index.d, embedding_model) 
            instance.index = index
            instance.chunks = chunks
            print(f"[VectorStore] Loaded FAISS index and {len(chunks)} chunks from {path}.")
            return instance
        except Exception as e:
            print(f"[VectorStore] Error loading VectorStore from {path}: {e}")
            raise


# Test code
if __name__ == "__main__":
    print("--- Testing utils/vector_store.py ---")
    import shutil
    import tempfile
    
    # Mock embedding model for testing
    class MockEmbeddingModel:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [np.random.rand(768).tolist() for _ in texts]

        def embed_query(self, text: str) -> List[float]:
            return np.random.rand(768).tolist()

    TEST_DIR = tempfile.mkdtemp()
    print(f"Using temporary directory: {TEST_DIR}")
    
    try:
        embedding_model = MockEmbeddingModel()
        vector_store = VectorStore(embedding_dim=768, embedding_model=embedding_model)
        
        # Test adding chunks
        sample_chunks = [
            {"id": "c1", "content": "The cat sat on the mat.", "metadata": {"source": "story.txt"}},
            {"id": "c2", "content": "Dogs are loyal animals.", "metadata": {"source": "wiki.md"}},
        ]
        
        added_ids = vector_store.add_chunks(sample_chunks)
        assert len(added_ids) == 2
        print("✓ Chunk addition test passed")
        
        # Test search
        query_embedding = embedding_model.embed_query("cat")
        results = vector_store.similarity_search_with_score(query_embedding, k=1)
        assert len(results) == 1
        assert "id" in results[0] and "content" in results[0]
        print("✓ Search test passed")
        
        # Test persistence
        vector_store.save_local(TEST_DIR)
        loaded_store = VectorStore.load_local(TEST_DIR, embedding_model)
        assert len(loaded_store.get_all_chunks()) == 2
        print("✓ Persistence test passed")
        
    finally:
        shutil.rmtree(TEST_DIR)
        print("✓ All tests passed!")