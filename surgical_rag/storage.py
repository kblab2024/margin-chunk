"""Linked List storage for chunks in ChromaDB with prev_id and next_id metadata."""

import uuid
from typing import List, Optional

import chromadb
from chromadb.config import Settings


class LinkedListStore:
    """
    Stores chunks in ChromaDB as a linked list with prev_id and next_id metadata.
    
    Each chunk is stored with:
    - A unique ID
    - The chunk text
    - Metadata containing prev_id and next_id for navigation
    """
    
    def __init__(
        self,
        collection_name: str = "chunks",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the storage.
        
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database. 
                               If None, uses in-memory storage.
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.collection_name = collection_name
    
    def _generate_id(self) -> str:
        """Generate a unique ID for a chunk."""
        return str(uuid.uuid4())
    
    def store_chunks(self, chunks: List[str], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        Store chunks as a linked list in ChromaDB.
        
        Args:
            chunks: List of text chunks to store.
            embeddings: Optional pre-computed embeddings. If None, ChromaDB will compute them.
            
        Returns:
            List of chunk IDs in order.
        """
        if not chunks:
            return []
        
        # Generate IDs for all chunks
        chunk_ids = [self._generate_id() for _ in chunks]
        
        # Prepare metadata with linked list pointers
        metadatas = []
        for i in range(len(chunks)):
            metadata = {
                "index": i,
                "prev_id": chunk_ids[i - 1] if i > 0 else "",
                "next_id": chunk_ids[i + 1] if i < len(chunks) - 1 else ""
            }
            metadatas.append(metadata)
        
        # Store in ChromaDB
        add_params = {
            "ids": chunk_ids,
            "documents": chunks,
            "metadatas": metadatas
        }
        if embeddings:
            add_params["embeddings"] = embeddings
        
        self.collection.add(**add_params)
        
        return chunk_ids
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        """
        Retrieve a chunk by its ID.
        
        Args:
            chunk_id: The ID of the chunk to retrieve.
            
        Returns:
            Dict with 'id', 'text', and 'metadata' keys, or None if not found.
        """
        if not chunk_id:
            return None
            
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        
        if not result["ids"]:
            return None
        
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0]
        }
    
    def get_neighbors(self, chunk_id: str, direction: str = "both") -> List[dict]:
        """
        Get neighboring chunks using linked list navigation.
        
        Args:
            chunk_id: The ID of the anchor chunk.
            direction: "prev", "next", or "both".
            
        Returns:
            List of neighboring chunk dicts.
        """
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk:
            return []
        
        neighbors = []
        
        if direction in ("prev", "both"):
            prev_id = chunk["metadata"].get("prev_id", "")
            if prev_id:
                prev_chunk = self.get_chunk_by_id(prev_id)
                if prev_chunk:
                    neighbors.append(prev_chunk)
        
        if direction in ("next", "both"):
            next_id = chunk["metadata"].get("next_id", "")
            if next_id:
                next_chunk = self.get_chunk_by_id(next_id)
                if next_chunk:
                    neighbors.append(next_chunk)
        
        return neighbors
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        query_embedding: Optional[List[float]] = None
    ) -> List[dict]:
        """
        Query for similar chunks.
        
        Args:
            query_text: The query text.
            n_results: Number of results to return.
            query_embedding: Optional pre-computed query embedding.
            
        Returns:
            List of matching chunk dicts with scores.
        """
        if query_embedding:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        else:
            result = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
        
        chunks = []
        for i in range(len(result["ids"][0])):
            chunks.append({
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i] if result["distances"] else None
            })
        
        return chunks
    
    def clear(self):
        """Clear all chunks from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
