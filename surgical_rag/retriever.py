"""Custom DSPy Retriever module for Surgical RAG."""

from typing import List, Optional

import dspy

from surgical_rag.storage import LinkedListStore


class SurgicalRetriever(dspy.Retrieve):
    """
    Custom DSPy Retriever that searches for Top-K anchors and 
    auto-fetches neighbors via linked list ID lookup.
    """
    
    def __init__(
        self,
        store: LinkedListStore,
        k: int = 3,
        fetch_neighbors: bool = True,
        neighbor_depth: int = 1
    ):
        """
        Initialize the retriever.
        
        Args:
            store: The LinkedListStore instance to query.
            k: Number of top anchor chunks to retrieve.
            fetch_neighbors: Whether to auto-fetch neighboring chunks.
            neighbor_depth: How many neighbors to fetch on each side.
        """
        super().__init__(k=k)
        self.store = store
        self.fetch_neighbors = fetch_neighbors
        self.neighbor_depth = neighbor_depth
    
    def _fetch_neighbor_chain(self, chunk_id: str, direction: str, depth: int) -> List[dict]:
        """
        Fetch a chain of neighbors in one direction.
        
        Args:
            chunk_id: Starting chunk ID.
            direction: "prev" or "next".
            depth: How many neighbors to fetch.
            
        Returns:
            List of neighbor chunks.
        """
        neighbors = []
        current_id = chunk_id
        
        for _ in range(depth):
            chunk = self.store.get_chunk_by_id(current_id)
            if not chunk:
                break
            
            next_key = f"{direction}_id"
            next_id = chunk["metadata"].get(next_key, "")
            
            if not next_id:
                break
            
            neighbor = self.store.get_chunk_by_id(next_id)
            if neighbor:
                neighbors.append(neighbor)
                current_id = next_id
            else:
                break
        
        return neighbors
    
    def forward(self, query: str, k: Optional[int] = None) -> dspy.Prediction:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query.
            k: Optional override for number of results.
            
        Returns:
            dspy.Prediction with 'passages' containing retrieved text.
        """
        sorted_chunks = self._retrieve_chunks_with_neighbors(query, k)
        
        if not sorted_chunks:
            return dspy.Prediction(passages=[])
        
        # Extract passages
        passages = [chunk["text"] for chunk in sorted_chunks]
        
        return dspy.Prediction(passages=passages)
    
    def _retrieve_chunks_with_neighbors(self, query: str, k: Optional[int] = None) -> List[dict]:
        """
        Internal method to retrieve chunks and their neighbors.
        
        Args:
            query: The search query.
            k: Optional override for number of results.
            
        Returns:
            Sorted list of chunk dicts.
        """
        k = k or self.k
        
        # Search for top-K anchor chunks
        anchors = self.store.query(query_text=query, n_results=k)
        
        if not anchors:
            return []
        
        # Collect all chunks (anchors + neighbors)
        all_chunks = {}  # Use dict to deduplicate by ID
        
        for anchor in anchors:
            all_chunks[anchor["id"]] = anchor
            
            if self.fetch_neighbors:
                # Fetch previous neighbors
                prev_neighbors = self._fetch_neighbor_chain(
                    anchor["id"], "prev", self.neighbor_depth
                )
                for neighbor in prev_neighbors:
                    all_chunks[neighbor["id"]] = neighbor
                
                # Fetch next neighbors
                next_neighbors = self._fetch_neighbor_chain(
                    anchor["id"], "next", self.neighbor_depth
                )
                for neighbor in next_neighbors:
                    all_chunks[neighbor["id"]] = neighbor
        
        # Sort by index to maintain document order
        return sorted(
            all_chunks.values(),
            key=lambda x: x["metadata"].get("index", 0)
        )
    
    def retrieve_with_context(self, query: str, k: Optional[int] = None) -> List[dict]:
        """
        Retrieve chunks with full metadata for fusion.
        
        Args:
            query: The search query.
            k: Optional override for number of results.
            
        Returns:
            List of chunk dicts with metadata.
        """
        return self._retrieve_chunks_with_neighbors(query, k)
