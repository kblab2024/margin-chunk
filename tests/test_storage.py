"""Tests for the LinkedListStore module."""

import pytest


def _generate_dummy_embeddings(n: int, dim: int = 384) -> list:
    """Generate dummy embeddings for testing."""
    import random
    return [[random.random() for _ in range(dim)] for _ in range(n)]


class TestLinkedListStore:
    """Tests for the LinkedListStore class."""
    
    def test_store_empty_chunks(self):
        """Test storing empty chunk list."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        result = store.store_chunks([])
        assert result == []
    
    def test_store_single_chunk(self):
        """Test storing a single chunk."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        embeddings = _generate_dummy_embeddings(1)
        chunk_ids = store.store_chunks(["Hello world"], embeddings=embeddings)
        
        assert len(chunk_ids) == 1
        
        chunk = store.get_chunk_by_id(chunk_ids[0])
        assert chunk is not None
        assert chunk["text"] == "Hello world"
        assert chunk["metadata"]["prev_id"] == ""
        assert chunk["metadata"]["next_id"] == ""
        assert chunk["metadata"]["index"] == 0
    
    def test_store_multiple_chunks_linked_list(self):
        """Test that multiple chunks are stored as a linked list."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        embeddings = _generate_dummy_embeddings(len(chunks))
        chunk_ids = store.store_chunks(chunks, embeddings=embeddings)
        
        assert len(chunk_ids) == 3
        
        # Check first chunk
        first = store.get_chunk_by_id(chunk_ids[0])
        assert first["metadata"]["prev_id"] == ""
        assert first["metadata"]["next_id"] == chunk_ids[1]
        
        # Check middle chunk
        second = store.get_chunk_by_id(chunk_ids[1])
        assert second["metadata"]["prev_id"] == chunk_ids[0]
        assert second["metadata"]["next_id"] == chunk_ids[2]
        
        # Check last chunk
        third = store.get_chunk_by_id(chunk_ids[2])
        assert third["metadata"]["prev_id"] == chunk_ids[1]
        assert third["metadata"]["next_id"] == ""
    
    def test_get_nonexistent_chunk(self):
        """Test retrieving a non-existent chunk."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        result = store.get_chunk_by_id("nonexistent-id")
        assert result is None
    
    def test_get_neighbors(self):
        """Test getting neighbors of a chunk."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        chunks = ["First", "Second", "Third"]
        embeddings = _generate_dummy_embeddings(len(chunks))
        chunk_ids = store.store_chunks(chunks, embeddings=embeddings)
        
        # Get both neighbors of middle chunk
        neighbors = store.get_neighbors(chunk_ids[1], direction="both")
        assert len(neighbors) == 2
        
        # Get only previous neighbor
        prev_neighbors = store.get_neighbors(chunk_ids[1], direction="prev")
        assert len(prev_neighbors) == 1
        assert prev_neighbors[0]["text"] == "First"
        
        # Get only next neighbor
        next_neighbors = store.get_neighbors(chunk_ids[1], direction="next")
        assert len(next_neighbors) == 1
        assert next_neighbors[0]["text"] == "Third"
    
    def test_query(self):
        """Test querying for similar chunks."""
        from surgical_rag.storage import LinkedListStore
        import uuid
        
        store = LinkedListStore(collection_name=f"test_query_{uuid.uuid4().hex[:8]}")
        chunks = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language"
        ]
        # Use distinct embeddings for different semantic content
        embeddings = [
            [1.0] + [0.0] * 383,  # fox topic
            [0.0, 1.0] + [0.0] * 382,  # ML topic
            [0.0, 0.0, 1.0] + [0.0] * 381  # Python topic
        ]
        store.store_chunks(chunks, embeddings=embeddings)
        
        # Query with embedding similar to ML topic
        query_embedding = [0.1, 0.9] + [0.0] * 382
        results = store.query("AI and machine learning", n_results=2, query_embedding=query_embedding)
        assert len(results) <= 2
        # The ML-related chunk should be in first position
        assert "machine learning" in results[0]["text"].lower() or "artificial intelligence" in results[0]["text"].lower()
    
    def test_clear(self):
        """Test clearing the store."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        embeddings = _generate_dummy_embeddings(1)
        chunk_ids = store.store_chunks(["Test chunk"], embeddings=embeddings)
        
        # Verify chunk exists
        assert store.get_chunk_by_id(chunk_ids[0]) is not None
        
        # Clear and verify
        store.clear()
        assert store.get_chunk_by_id(chunk_ids[0]) is None


class TestLinkedListNavigation:
    """Tests for linked list navigation."""
    
    def test_empty_id_returns_none(self):
        """Test that empty ID returns None."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        result = store.get_chunk_by_id("")
        assert result is None
    
    def test_navigate_chain(self):
        """Test navigating through the linked list."""
        from surgical_rag.storage import LinkedListStore
        
        store = LinkedListStore()
        chunks = ["A", "B", "C", "D", "E"]
        embeddings = _generate_dummy_embeddings(len(chunks))
        chunk_ids = store.store_chunks(chunks, embeddings=embeddings)
        
        # Start from C and navigate forward
        c = store.get_chunk_by_id(chunk_ids[2])
        assert c["text"] == "C"
        
        d_id = c["metadata"]["next_id"]
        d = store.get_chunk_by_id(d_id)
        assert d["text"] == "D"
        
        e_id = d["metadata"]["next_id"]
        e = store.get_chunk_by_id(e_id)
        assert e["text"] == "E"
        
        # E should have no next
        assert e["metadata"]["next_id"] == ""
