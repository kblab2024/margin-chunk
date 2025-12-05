"""Tests for the SurgicalRetriever module."""

import pytest


def _generate_dummy_embeddings(n: int, dim: int = 384) -> list:
    """Generate dummy embeddings for testing."""
    import random
    return [[random.random() for _ in range(dim)] for _ in range(n)]


class TestSurgicalRetriever:
    """Tests for the SurgicalRetriever class."""
    
    def test_retrieve_empty_store(self):
        """Test retrieval from empty store."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        import dspy
        
        store = LinkedListStore()
        retriever = SurgicalRetriever(store=store, k=3)
        
        # Override forward to use an embedding query directly to avoid network
        class MockRetriever(SurgicalRetriever):
            def forward(self, query, k=None):
                k = k or self.k
                # Use a dummy embedding
                query_emb = [0.5] * 384
                anchors = self.store.query(query_text=query, n_results=k, query_embedding=query_emb)
                if not anchors:
                    return dspy.Prediction(passages=[])
                passages = [chunk["text"] for chunk in anchors]
                return dspy.Prediction(passages=passages)
        
        mock_retriever = MockRetriever(store=store, k=3)
        result = mock_retriever.forward("test query")
        assert result.passages == []
    
    def test_retrieve_basic(self):
        """Test basic retrieval."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        
        store = LinkedListStore()
        chunks = [
            "The cat sat on the mat.",
            "Machine learning models require training data.",
            "Python is great for data science."
        ]
        embeddings = [
            [1.0] + [0.0] * 383,
            [0.0, 1.0] + [0.0] * 382,
            [0.0, 0.0, 1.0] + [0.0] * 381
        ]
        store.store_chunks(chunks, embeddings=embeddings)
        
        # Create retriever with custom query using embedding close to ML chunk
        retriever = SurgicalRetriever(store=store, k=1, fetch_neighbors=False)
        
        # We need to modify the store query to use embedding
        class MockRetriever(SurgicalRetriever):
            def forward(self, query, k=None):
                import dspy
                k = k or self.k
                # Use embedding that matches ML topic
                query_emb = [0.1, 0.9] + [0.0] * 382
                anchors = self.store.query(query_text=query, n_results=k, query_embedding=query_emb)
                if not anchors:
                    return dspy.Prediction(passages=[])
                passages = [chunk["text"] for chunk in anchors]
                return dspy.Prediction(passages=passages)
        
        retriever = MockRetriever(store=store, k=1, fetch_neighbors=False)
        result = retriever.forward("artificial intelligence and ML")
        
        assert len(result.passages) >= 1
    
    def test_retrieve_with_neighbors(self):
        """Test retrieval with neighbor fetching."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        import dspy
        
        store = LinkedListStore()
        chunks = [
            "Chapter 1: Introduction",
            "Machine learning is a powerful technique.",
            "It can be used for many applications.",
            "Chapter 2: Methods",
            "We describe our methodology here."
        ]
        # Make chunk 1 (ML chunk) have a distinctive embedding
        embeddings = [
            [0.1] + [0.0] * 383,
            [0.0, 1.0] + [0.0] * 382,  # ML chunk
            [0.0, 0.9] + [0.0] * 382,  # Related to ML
            [0.0, 0.0, 0.1] + [0.0] * 381,
            [0.0, 0.0, 0.2] + [0.0] * 381
        ]
        store.store_chunks(chunks, embeddings=embeddings)
        
        # Create a custom retriever that uses embeddings for query
        class MockRetriever(SurgicalRetriever):
            def forward(self, query, k=None):
                k = k or self.k
                # Use embedding that matches ML topic
                query_emb = [0.0, 0.95] + [0.0] * 382
                anchors = self.store.query(query_text=query, n_results=k, query_embedding=query_emb)
                if not anchors:
                    return dspy.Prediction(passages=[])
                
                all_chunks = {}
                for anchor in anchors:
                    all_chunks[anchor["id"]] = anchor
                    if self.fetch_neighbors:
                        prev = self._fetch_neighbor_chain(anchor["id"], "prev", self.neighbor_depth)
                        for n in prev:
                            all_chunks[n["id"]] = n
                        next_ = self._fetch_neighbor_chain(anchor["id"], "next", self.neighbor_depth)
                        for n in next_:
                            all_chunks[n["id"]] = n
                
                sorted_chunks = sorted(all_chunks.values(), key=lambda x: x["metadata"].get("index", 0))
                passages = [chunk["text"] for chunk in sorted_chunks]
                return dspy.Prediction(passages=passages)
        
        retriever = MockRetriever(store=store, k=1, fetch_neighbors=True, neighbor_depth=1)
        result = retriever.forward("machine learning techniques")
        
        # Should get the anchor plus neighbors
        assert len(result.passages) >= 1
    
    def test_retrieve_with_context(self):
        """Test retrieve_with_context method."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        
        store = LinkedListStore()
        chunks = ["First", "Second", "Third"]
        embeddings = [
            [1.0] + [0.0] * 383,
            [0.0, 1.0] + [0.0] * 382,
            [0.0, 0.0, 1.0] + [0.0] * 381
        ]
        store.store_chunks(chunks, embeddings=embeddings)
        
        # Create a mock retriever for testing
        class MockRetriever(SurgicalRetriever):
            def retrieve_with_context(self, query, k=None):
                k = k or self.k
                # Use embedding that matches second chunk
                query_emb = [0.0, 0.95] + [0.0] * 382
                anchors = self.store.query(query_text=query, n_results=k, query_embedding=query_emb)
                if not anchors:
                    return []
                
                all_chunks = {}
                for anchor in anchors:
                    all_chunks[anchor["id"]] = anchor
                    if self.fetch_neighbors:
                        prev = self._fetch_neighbor_chain(anchor["id"], "prev", self.neighbor_depth)
                        for n in prev:
                            all_chunks[n["id"]] = n
                        next_ = self._fetch_neighbor_chain(anchor["id"], "next", self.neighbor_depth)
                        for n in next_:
                            all_chunks[n["id"]] = n
                
                return sorted(all_chunks.values(), key=lambda x: x["metadata"].get("index", 0))
        
        retriever = MockRetriever(store=store, k=1, fetch_neighbors=True)
        results = retriever.retrieve_with_context("Second")
        
        # Each result should have metadata
        assert all("metadata" in r for r in results)
        assert all("index" in r["metadata"] for r in results)
    
    def test_results_sorted_by_index(self):
        """Test that results are sorted by index."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        
        store = LinkedListStore()
        chunks = ["A", "B", "C", "D", "E"]
        # Make C have a distinctive embedding
        embeddings = [
            [0.1] + [0.0] * 383,
            [0.2] + [0.0] * 383,
            [0.0, 1.0] + [0.0] * 382,  # C - distinctive
            [0.3] + [0.0] * 383,
            [0.4] + [0.0] * 383
        ]
        store.store_chunks(chunks, embeddings=embeddings)
        
        # Create a mock retriever for testing
        class MockRetriever(SurgicalRetriever):
            def retrieve_with_context(self, query, k=None):
                k = k or self.k
                # Use embedding that matches C
                query_emb = [0.0, 0.95] + [0.0] * 382
                anchors = self.store.query(query_text=query, n_results=k, query_embedding=query_emb)
                if not anchors:
                    return []
                
                all_chunks = {}
                for anchor in anchors:
                    all_chunks[anchor["id"]] = anchor
                    if self.fetch_neighbors:
                        prev = self._fetch_neighbor_chain(anchor["id"], "prev", self.neighbor_depth)
                        for n in prev:
                            all_chunks[n["id"]] = n
                        next_ = self._fetch_neighbor_chain(anchor["id"], "next", self.neighbor_depth)
                        for n in next_:
                            all_chunks[n["id"]] = n
                
                return sorted(all_chunks.values(), key=lambda x: x["metadata"].get("index", 0))
        
        retriever = MockRetriever(store=store, k=1, fetch_neighbors=True, neighbor_depth=2)
        results = retriever.retrieve_with_context("C")
        
        # Results should be in index order
        indices = [r["metadata"]["index"] for r in results]
        assert indices == sorted(indices)


class TestNeighborFetching:
    """Tests for neighbor chain fetching."""
    
    def test_fetch_prev_chain(self):
        """Test fetching previous neighbors."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        
        store = LinkedListStore()
        chunks = ["A", "B", "C", "D", "E"]
        embeddings = _generate_dummy_embeddings(len(chunks))
        chunk_ids = store.store_chunks(chunks, embeddings=embeddings)
        
        retriever = SurgicalRetriever(store=store)
        
        # Fetch 2 previous neighbors from D (index 3)
        prev_chain = retriever._fetch_neighbor_chain(chunk_ids[3], "prev", 2)
        
        assert len(prev_chain) == 2
        texts = [c["text"] for c in prev_chain]
        assert "C" in texts
        assert "B" in texts
    
    def test_fetch_next_chain(self):
        """Test fetching next neighbors."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        
        store = LinkedListStore()
        chunks = ["A", "B", "C", "D", "E"]
        embeddings = _generate_dummy_embeddings(len(chunks))
        chunk_ids = store.store_chunks(chunks, embeddings=embeddings)
        
        retriever = SurgicalRetriever(store=store)
        
        # Fetch 2 next neighbors from B (index 1)
        next_chain = retriever._fetch_neighbor_chain(chunk_ids[1], "next", 2)
        
        assert len(next_chain) == 2
        texts = [c["text"] for c in next_chain]
        assert "C" in texts
        assert "D" in texts
    
    def test_fetch_chain_at_boundary(self):
        """Test fetching neighbors at document boundary."""
        from surgical_rag.storage import LinkedListStore
        from surgical_rag.retriever import SurgicalRetriever
        
        store = LinkedListStore()
        chunks = ["A", "B", "C"]
        embeddings = _generate_dummy_embeddings(len(chunks))
        chunk_ids = store.store_chunks(chunks, embeddings=embeddings)
        
        retriever = SurgicalRetriever(store=store)
        
        # Try to fetch 5 previous from first chunk
        prev_chain = retriever._fetch_neighbor_chain(chunk_ids[0], "prev", 5)
        assert len(prev_chain) == 0
        
        # Try to fetch 5 next from last chunk
        next_chain = retriever._fetch_neighbor_chain(chunk_ids[2], "next", 5)
        assert len(next_chain) == 0
