"""Tests for the GeometricChunker module."""

from unittest.mock import MagicMock, patch
import pytest
import numpy as np


class TestGeometricChunker:
    """Tests for the GeometricChunker class."""
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_empty_text(self, mock_st):
        """Test chunking empty text."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        chunker = GeometricChunker()
        result = chunker.chunk("")
        assert result == []
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_single_sentence(self, mock_st):
        """Test chunking a single sentence."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        chunker = GeometricChunker()
        result = chunker.chunk("This is a single sentence.")
        assert len(result) == 1
        assert result[0] == "This is a single sentence."
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_similar_sentences_stay_together(self, mock_st):
        """Test that semantically similar sentences stay in the same chunk."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        # Return embeddings that are very similar (high cosine similarity)
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],  # sentence 1
            [0.9, 0.1, 0.0],  # sentence 2 - similar to 1
            [0.85, 0.15, 0.0]  # sentence 3 - similar to 2
        ])
        mock_st.return_value = mock_model
        
        # Use a lower threshold to be more lenient about keeping sentences together
        chunker = GeometricChunker(similarity_threshold=0.3)
        text = "The cat sat on the mat. The cat was very comfortable. The cat fell asleep."
        result = chunker.chunk(text)
        # Similar sentences should stay together
        assert len(result) == 1
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_dissimilar_sentences_split(self, mock_st):
        """Test that semantically dissimilar sentences are split."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        # Return embeddings that are very different (low cosine similarity)
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],  # sentence 1
            [0.0, 1.0, 0.0],  # sentence 2 - orthogonal to 1
            [0.0, 0.0, 1.0]   # sentence 3 - orthogonal to 2
        ])
        mock_st.return_value = mock_model
        
        # Use a high threshold to force splits
        chunker = GeometricChunker(similarity_threshold=0.9)
        text = (
            "The cat sat on the mat. "
            "Quantum physics describes the behavior of particles at atomic scales. "
            "I love chocolate ice cream."
        )
        result = chunker.chunk(text)
        # Dissimilar sentences should be split
        assert len(result) == 3
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_chunk_with_metadata(self, mock_st):
        """Test chunking with metadata."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        # Return embeddings that are very similar
        mock_model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0]
        ])
        mock_st.return_value = mock_model
        
        chunker = GeometricChunker(similarity_threshold=0.5)
        text = "First sentence. Second sentence. Third sentence."
        result = chunker.chunk_with_metadata(text)
        
        assert all("text" in item for item in result)
        assert all("index" in item for item in result)
        for i, item in enumerate(result):
            assert item["index"] == i
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_sentence_splitting(self, mock_st):
        """Test that sentences are split correctly."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        chunker = GeometricChunker()
        sentences = chunker._split_into_sentences("Hello world! How are you? I am fine.")
        assert len(sentences) == 3
        assert sentences[0] == "Hello world!"
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine."


class TestCosineSimlarity:
    """Tests for cosine similarity calculation."""
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_identical_vectors(self, mock_st):
        """Test that identical vectors have similarity 1."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        chunker = GeometricChunker()
        vec = np.array([1.0, 2.0, 3.0])
        similarity = chunker._compute_cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_orthogonal_vectors(self, mock_st):
        """Test that orthogonal vectors have similarity 0."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        chunker = GeometricChunker()
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        similarity = chunker._compute_cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6
    
    @patch('surgical_rag.chunker.SentenceTransformer')
    def test_zero_vector(self, mock_st):
        """Test handling of zero vectors."""
        from surgical_rag.chunker import GeometricChunker
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        chunker = GeometricChunker()
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])
        similarity = chunker._compute_cosine_similarity(vec1, vec2)
        assert similarity == 0.0
