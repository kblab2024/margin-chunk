"""Geometric Chunker using sentence embeddings for semantic text splitting."""

import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class GeometricChunker:
    """
    Splits text into semantic chunks using geometric splitting based on
    cosine similarity between adjacent sentences.
    
    Uses `all-MiniLM-L6-v2` model for sentence embeddings. No LLMs involved.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.5):
        """
        Initialize the chunker.
        
        Args:
            model_name: The sentence transformer model to use.
            similarity_threshold: Threshold below which to split chunks.
                                  Lower values create larger chunks.
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks using geometric splitting.
        
        The algorithm:
        1. Split text into sentences
        2. Compute embeddings for each sentence
        3. Calculate cosine similarity between adjacent sentences
        4. Split where similarity drops below threshold
        
        Args:
            text: The input text to chunk.
            
        Returns:
            List of text chunks.
        """
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return sentences if sentences else []
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        # Find split points based on similarity drops
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = self._compute_cosine_similarity(embeddings[i - 1], embeddings[i])
            
            if similarity < self.similarity_threshold:
                # Similarity dropped - create new chunk
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i]]
            else:
                # Continue current chunk
                current_chunk_sentences.append(sentences[i])
        
        # Add the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        return chunks
    
    def chunk_with_metadata(self, text: str) -> List[dict]:
        """
        Split text and return chunks with index metadata.
        
        Args:
            text: The input text to chunk.
            
        Returns:
            List of dicts with 'text' and 'index' keys.
        """
        chunks = self.chunk(text)
        return [{"text": chunk, "index": i} for i, chunk in enumerate(chunks)]
