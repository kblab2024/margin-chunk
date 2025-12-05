"""Tests for the ChunkFusion module."""

import pytest

from surgical_rag.fusion import ChunkFusion


class TestMergeIndices:
    """Tests for index merging."""
    
    def test_empty_list(self):
        """Test merging empty list."""
        result = ChunkFusion.merge_indices([])
        assert result == []
    
    def test_single_range(self):
        """Test merging single range."""
        result = ChunkFusion.merge_indices([[1, 2, 3]])
        assert result == [[1, 2, 3]]
    
    def test_overlapping_ranges(self):
        """Test merging overlapping ranges."""
        # [4,5] + [5,6] should merge to [4,5,6]
        result = ChunkFusion.merge_indices([[4, 5], [5, 6]])
        assert result == [[4, 5, 6]]
    
    def test_non_overlapping_ranges(self):
        """Test non-overlapping ranges stay separate."""
        result = ChunkFusion.merge_indices([[1, 2], [5, 6]])
        assert result == [[1, 2], [5, 6]]
    
    def test_complex_merge(self):
        """Test complex merging scenario."""
        # [1,2] + [2,3] + [5,6] + [8,9] + [9,10] -> [1,2,3], [5,6], [8,9,10]
        result = ChunkFusion.merge_indices([[1, 2], [2, 3], [5, 6], [8, 9], [9, 10]])
        assert result == [[1, 2, 3], [5, 6], [8, 9, 10]]
    
    def test_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        result = ChunkFusion.merge_indices([[5, 6], [1, 2], [2, 3]])
        assert result == [[1, 2, 3], [5, 6]]


class TestFuseChunks:
    """Tests for chunk fusion."""
    
    def test_empty_chunks(self):
        """Test fusing empty chunk list."""
        result = ChunkFusion.fuse_chunks([])
        assert result == []
    
    def test_single_chunk(self):
        """Test fusing single chunk."""
        chunks = [{"text": "Hello", "metadata": {"index": 0}}]
        result = ChunkFusion.fuse_chunks(chunks)
        
        assert len(result) == 1
        assert result[0]["text"] == "Hello"
        assert result[0]["indices"] == [0]
    
    def test_consecutive_chunks_merge(self):
        """Test that consecutive chunks are merged."""
        chunks = [
            {"text": "Hello", "metadata": {"index": 0}},
            {"text": "world", "metadata": {"index": 1}},
            {"text": "!", "metadata": {"index": 2}}
        ]
        result = ChunkFusion.fuse_chunks(chunks)
        
        assert len(result) == 1
        assert result[0]["text"] == "Hello world !"
        assert result[0]["indices"] == [0, 1, 2]
    
    def test_non_consecutive_chunks_stay_separate(self):
        """Test that non-consecutive chunks stay separate."""
        chunks = [
            {"text": "First", "metadata": {"index": 0}},
            {"text": "Second", "metadata": {"index": 5}},
            {"text": "Third", "metadata": {"index": 10}}
        ]
        result = ChunkFusion.fuse_chunks(chunks)
        
        assert len(result) == 3
    
    def test_mixed_consecutive_and_gaps(self):
        """Test chunks with both consecutive and gaps."""
        chunks = [
            {"text": "A", "metadata": {"index": 0}},
            {"text": "B", "metadata": {"index": 1}},
            {"text": "C", "metadata": {"index": 5}},
            {"text": "D", "metadata": {"index": 6}}
        ]
        result = ChunkFusion.fuse_chunks(chunks)
        
        assert len(result) == 2
        assert result[0]["text"] == "A B"
        assert result[0]["indices"] == [0, 1]
        assert result[1]["text"] == "C D"
        assert result[1]["indices"] == [5, 6]
    
    def test_unsorted_chunks(self):
        """Test that unsorted chunks are handled correctly."""
        chunks = [
            {"text": "B", "metadata": {"index": 1}},
            {"text": "A", "metadata": {"index": 0}},
            {"text": "C", "metadata": {"index": 2}}
        ]
        result = ChunkFusion.fuse_chunks(chunks)
        
        assert len(result) == 1
        assert result[0]["text"] == "A B C"
        assert result[0]["indices"] == [0, 1, 2]


class TestGetContinuousBlocks:
    """Tests for get_continuous_blocks."""
    
    def test_basic_blocks(self):
        """Test getting continuous blocks."""
        chunks = [
            {"text": "Hello", "metadata": {"index": 0}},
            {"text": "world", "metadata": {"index": 1}},
            {"text": "!", "metadata": {"index": 5}}
        ]
        result = ChunkFusion.get_continuous_blocks(chunks)
        
        assert len(result) == 2
        assert result[0] == ("Hello world", [0, 1])
        assert result[1] == ("!", [5])
