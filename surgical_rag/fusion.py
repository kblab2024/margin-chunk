"""Chunk Fusion module to merge overlapping chunk indices into continuous text blocks."""

from typing import List, Tuple


class ChunkFusion:
    """
    Merges overlapping chunk indices into continuous text blocks.
    
    Example: [4,5] + [5,6] -> [4,5,6] merged into one continuous block.
    """
    
    @staticmethod
    def merge_indices(indices_list: List[List[int]]) -> List[List[int]]:
        """
        Merge overlapping or adjacent index ranges.
        
        Args:
            indices_list: List of index lists, e.g., [[4, 5], [5, 6], [10, 11]]
            
        Returns:
            Merged index ranges, e.g., [[4, 5, 6], [10, 11]]
        """
        if not indices_list:
            return []
        
        # Flatten and get unique indices, then sort
        all_indices = set()
        for indices in indices_list:
            all_indices.update(indices)
        
        if not all_indices:
            return []
        
        sorted_indices = sorted(all_indices)
        
        # Group consecutive indices
        merged = []
        current_group = [sorted_indices[0]]
        
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == sorted_indices[i - 1] + 1:
                # Consecutive - add to current group
                current_group.append(sorted_indices[i])
            else:
                # Gap - start new group
                merged.append(current_group)
                current_group = [sorted_indices[i]]
        
        # Add the last group
        merged.append(current_group)
        
        return merged
    
    @staticmethod
    def fuse_chunks(chunks: List[dict]) -> List[dict]:
        """
        Fuse chunks with overlapping or adjacent indices into continuous blocks.
        
        Args:
            chunks: List of chunk dicts with 'text' and 'metadata' (containing 'index').
            
        Returns:
            List of fused chunk dicts with merged text and index ranges.
        """
        if not chunks:
            return []
        
        # Sort chunks by index
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get("metadata", {}).get("index", 0)
        )
        
        # Group consecutive chunks
        fused = []
        current_group = [sorted_chunks[0]]
        
        for i in range(1, len(sorted_chunks)):
            current_idx = sorted_chunks[i].get("metadata", {}).get("index", 0)
            prev_idx = current_group[-1].get("metadata", {}).get("index", 0)
            
            if current_idx == prev_idx + 1:
                # Consecutive - add to current group
                current_group.append(sorted_chunks[i])
            else:
                # Gap - finalize current group and start new one
                fused.append(ChunkFusion._merge_group(current_group))
                current_group = [sorted_chunks[i]]
        
        # Add the last group
        fused.append(ChunkFusion._merge_group(current_group))
        
        return fused
    
    @staticmethod
    def _merge_group(chunks: List[dict]) -> dict:
        """
        Merge a group of consecutive chunks into a single block.
        
        Args:
            chunks: List of consecutive chunk dicts.
            
        Returns:
            Merged chunk dict with combined text and index range.
        """
        if not chunks:
            return {}
        
        if len(chunks) == 1:
            return {
                "text": chunks[0]["text"],
                "indices": [chunks[0].get("metadata", {}).get("index", 0)],
                "metadata": chunks[0].get("metadata", {})
            }
        
        # Merge text
        merged_text = " ".join(chunk["text"] for chunk in chunks)
        
        # Get index range
        indices = [chunk.get("metadata", {}).get("index", 0) for chunk in chunks]
        
        return {
            "text": merged_text,
            "indices": indices,
            "metadata": {
                "start_index": min(indices),
                "end_index": max(indices),
                "chunk_count": len(chunks)
            }
        }
    
    @staticmethod
    def get_continuous_blocks(chunks: List[dict]) -> List[Tuple[str, List[int]]]:
        """
        Get continuous text blocks with their index ranges.
        
        Args:
            chunks: List of chunk dicts.
            
        Returns:
            List of (text, indices) tuples.
        """
        fused = ChunkFusion.fuse_chunks(chunks)
        return [(block["text"], block["indices"]) for block in fused]
