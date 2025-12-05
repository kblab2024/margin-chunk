# Surgical RAG

A semantic chunking and retrieval system using DSPy and ChromaDB.

## Features

1. **Geometric Chunker**: Uses `all-MiniLM-L6-v2` sentence embeddings to split text where cosine similarity between adjacent sentences drops. No LLMs required.

2. **Linked List Storage**: Stores chunks in ChromaDB with `prev_id` and `next_id` metadata for navigation between chunks.

3. **Surgical Retriever**: Custom DSPy module that searches for Top-K anchors, then auto-fetches neighbors via linked list ID lookup.

4. **Chunk Fusion**: Merges overlapping chunk indices (e.g., [4,5] + [5,6] -> [4,5,6]) into continuous text blocks.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Chunking Text

```python
from surgical_rag import GeometricChunker

# Initialize chunker with optional similarity threshold
chunker = GeometricChunker(similarity_threshold=0.5)

# Split text into semantic chunks
text = """
Machine learning is a powerful technique. It enables computers to learn from data.
The weather today is sunny. Birds are singing in the trees.
"""
chunks = chunker.chunk(text)
# Returns: ['Machine learning is a powerful technique. It enables computers to learn from data.',
#           'The weather today is sunny. Birds are singing in the trees.']
```

### 2. Storing Chunks as Linked List

```python
from surgical_rag import LinkedListStore

# Initialize store (in-memory or persistent)
store = LinkedListStore(collection_name="my_docs", persist_directory="./chroma_db")

# Store chunks with automatic linked list metadata
chunk_ids = store.store_chunks(chunks)

# Navigate the linked list
chunk = store.get_chunk_by_id(chunk_ids[0])
neighbors = store.get_neighbors(chunk_ids[0], direction="both")
```

### 3. Retrieving with Context

```python
from surgical_rag import SurgicalRetriever

# Initialize retriever with neighbor fetching
retriever = SurgicalRetriever(
    store=store,
    k=3,                    # Top-K anchors
    fetch_neighbors=True,   # Auto-fetch neighbors
    neighbor_depth=1        # How many neighbors on each side
)

# Retrieve passages (DSPy compatible)
result = retriever.forward("machine learning techniques")
passages = result.passages

# Or get full context with metadata
chunks_with_context = retriever.retrieve_with_context("machine learning techniques")
```

### 4. Fusing Overlapping Chunks

```python
from surgical_rag import ChunkFusion

# Merge overlapping index ranges
indices = [[4, 5], [5, 6], [10, 11]]
merged = ChunkFusion.merge_indices(indices)
# Returns: [[4, 5, 6], [10, 11]]

# Fuse chunks into continuous text blocks
chunks_with_metadata = retriever.retrieve_with_context("query")
fused_blocks = ChunkFusion.fuse_chunks(chunks_with_metadata)

# Get text blocks with their index ranges
blocks = ChunkFusion.get_continuous_blocks(chunks_with_metadata)
# Returns: [("merged text here", [4, 5, 6]), ("other text", [10, 11])]
```

## Complete Example

```python
from surgical_rag import GeometricChunker, LinkedListStore, SurgicalRetriever, ChunkFusion

# 1. Chunk the document
chunker = GeometricChunker(similarity_threshold=0.5)
document = """
Chapter 1: Introduction to Machine Learning.
Machine learning is a subset of artificial intelligence.
It allows systems to learn and improve from experience.

Chapter 2: Neural Networks.
Neural networks are computing systems inspired by biological neural networks.
They consist of interconnected nodes that process information.
"""
chunks = chunker.chunk(document)

# 2. Store in ChromaDB with linked list structure
store = LinkedListStore()
chunk_ids = store.store_chunks(chunks)

# 3. Create surgical retriever
retriever = SurgicalRetriever(store=store, k=2, fetch_neighbors=True, neighbor_depth=1)

# 4. Retrieve and fuse
results = retriever.retrieve_with_context("neural networks AI")
fused = ChunkFusion.fuse_chunks(results)

for block in fused:
    print(f"Indices {block['indices']}: {block['text'][:100]}...")
```

## License

MIT License
