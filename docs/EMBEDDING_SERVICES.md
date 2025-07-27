# Embedding Services Usage Guide

## Overview

The medical diagnosis system now supports multiple embedding services that you can easily switch between:

1. **SentenceTransformers** (Recommended for quality)
2. **Together AI** (For API-based embeddings)
3. **Simple Hash** (Fast fallback option)

## Quick Start

### 1. Switch Embedding Service

Edit the `.env` file and change the `EMBEDDING_SERVICE` variable:

```bash
# For best quality (requires sentence-transformers package)
EMBEDDING_SERVICE=sentence_transformers

# For Together AI integration
EMBEDDING_SERVICE=together_ai

# For fast fallback (no dependencies)
EMBEDDING_SERVICE=simple_hash
```

### 2. Run the Embedding Script

```bash
cd src/knowledge
python run_emb.py
```

## Detailed Configuration

### SentenceTransformers
- **Best for**: High-quality embeddings, research, production
- **Requirements**: `pip install sentence-transformers`
- **Model**: `all-MiniLM-L6-v2` (384 dimensions, 50MB)
- **Speed**: Medium (CPU-based)

```bash
# Set in .env
EMBEDDING_SERVICE=sentence_transformers
```

### Together AI
- **Best for**: API-based scaling, cloud deployment
- **Requirements**: Together AI API key
- **Model**: Uses hash-based fallback (768 dimensions)
- **Speed**: Fast (but requires API calls)

```bash
# Set in .env
EMBEDDING_SERVICE=together_ai
TOGETHER_API_KEY=your_api_key_here
```

### Simple Hash
- **Best for**: Testing, development, no dependencies
- **Requirements**: None
- **Model**: Hash-based (384 dimensions)
- **Speed**: Very fast

```bash
# Set in .env
EMBEDDING_SERVICE=simple_hash
```

## Usage Examples

### Test All Services

```bash
cd src/knowledge
python test_embeddings.py --test
```

### Switch Services Programmatically

```bash
# Switch to Together AI
python test_embeddings.py --switch together_ai

# Switch to SentenceTransformers
python test_embeddings.py --switch sentence_transformers

# Switch to Simple Hash
python test_embeddings.py --switch simple_hash
```

### Manual Switching

1. Edit `.env` file:
```
EMBEDDING_SERVICE=together_ai
```

2. Run embedding script:
```bash
python run_emb.py
```

## Performance Comparison

| Service | Quality | Speed | Dependencies | Use Case |
|---------|---------|-------|--------------|----------|
| SentenceTransformers | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | sentence-transformers | Production, Research |
| Together AI | ⭐⭐⭐ | ⭐⭐⭐⭐ | together, API key | Cloud, Scaling |
| Simple Hash | ⭐⭐ | ⭐⭐⭐⭐⭐ | None | Testing, Development |

## Troubleshooting

### SentenceTransformers Issues
```bash
# Install dependencies
pip install sentence-transformers torch

# For M1/M2 Macs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Together AI Issues
```bash
# Check API key
echo $TOGETHER_API_KEY

# Install Together AI
pip install together
```

### Simple Hash Fallback
If other services fail, the system automatically falls back to simple hash embeddings.

## Integration with Medical Diagnosis System

The embedding service is automatically used by:
- Knowledge base creation (`knowledge_base.py`)
- Document search and retrieval
- Medical textbook processing

The system will automatically detect and use the configured embedding service.

## Advanced Configuration

### Custom Embedding Models

You can modify the embedding services in `run_emb.py`:

```python
# Custom SentenceTransformers model
ST_MODEL_NAME = "all-mpnet-base-v2"  # Higher quality, larger size

# Custom Together AI model (if they add embedding support)
TOGETHER_MODEL = "your-preferred-model"
```

### Batch Processing

Adjust batch sizes for different services:

```python
# In run_emb.py
embeddings = embedding_service.encode(
    docs, 
    batch_size=256,  # Adjust based on memory/speed needs
    show_progress=True
)
```

## Best Practices

1. **Development**: Use `simple_hash` for quick testing
2. **Production**: Use `sentence_transformers` for best quality
3. **Scaling**: Use `together_ai` for API-based deployment
4. **Always test**: Use `test_embeddings.py` before processing large datasets
