```markdown
# RAGister

A production-ready Retrieval-Augmented Generation (RAG) Q&A pipeline for private documents supporting PDF, DOCX, and Wikipedia sources. Features Pinecone and Chroma vector stores with conversation memory.

## Features

- Document loading: PDF, DOCX, Wikipedia
- Intelligent chunking with overlap
- Dual vector store support: Pinecone (cloud) and Chroma (local)
- Conversation memory with full chat history
- Interactive CLI with commands: `help`, `history`, `clear`, `delete_index`
- Production-grade exception handling
- Embedding cost estimation
- Streamlit GUI option

## Quick Start (CLI)

```bash
git clone https://github.com/AnubhavKarki/RAGister.git
cd RAGister
pip install -q -r requirements.txt
python rag_pipeline.py
```

## Installation

```bash
pip install pypdf langchain langchain_community langchain_openai docx2txt wikipedia langchain_text_splitters langchain_pinecone chromadb streamlit-chat
```

**Environment Variables:**
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key  # Optional for Pinecone
```

## Pipeline Flow

1. **Load**: PDF/DOCX files or Wikipedia pages
2. **Chunk**: Intelligent text splitting (256 tokens, 50 overlap)
3. **Embed**: OpenAI `text-embedding-3-small` (1536 dimensions)
4. **Store**: Pinecone index or Chroma local database
5. **Query**: Semantic search + LLM generation with memory

## Usage Examples

```
1. Load Document:
------------------------------
Enter filename (.pdf, .docx) or 'wikipedia': document.pdf

2. Chunking data...
Created 45 chunks

3. Embedding Cost:
Total Tokens: 12,456
Embedding Cost in USD: 0.004982

4. Choose Vector Store:
Enter 'pinecone' or 'chroma': pinecone
Enter Pinecone index name: my-rag-index

Q&A Ready!
Commands: 'quit', 'clear', 'delete_index', 'history', 'help'
Messages in history: 0
--------------------------------------------------
Your question: What is this document about?
```

## Commands

| Command               | Description                                 |
|-----------------------|---------------------------------------------|
| `quit`/`exit`/`q`     | Exit application                            |
| `clear`               | Clear conversation history                  |
| `delete_index`        | Delete current indexes and restart pipeline |
| `history`             | Show recent chat history                    |
| `help`                | Show this command menu                      |

## Vector Stores

### Pinecone (Cloud)
```python
vector_store = insert_or_fetch_embeddings("my-index", chunks)
```
- Automatic index creation
- Serverless (AWS us-east-1)
- Cosine similarity metric

### Chroma (Local)
```python
vector_store = create_embeddings_chroma(chunks, "/tmp/chroma_db")
```
- Persistent SQLite storage
- In-memory fallback
- Colab-compatible (/tmp/)

## GUI (Streamlit)

```bash
pip install streamlit streamlit-chat
streamlit run rag_pipeline.py
```

Features file upload, chat interface, and full RAGister integration.

## Architecture

```
Document → Chunking → Embeddings → Vector Store → RAG Chain → LLM
                ↓
            Chat History (LCEL + MessagesPlaceholder)
```

**Key Components:**
- **Retriever**: Similarity search (k=3 nearest neighbors)
- **Memory**: `ChatMessageHistory` + `MessagesPlaceholder`
- **Chain**: `RunnableParallel` → `ChatPromptTemplate` → `ChatOpenAI`
- **LLM**: GPT-4o-mini (`gpt-4o-mini`)

## Requirements

See `requirements.txt`:
```
pypdf
langchain
langchain_community
langchain_openai
docx2txt
wikipedia
langchain_text_splitters
langchain_pinecone
chromadb
streamlit-chat
```

## Error Handling

- Input validation on all user prompts
- Vector store creation fallbacks
- Chroma directory permissions (/tmp/)
- Graceful index deletion
- KeyboardInterrupt support

## Production Notes

- **Scalable**: Pinecone handles millions of vectors
- **Memory**: In-memory chat history (persistent via file export)
- **Cost**: Embedding costs displayed pre-processing
- **Robust**: Full exception handling, no crashes

## License

MIT License - see LICENSE file for details.

## Author

[Anubhav Karki](https://github.com/AnubhavKarki)
```