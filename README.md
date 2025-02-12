# Business Insights Chatbot

A powerful chatbot implementation using LangChain and Hugging Face models to analyze and provide insights from business documents. The chatbot processes PDF documents, creates embeddings, and enables natural language conversations about the document contents.

## Features

- PDF document processing and analysis
- Vector storage using Chroma DB
- Conversational memory with context window
- Document chunking and semantic search
- Integration with Hugging Face models
- Comprehensive error handling and logging

## Prerequisites

- Python 3.8+
- Hugging Face API token
- Sufficient storage space for document embeddings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/business-insights-chatbot.git
cd business-insights-chatbot
```

2. Install required packages:
```bash
pip install langchain langchain-community langchain-huggingface chromadb pypdf
```

3. Set up your environment variables:
```bash
export HUGGINGFACE_API_TOKEN=your_token_here
```

## Configuration

Create a `config.py` file in the `src` directory with the following settings:

```python
class Config:
    UPLOAD_DIR = "path/to/pdf/directory"
    CHROMA_DIR = "path/to/chroma/storage"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    MODEL_REPO_ID = "your-chosen-model-repo"
    TEMPERATURE = 0.7
    MAX_LENGTH = 512
    TOP_P = 0.95
    RETRIEVER_K = 4
    HUGGINGFACE_API_TOKEN = "your-token-here"
```

## Usage

```python
from business_insights_chatbot import BusinessInsightsChatbot

# Initialize the chatbot
chatbot = BusinessInsightsChatbot()

# Load and process documents
texts = chatbot.load_documents()

# Create vector store
chatbot.create_vector_store(texts)

# Set up conversation chain
chatbot.setup_conversation_chain()

# Process queries
response = chatbot.process_query("What are the key financial metrics for Q2?")
print(response['answer'])
print("Sources:", response['sources'])
```

## Architecture

The chatbot uses a pipeline architecture:
1. Document Loading: Processes PDF files from the specified directory
2. Text Splitting: Chunks documents into manageable pieces
3. Embedding Generation: Creates vector embeddings using Hugging Face models
4. Vector Storage: Stores embeddings in Chroma DB
5. Query Processing: Uses conversation chain to process user queries and retrieve relevant information

## Error Handling

The implementation includes comprehensive error handling:
- Document loading errors
- Vector store creation failures
- Model initialization issues
- Query processing errors

All errors are logged using Python's logging module for debugging and monitoring.

## Memory Management

The chatbot maintains a conversation history of the last 5 exchanges using ConversationBufferWindowMemory. This provides context for follow-up questions while managing memory usage.

## Customization

You can customize the chatbot by modifying:
- Chunk size and overlap in text splitting
- Number of retrieved documents (RETRIEVER_K)
- Model parameters (temperature, max_length, top_p)
- Memory window size
- Document loading patterns

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the excellent framework
- Hugging Face for model hosting and embeddings
- Chroma DB for vector storage
