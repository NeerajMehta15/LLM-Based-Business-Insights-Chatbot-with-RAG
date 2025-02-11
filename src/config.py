import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
    MODEL_REPO_ID = "mistralai/Mistral-7B-v0.1"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TEMPERATURE = 0.7
    MAX_LENGTH = 512
    TOP_P = 0.95
    RETRIEVER_K = 3
    
    # Directories
    UPLOAD_DIR = "uploaded_pdfs"
    CHROMA_DIR = "chroma_db"
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"