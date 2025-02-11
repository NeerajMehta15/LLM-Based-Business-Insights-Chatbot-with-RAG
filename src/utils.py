import logging
import os

def setup_logging():
    """Configure logging for the application"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/app.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [Config.UPLOAD_DIR, Config.CHROMA_DIR, 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)