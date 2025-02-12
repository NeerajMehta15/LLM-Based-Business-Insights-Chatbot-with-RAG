# src/chatbot.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEndpoint
import logging
from src.config import Config

class BusinessInsightsChatbot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vector_store = None
        self.conversation_chain = None
        
    def load_documents(self):
        """Load and process PDF documents"""
        try:
            loader = DirectoryLoader(
                Config.UPLOAD_DIR,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            self.logger.info(f"Processed {len(texts)} text chunks")
            return texts
        except Exception as e:
            self.logger.error(f"Error loading documents: {str(e)}")
            raise

    def create_vector_store(self, texts):
        """Create and persist vector store"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            
            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=Config.CHROMA_DIR
            )
            # Removed persist() call as it's no longer needed
            self.logger.info("Vector store created")
        except Exception as e:
            self.logger.error(f"Error creating vector store: {str(e)}")
            raise

    def initialize_llm(self):
        """Initialize HuggingFace model"""
        return HuggingFaceEndpoint(
            repo_id=Config.MODEL_REPO_ID,
            task="text-generation",
            model_kwargs={
                "temperature": Config.TEMPERATURE,
                "max_length": Config.MAX_LENGTH,
                "top_p": Config.TOP_P
            },
            huggingfacehub_api_token=Config.HUGGINGFACE_API_TOKEN
        )

    def setup_conversation_chain(self):
        """Set up the conversational chain"""
        try:
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=5  # Keep last 5 exchanges
            )
            
            llm = self.initialize_llm()
            
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": Config.RETRIEVER_K}
                ),
                memory=memory,
                return_source_documents=True
            )
            self.logger.info("Conversation chain setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up conversation chain: {str(e)}")
            raise

    def process_query(self, query):
        """Process user query and return response"""
        try:
            if not self.conversation_chain:
                raise ValueError("Conversation chain not initialized")
                
            # Using invoke instead of __call__
            response = self.conversation_chain.invoke({"question": query})
            return {
                'answer': response['answer'],
                'sources': [doc.metadata for doc in response['source_documents']]
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise