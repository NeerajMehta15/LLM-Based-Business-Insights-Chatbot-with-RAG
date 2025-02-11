import streamlit as st
from src.chatbot import BusinessInsightsChatbot
from src.config import Config
from src.utils import setup_logging, ensure_directories
import os

def main():
    setup_logging()
    ensure_directories()
    
    st.title("Business Insights Chatbot")
    st.write("Upload annual reports and ask questions about the business")
    
    if not Config.HUGGINGFACE_API_TOKEN:
        st.error("Please set your HuggingFace API token in the .env file")
        st.stop()
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(Config.UPLOAD_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
        
        chatbot = BusinessInsightsChatbot()
        
        with st.spinner("Processing documents..."):
            texts = chatbot.load_documents()
            chatbot.create_vector_store(texts)
            chatbot.setup_conversation_chain()
        
        st.success("Ready to answer questions!")
        query = st.text_input("Ask a question about the business:")
        
        if query:
            with st.spinner("Generating response..."):
                try:
                    response = chatbot.process_query(query)
                    st.write("Answer:", response['answer'])
                    
                    with st.expander("View Sources"):
                        for source in response['sources']:
                            st.write(f"- {source['source']}, Page {source['page']}")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()