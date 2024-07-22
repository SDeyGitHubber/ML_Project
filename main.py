import os
from dotenv import load_dotenv
import streamlit as st
from llama_functions import load_documents, create_or_load_index, query_index
from chromadb_utils import initialize_chromadb

# Load environment variables
load_dotenv()

# Initialize ChromaDB
chromadb_manager = initialize_chromadb()
if not chromadb_manager:
    st.error("Failed to initialize ChromaDB. Check your credentials and configuration.")
    st.stop()

# Streamlit app
st.title("Document Query with LlamaIndex and ChromaDB")

# Load documents and create or load index
documents = load_documents("data")
index = create_or_load_index(documents, 'storage')

query = st.text_input("Enter your query:")
if st.button("Submit"):
    try:
        response = query_index(index, query)
        st.write(response)
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")