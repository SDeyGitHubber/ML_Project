import os
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.response.pprint_utils import pprint_response

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_documents(data_dir: str):
    """Load documents from the specified directory."""
    return SimpleDirectoryReader(data_dir).load_data()

def create_or_load_index(documents, persist_dir: str):
    """Create a new index or load an existing one."""
    if not os.path.exists(persist_dir):
        embeddings = [embedding_model.encode(doc.get_content()) for doc in documents]
        index = VectorStoreIndex.from_embeddings(documents, embeddings)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    return index

def query_index(index, query: str):
    """Query the index and return the response."""
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    pprint_response(response, show_source=True)
    return response