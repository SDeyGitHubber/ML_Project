import chromadb
from chromadb.config import Settings

class ChromaDBManager:
    def __init__(self):
        settings = Settings()
        self.client = chromadb.Client(settings)
        self.collection = self.client.create_collection('chatbot_history')

    def save_to_db(self, query, response):
        self.collection.add(
            documents=[{'query': query, 'response': response}],
            metadatas=[],
            ids=[]
        )

    def fetch_from_db(self, query):
        results = self.collection.query({'query': query})
        if results:
            return results[0]['response']
        return None

def initialize_chromadb():
    try:
        return ChromaDBManager()
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        return None