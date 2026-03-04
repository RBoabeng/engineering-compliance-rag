import os
import yaml
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 1. Load Configuration

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully.")


# 2. Setup Global Settings

print(f"Loading Hugging Face embedding model: {config['embedding']['model_name']}...")
Settings.embed_model = HuggingFaceEmbedding(model_name=config['embedding']['model_name'])
Settings.llm = None  # LLM is not needed for ingestion, saving RAM

# Apply chunking strategies from config
Settings.chunk_size = config['embedding']['chunk_size']
Settings.chunk_overlap = config['embedding']['chunk_overlap']


# 3. Metadata Extraction Logic

def extract_metadata(file_path):
    """
    Tags the document chunk with the engineering domain based on its subfolder.
    """
    parts = file_path.split(os.sep)
    domain = parts[-2] if len(parts) >= 2 else "general"
    
    return {
        "domain": domain,
        "file_name": os.path.basename(file_path)
    }


# 4. Load and Parse Documents

print(f"Reading PDFs from the {config['data']['input_dir']} directory...")
reader = SimpleDirectoryReader(
    input_dir=config['data']['input_dir'],
    recursive=True,
    file_metadata=extract_metadata,
    required_exts=[".pdf"]
)
documents = reader.load_data()
print(f"Successfully loaded {len(documents)} document chunks/pages.")


# 5. Initialize Vector Database

print(f"Initializing ChromaDB at {config['data']['persist_dir']}...")
db = chromadb.PersistentClient(path=config['data']['persist_dir'])
chroma_collection = db.get_or_create_collection(config['data']['collection_name'])

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# 6. Embed and Index

print("Generating embeddings and indexing... (This may take a few minutes)")
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    show_progress=True
)

print("Ingestion complete! Your local vector database is ready.")