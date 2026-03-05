import streamlit as st
import yaml
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


# 1. Page Configuration

st.set_page_config(page_title="Engineering Compliance AI", page_icon="🏗️", layout="wide")
st.title("Engineering Compliance Assistant")
st.markdown("Ask questions about engineering codes and get hallucination-free, cited answers.")


# 2. Load Pipeline (Cached for Speed)

@st.cache_resource(show_spinner="Loading AI Models and Database...")
def initialize_rag():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Initialize Models (with our memory fix!)
    Settings.embed_model = HuggingFaceEmbedding(model_name=config['embedding']['model_name'])
    Settings.llm = Ollama(
        model=config['llm']['model_name'], 
        request_timeout=120.0,
        context_window=4096,
        additional_kwargs={"num_ctx": 4096} 
    )

    # Connect to Database
    db = chromadb.PersistentClient(path=config['data']['persist_dir'])
    chroma_collection = db.get_or_create_collection(config['data']['collection_name'])
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

index = initialize_rag()

# 3. Sidebar: Metadata Filtering

with st.sidebar:
    st.header("Search Filters")
    st.write("Limit the AI's search to a specific engineering domain.")
    # You can add "highway", "water", etc. to this list as you add more PDFs!
    selected_domain = st.selectbox("Select Domain:", ["All Domains", "structural", "water", "highway"])
    
    st.markdown("---")
    st.markdown("**Tech Stack:**\n* LlamaIndex\n* ChromaDB\n* Ollama (Llama 3.2)\n* Streamlit")


# 4. Chat Interface

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a compliance question... (e.g., 'What are the flood opening requirements?')"):
    # Add user message to chat state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Searching manuals and generating answer..."):
            
            # Apply Metadata Filters if a specific domain is chosen
            filters = None
            if selected_domain != "All Domains":
                filters = MetadataFilters(
                    filters=[ExactMatchFilter(key="domain", value=selected_domain)]
                )
            
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                filters=filters
            )
            
            response = query_engine.query(prompt)
            
            # Format the output with citations
            answer_text = response.response + "\n\n**📚 Sources:**\n"
            for node in response.source_nodes:
                file_name = node.metadata.get('file_name', 'Unknown')
                domain = node.metadata.get('domain', 'Unknown')
                answer_text += f"* [{domain.upper()}] `{file_name}`\n"
            
            st.markdown(answer_text)
            
    # Add AI response to chat state
    st.session_state.messages.append({"role": "assistant", "content": answer_text})