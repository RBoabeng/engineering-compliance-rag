# AI-Powered Engineering Compliance Assistant (RAG)

An enterprise-grade, 100% local Retrieval-Augmented Generation (RAG) pipeline designed for civil, environmental, and structural engineers.

This application ingests dense regulatory manuals (e.g., FEMA building codes, EPA water regulations) and allows users to ask natural language questions. To prevent AI "hallucinations," the system forces the LLM to pull answers exclusively from the ingested documents and provides exact file and domain citations for verification.

## The Business Problem & Solution

* **The Problem:** Engineers spend hundreds of hours manually parsing massive PDF rulebooks to ensure infrastructure designs comply with federal and state regulations. Standard LLMs (like ChatGPT) cannot be trusted for this due to hallucinations and lack of source verification. Furthermore, engineering blueprints and queries are often highly confidential and cannot be sent to cloud APIs.
* **The Solution:** A fully offline, privacy-preserving AI assistant. By utilizing a local vector database and local LLMs via Ollama, this project creates an "open-book exam" for the AI. It retrieves the exact regulatory text, synthesizes an answer, and cites its sources—all without leaving the user's local machine.

## Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** LlamaIndex (Data Framework)
* **Local LLM Server:** Ollama (running `Llama 3.2`)
* **Embeddings:** Hugging Face (`BAAI/bge-small-en-v1.5` for high-efficiency CPU processing)
* **Vector Database:** ChromaDB (Local Persistent Storage)
* **Frontend:** Streamlit

## Project Structure

```
engineering-compliance-rag/
│
├── data/                       # ⚠️ Create this folder! Place your PDFs here.
│   ├── structural/             # e.g., FEMA Coastal Construction Manual
│   ├── water/                  # e.g., EPA SWMM User's Manual
│   └── highway/                # e.g., MUTCD Guidelines
│
├── chroma_db/                  # Automatically generated local vector store
├── notebooks/                  # Jupyter notebooks for experimentation
│   └── 01_test_retrieval.ipynb 
│
├── app.py                      # The Streamlit web application
├── ingest.py                   # Data ingestion and chunking script
├── config.yaml                 # Centralized configuration panel
├── requirements.txt            # Python dependencies
└── README.md

```

# Installation & Setup

**1.Clone the repository and set up the environment**

```
git clone [https://github.com/YourUsername/engineering-compliance-rag.git](https://github.com/YourUsername/engineering-compliance-rag.git)
cd engineering-compliance-rag
python -m venv engineering_rag_env

# On Windows:
engineering_rag_env\Scripts\activate
# On Mac/Linux:
source engineering_rag_env/bin/activate

pip install -r requirements.txt
```

**2. Install Ollama and Download the LLM**

* Download and install Ollama.
* Pull the Llama 3.2 model to your local machine:

```
ollama pull llama3.2
```

**3.Add Your Engineering Data**
Because engineering manuals are massive, they are not included in this repository.

* Create a `data/` folder in the root directory.

* Inside `data/`, create subfolders for your domains (e.g., `water/`, `structural/`).

* Download relevant PDFs (like the EPA SWMM Manual) and place them in the corresponding subfolders.

# Running the Pipeline
**1.Ingest the Data**

Run the ingestion script to parse the PDFs, chunk the text, generate embeddings, and populate the local ChromaDB.

```
python ingest.py
```

**2.Launch  the Web Application**

Start the Streamlit UI to interact with your compliance assistant.

```
streamlit rung app.py
```

# Key Features

* **Metadata Routing**: Chunks are automatically tagged with their engineering domain based on folder structure, allowing users to filter searches (e.g., only search "Water" regulations).

* **RAM Optimized**: The context window is strictly managed in `config.yaml` to ensure the local LLM runs smoothly on standard consumer hardware (8GB-16GB RAM) without Out-Of-Memory errors.

Verifiable Citations: Every generated answer includes a `Sources` section, listing the specific domain and file name used to formulate the response.
