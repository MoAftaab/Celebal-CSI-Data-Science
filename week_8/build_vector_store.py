import os
import sys
import shutil
import logging

# Add the project root to the Python path to allow imports from `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval import DataRetriever
from langchain_chroma import Chroma

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Correctly navigate up one level from `week_8` to the project root, then into `week_8`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(project_root, "week_8", "data", "raw", "Training Dataset.csv")
VECTOR_STORE_PATH = os.path.join(project_root, "week_8", "models", "vector_store")
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def main():
    """
    Builds and saves the ChromaDB vector store locally.
    This script should be run from the project's root directory.
    """
    logging.info("Starting vector store build process...")

    # Ensure the data file exists
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data file not found at: {DATA_PATH}")
        logging.error("Please ensure the dataset is available before running this script.")
        return

    # 1. Initialize the DataRetriever
    retriever = DataRetriever(embedding_model_name=EMBEDDING_MODEL, vector_store_path=VECTOR_STORE_PATH)

    # 2. Load and split documents
    logging.info(f"Loading documents from {DATA_PATH}...")
    documents = retriever._load_and_split_documents(DATA_PATH)
    if not documents:
        logging.error("No documents were loaded. Aborting.")
        return
    logging.info(f"Loaded and split {len(documents)} document chunks.")

    # 3. Create the vector store
    logging.info(f"Creating new vector store at {VECTOR_STORE_PATH}...")

    # Remove old store if it exists
    if os.path.exists(VECTOR_STORE_PATH):
        logging.warning(f"Removing existing vector store at {VECTOR_STORE_PATH}")
        shutil.rmtree(VECTOR_STORE_PATH)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    # Create and persist the new vector store
    try:
        Chroma.from_documents(
            documents=documents,
            embedding=retriever.embeddings_model,
            persist_directory=VECTOR_STORE_PATH
        )
        logging.info("Successfully created and saved the vector store.")
        logging.info(f"Vector store is ready at: {VECTOR_STORE_PATH}")
    except Exception as e:
        logging.error(f"Failed to create vector store: {e}", exc_info=True)

if __name__ == "__main__":
    main() 