import os
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
import chromadb
from langchain_community.vectorstores import Chroma
import logging
import sys

class DataRetriever:
    """
    Class for retrieving relevant information from the dataset based on user queries
    """
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 persist_directory: Optional[str] = None):
        """
        Initialize the retriever with an embedding model
        
        Args:
            embedding_model_name: The name of the sentence transformer model to use
            persist_directory: Directory to persist the vector store (optional)
        """
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name
            )
            self.logger.info(f"Initialized embeddings with model: {embedding_model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing embeddings: {e}")
            raise
            
        self.vector_store = None
        self.retriever = None
        
    def _convert_df_to_documents(self, df: pd.DataFrame, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[Document]:
        """
        Convert a pandas DataFrame to a list of Document objects
        
        Args:
            df: Input DataFrame
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of Document objects
        """
        # Create a string representation of the dataset overview
        dataset_info = f"Dataset Overview:\nThis dataset contains {len(df)} records with {len(df.columns)} columns.\n"
        dataset_info += f"Column names: {', '.join(df.columns.tolist())}\n\n"
        
        # Add column descriptions
        dataset_info += "Column Descriptions:\n"
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique()
            missing = df[col].isnull().sum()
            dataset_info += f"- {col}: Type={dtype}, Unique Values={unique_vals}, Missing Values={missing}\n"
        
        # Add sample records
        dataset_info += "\nSample Records:\n"
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            dataset_info += f"Record {i+1}:\n"
            for col, val in row.items():
                dataset_info += f"  {col}: {val}\n"
            dataset_info += "\n"
        
        # Create documents from all records
        documents = []
        
        # Add the dataset overview as a document
        documents.append(Document(
            page_content=dataset_info,
            metadata={"source": "dataset_overview"}
        ))
        
        # Create documents for each record
        for idx, row in df.iterrows():
            record_text = f"Record ID: {idx}\n"
            metadata = {"record_id": idx}
            
            for col, val in row.items():
                record_text += f"{col}: {val}\n"
                metadata[col] = str(val)
            
            documents.append(Document(
                page_content=record_text,
                metadata=metadata
            ))
        
        # Add statistical information
        stats_text = "Dataset Statistics:\n"
        
        # Add numerical stats if available
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            stats = df[num_cols].describe()
            stats_text += f"Numerical Statistics:\n{stats.to_string()}\n\n"
        
        # Add categorical stats if available
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            stats_text += "Categorical Statistics:\n"
            for col in cat_cols:
                stats_text += f"{col} value counts:\n{df[col].value_counts().to_string()}\n\n"
        
        documents.append(Document(
            page_content=stats_text,
            metadata={"source": "dataset_statistics"}
        ))
        
        # Split documents into chunks if they're too large
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunked_documents = []
        for doc in documents:
            splits = text_splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                metadata = doc.metadata.copy()
                metadata["chunk_id"] = i
                chunked_documents.append(Document(page_content=split, metadata=metadata))
        
        return chunked_documents

    def create_knowledge_base(self, df: pd.DataFrame, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        """
        Create or load the knowledge base from the DataFrame.
        Args:
            df: Input DataFrame
            chunk_size: Size of chunks for text splitting
            chunk_overlap: Overlap between chunks
        """
        import shutil
        import time

        force_rebuild = False

        # 1. Try to load the existing vector store
        if self.persist_directory and os.path.exists(self.persist_directory):
            try:
                self.logger.info(f"Loading existing vector store from {self.persist_directory}")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                # Test if the vector store works and is not empty
                test_results = self.vector_store.similarity_search("test", k=1)
                if not test_results:
                    self.logger.warning("Existing vector store is empty. Forcing rebuild.")
                    force_rebuild = True
                else:
                    self.logger.info("Successfully loaded existing vector store.")
            except Exception as e:
                self.logger.warning(f"Failed to load vector store from {self.persist_directory}: {e}. Forcing rebuild.")
                force_rebuild = True
        else:
            self.logger.info("No existing vector store found. Building a new one.")
            force_rebuild = True

        # 2. Rebuild the vector store if necessary
        if force_rebuild:
            # Convert DataFrame to documents
            documents = self._convert_df_to_documents(df, chunk_size, chunk_overlap)
            self.logger.info(f"Creating new vector store with {len(documents)} documents.")

            # Clean up the old directory if it exists
            if self.persist_directory and os.path.exists(self.persist_directory):
                self.logger.info(f"Removing existing vector store at {self.persist_directory}")
                try:
                    shutil.rmtree(self.persist_directory)
                    # Add a small delay to allow the OS to release file handles, especially on Windows
                    time.sleep(0.5)
                except Exception as e:
                    self.logger.error(f"Could not remove directory {self.persist_directory}: {e}. "
                                      f"Attempting to create vector store in memory instead.")
                    self.persist_directory = None # Fallback to in-memory

            # Create the new vector store
            try:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                if self.persist_directory:
                    self.logger.info(f"Persisting vector store to {self.persist_directory}")
                    self.vector_store.persist()
            except Exception as e:
                self.logger.error(f"Fatal error creating new vector store: {e}")
                # If creation fails, we cannot proceed.
                raise RuntimeError(f"Failed to create the vector store: {e}")

        # 3. Create the retriever
        if self.vector_store:
            self.logger.info("Creating retriever from vector store.")
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        else:
            # This should not be reached if the logic above is correct, but it's a safeguard.
            raise RuntimeError("Vector store is not available. Cannot create retriever.")

        # 4. Final verification
        if not self.retriever:
            raise RuntimeError("Failed to create the retriever.")
        
        self.logger.info("Knowledge base initialization completed successfully.")
        
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Document], List[float]]:
        """
        Retrieve relevant documents based on the query
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents and their similarity scores
        """
        if not self.vector_store:
            raise ValueError("Knowledge base has not been created. Call create_knowledge_base first.")
        
        try:
            retrieved_docs = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            documents = [doc for doc, _ in retrieved_docs]
            scores = [score for _, score in retrieved_docs]
            
            return documents, scores
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            # Return empty results rather than crashing
            return [], [] 