import os
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from .retrieval import DataRetriever
from .generator import Generator
from .utils import format_response, load_dataset, preprocess_data
import time

class RAGPipeline:
    """
    Class that integrates the retrieval and generation components for the RAG Q&A chatbot
    """
    def __init__(self, 
                 data_path: str,
                 vector_store_path: Optional[str] = None,
                 model_type: str = "openai",
                 model_name: str = "gpt-3.5-turbo-instruct",
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 temperature: float = 0.3,
                 max_tokens: int = 1024):
        """
        Initialize the RAG pipeline
        
        Args:
            data_path: Path to the data file
            vector_store_path: Path to store the vector database
            model_type: Type of model to use
            model_name: Name of the model to use
            embedding_model_name: Name of the embedding model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
        """
        # Set up logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.data_path = data_path
        self.vector_store_path = vector_store_path
        
        # Load and process data
        self.df = load_dataset(data_path)
        self.processed_df = preprocess_data(self.df)
        
        # Initialize retriever and generator
        self.retriever = DataRetriever(
            embedding_model_name=embedding_model_name,
            persist_directory=vector_store_path
        )
        
        self.generator = Generator(
            model_type=model_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self) -> None:
        """
        Initialize the knowledge base from the dataset
        """
        try:
            self.logger.info("Initializing knowledge base...")
            start_time = time.time()
            # Try with a retry mechanism to handle file access errors
            max_retries = 3
            retry_delay = 2  # seconds
            for attempt in range(max_retries):
                try:
                    self.retriever.create_knowledge_base(self.processed_df)
                    end_time = time.time()
                    self.logger.info(f"Knowledge base created in {end_time - start_time:.2f} seconds")
                    # Check that the vector store and retriever are valid
                    if not self.retriever.vector_store or not self.retriever.retriever:
                        raise RuntimeError("Knowledge base was not created successfully. Vector store or retriever is None.")
                    return
                except Exception as e:
                    if "process cannot access the file" in str(e) and attempt < max_retries - 1:
                        self.logger.warning(f"File access error (attempt {attempt+1}/{max_retries}): {e}")
                        self.logger.info(f"Waiting {retry_delay} seconds before retrying...")
                        time.sleep(retry_delay)
                    else:
                        raise
        except Exception as e:
            self.logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Check that the knowledge base is ready
            if not self.retriever or not self.retriever.vector_store or not self.retriever.retriever:
                raise RuntimeError("Knowledge base has not been created. Call create_knowledge_base first.")
            # Retrieve relevant documents
            self.logger.info(f"Processing query: {query}")
            start_time = time.time()
            documents, scores = self.retriever.retrieve(query, top_k=top_k)
            retrieval_time = time.time() - start_time
            self.logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.2f} seconds")
            
            # Generate response based on retrieved documents
            generation_start = time.time()
            response, source_info = self.generator.generate_response(query, documents)
            generation_time = time.time() - generation_start
            self.logger.info(f"Generated response in {generation_time:.2f} seconds")
            
            # Format response for display
            formatted_response = format_response(response, source_info)
            
            # Return the result
            return {
                "query": query,
                "raw_response": response,
                "formatted_response": formatted_response,
                "sources": source_info.get("sources", []),
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": retrieval_time + generation_time,
                "retrieved_docs": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents],
                "similarity_scores": scores
            }
        
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "formatted_response": f"Sorry, I encountered an error while processing your query: {str(e)}"
            } 