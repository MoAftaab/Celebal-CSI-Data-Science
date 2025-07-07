import os
import logging
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rag_pipeline():
    """
    Test the RAG pipeline
    """
    # Load environment variables
    load_dotenv()
    
    # Set constants
    DATA_PATH = os.path.join("data", "raw", "Training Dataset.csv")
    VECTOR_STORE_PATH = os.path.join("models", "vector_store")
    MODEL_TYPE = "gemini"
    MODEL_NAME = "models/gemini-1.5-flash"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Set Gemini API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBpTRDaiUWFEYSU0OqePxyoCP01uaoC01c")
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    
    try:
        logger.info("Initializing RAG pipeline...")
        
        # Initialize RAG pipeline
        pipeline = RAGPipeline(
            data_path=DATA_PATH,
            vector_store_path=VECTOR_STORE_PATH,
            model_type=MODEL_TYPE,
            model_name=MODEL_NAME,
            embedding_model_name=EMBEDDING_MODEL
        )
        
        logger.info("RAG pipeline initialized successfully!")
        
        # Test with a query
        query = "What is the overall loan approval rate?"
        logger.info(f"Processing query: {query}")
        
        result = pipeline.process_query(query)
        
        logger.info("Result:")
        logger.info(f"Raw response: {result['raw_response']}")
        logger.info(f"Retrieval time: {result['retrieval_time']:.2f} seconds")
        logger.info(f"Generation time: {result['generation_time']:.2f} seconds")
        logger.info(f"Total time: {result['total_time']:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing RAG pipeline: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG pipeline...")
    success = test_rag_pipeline()
    
    if success:
        print("✅ RAG pipeline test successful!")
    else:
        print("❌ RAG pipeline test failed.") 