import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gemini_api():
    """
    Test if the Gemini API is working
    """
    # Load environment variables
    load_dotenv()
    
    # Get Gemini API key (with fallback to the one in the code)
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBpTRDaiUWFEYSU0OqePxyoCP01uaoC01c")
    
    if not api_key:
        logger.error("No Gemini API key found in environment variables")
        return False, None
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Get available models
        available_models = []
        try:
            print("Available models:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    model_name = m.name
                    print(f"- {model_name}")
                    available_models.append(model_name)
        except Exception as e:
            print(f"Could not list models: {e}")
        
        # Try models in this order: preferred models first, then any available models
        preferred_models = [
            "models/gemini-2.5-flash"
             # Try some lighter models if quota is an issue
        ]
        
        # Try preferred models first
        models_to_try = preferred_models.copy()
        
        # Add any other available models that support text generation
        for model in available_models:
            if model not in models_to_try:
                models_to_try.append(model)
        
        # Limit to first 5 models to avoid excessive API calls
        models_to_try = models_to_try[:5]
        
        for model_name in models_to_try:
            try:
                print(f"\nTrying with model: {model_name}")
                # Create a model
                model = genai.GenerativeModel(model_name)
                
                # Test with a simple prompt
                response = model.generate_content("Hello, give me a one-sentence description of RAG (Retrieval Augmented Generation).")
                
                # Check if response is valid
                if response and hasattr(response, 'text'):
                    logger.info(f"Gemini API test successful with model {model_name}!")
                    logger.info(f"Response: {response.text[:100]}...")  # Show first 100 chars
                    return True, model_name
                else:
                    logger.error(f"Gemini API returned an invalid response with model {model_name}")
            except Exception as e:
                logger.error(f"Error testing Gemini API with model {model_name}: {e}")
        
        # If all models failed
        return False, None
            
    except Exception as e:
        logger.error(f"Error testing Gemini API: {e}")
        return False, None

if __name__ == "__main__":
    print("Testing Gemini API...")
    success, working_model = test_gemini_api()
    
    if success:
        print(f"✅ Gemini API is working properly with model: {working_model}")
        print("Update your generator.py file to use this model name.")
    else:
        print("❌ Gemini API test failed with all model options. Please check your API key and internet connection.")
        print("Using SimpleLLM as fallback is recommended.") 