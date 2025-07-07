import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# No more load_dotenv() call

# Set up logging
logger = logging.getLogger(__name__)

class Generator:
    """
    Class for generating responses using Gemini language model
    """
    def __init__(self, 
                 model_type: str = "gemini",
                 model_name: str = "models/gemini-2.5-flash",
                 temperature: float = 0.3,
                 max_tokens: int = 1024):
        # Set up logging FIRST
        self.logger = logging.getLogger(__name__)
        """
        Initialize the generator with Gemini model
        
        Args:
            model_type: Type of model to use (always 'gemini', kept for backward compatibility)
            model_name: Name of the model to use (default: 'models/gemini-2.5-flash')
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
        """
        self.model_type = "gemini"  # Force model type to be gemini
        self.model_name = "models/gemini-2.5-flash"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = self._init_llm()
        
    def _init_llm(self):
        """
        Initialize the language model
        
        Returns:
            Initialized language model instance
        """
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBpTRDaiUWFEYSU0OqePxyoCP01uaoC01c")
        if not api_key:
            self.logger.warning("GEMINI_API_KEY not found in environment variables, using default key")
            
        # Configure Google Genai
        genai.configure(api_key=api_key)
        
        self.logger.info(f"Using langchain_google_genai integration with model {self.model_name}")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=api_key,
            convert_system_message_to_human=True,
        )

    def generate_response(self, query: str, retrieved_docs: List[Document], 
                          response_type: str = "detailed") -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response based on the query and retrieved documents
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents
            response_type: Type of response to generate (detailed or concise)
            
        Returns:
            Tuple of (response, source_info)
        """
        # Check if we have documents
        if not retrieved_docs:
            return "I couldn't find any relevant information to answer your question.", {"sources": []}
        
        # Prepare documents context
        docs_context = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs):
            # Get document content and metadata
            content = doc.page_content
            metadata = doc.metadata
            
            # Add document to context
            docs_context.append(f"Document {i+1}:\n{content}")
            
            # Extract source information if available
            source = metadata.get("source", f"Document {i+1}")
            if source not in sources:
                sources.append(source)
        
        # Combine documents into a single context
        context = "\n\n".join(docs_context)
        
        # Create a more robust prompt template for detailed answers with statistical insights
        template = """
        You are an expert Data Analyst AI assistant. Your primary function is to analyze the provided loan approval data context and extract meaningful, data-driven statistical insights.

        **Context from the loan dataset:**
        ---
        {context}
        ---

        **User's Question:**
        {query}

        **Instructions for your answer:**
        1.  **Direct Summary:** Begin with a concise, direct answer to the user's question.
        2.  **Detailed Analysis:** Provide a detailed breakdown of the findings. Use bullet points to structure the information.
        3.  **Key Statistical Insights:** This is the most important part. Identify and present key statistics. This includes, but is not limited to:
            *   **Counts and Frequencies:** (e.g., "Out of 100 applicants with a good credit history, 85 were approved.")
            *   **Percentages and Proportions:** (e.g., "This represents an 85% approval rate for this group.")
            *   **Averages and Medians:** (e.g., "The average loan amount for approved applications was $125,000.")
            *   **Comparisons:** (e.g., "Applicants with a college degree had a 15% higher approval rate than those without.")
        4.  **Explain the 'Why':** Don't just state the numbers. Explain what these statistics imply in the context of loan approvals. For example, "A higher approval rate for graduates suggests that education level is a positive factor in loan decisions."
        5.  **Clarity and Formatting:** Use bolding for key terms and statistics to make the response easy to read and understand.
        6.  **Data Limitations:** If the context doesn't have enough information, clearly state what's missing. For example, "The data does not contain information on applicant age, so I cannot analyze its impact."
        7.  **No External Knowledge:** Base your entire answer ONLY on the provided context. Do not invent data.

        **Detailed Answer with Statistical Insights:**
        """
        
        # Create the prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "query"]
        )
        
        # Generate the full prompt text
        full_prompt = prompt.format(context=context, query=query)
        
        try:
            # Generate the response
            self.logger.info(f"Generating response for query: {query}")
            response = self.llm.invoke(full_prompt).content
            
            # Return the response and source information
            return response, {"sources": sources}
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response. Please try again later.", {"sources": []} 