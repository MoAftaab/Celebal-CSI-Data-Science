import sys
import sqlite3
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import time
import logging
from pathlib import Path
from src.rag_pipeline import RAGPipeline
from src.utils import load_dataset, preprocess_data, create_visualizations
import shutil

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Robust Path Configuration ---
# Get the absolute path of the directory where this script is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants
DATA_PATH = os.path.join(APP_DIR, "data", "raw", "Training Dataset.csv")
VECTOR_STORE_PATH = os.path.join(APP_DIR, "models", "vector_store")
CACHE_FOLDER = os.path.join(APP_DIR, "models", "huggingface_cache")
DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE", "gemini")  # Use Gemini as default
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")  # Use the working model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Set Gemini API key directly if available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBpTRDaiUWFEYSU0OqePxyoCP01uaoC01c")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs(os.path.join("models", "vector_store"), exist_ok=True)
    
    # Create visualizations directory
    os.makedirs(os.path.join("visualizations"), exist_ok=True)

def initialize_rag_pipeline():
    """Initialize the RAG pipeline"""
    try:
        # Get API keys from environment variables
        gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBpTRDaiUWFEYSU0OqePxyoCP01uaoC01c")
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        
        # Load the dataset
        df = load_dataset(DATA_PATH)
        
        # Create logger instance
        logger = logging.getLogger(__name__)
        
        # Initialize the RAG pipeline
        rag_pipeline = RAGPipeline(
            data_path=DATA_PATH,
            model_type="gemini",  # Always use Gemini
            model_name="models/gemini-2.5-flash",  # Use the most reliable model
            vector_store_path=VECTOR_STORE_PATH,
            embedding_model_name=DEFAULT_EMBEDDING_MODEL,
            cache_folder=CACHE_FOLDER,
            temperature=0.3,
            max_tokens=1024
        )
        
        return rag_pipeline
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        return None

def display_data_visualization(df):
    """Display data visualizations"""
    st.subheader("📊 Data Visualizations")
    
    # Create visualizations
    visualizations = create_visualizations(df, save_dir="visualizations")
    
    # Create columns for visualizations
    col1, col2 = st.columns(2)
    
    # Display PNG visualizations if available
    with col1:
        if "loan_status_pie" in visualizations:
            st.image(visualizations["loan_status_pie"], caption="Loan Status Distribution", use_column_width=True)
        if "property_area_loan_status" in visualizations:
            st.image(visualizations["property_area_loan_status"], caption="Property Area vs Loan Status", use_column_width=True)
    
    with col2:
        if "income_by_loan_status" in visualizations:
            st.image(visualizations["income_by_loan_status"], caption="Income by Loan Status", use_column_width=True)
        if "education_loan_status" in visualizations:
            st.image(visualizations["education_loan_status"], caption="Education vs Loan Status", use_column_width=True)
    
    if "credit_history_vs_loan_status" in visualizations:
        st.image(visualizations["credit_history_vs_loan_status"], caption="Credit History vs Loan Status", use_column_width=True)

def display_data_overview(df):
    """Display an overview of the dataset"""
    st.subheader("📋 Dataset Overview")
    
    # Show basic dataset statistics with metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Records", value=f"{len(df):,}")
    with col2:
        st.metric(label="Features", value=len(df.columns))
    with col3:
        if 'Loan_Status' in df.columns:
            try:
                if df['Loan_Status'].dtype == bool:
                    approval_rate = df['Loan_Status'].mean() * 100
                else:
                    approval_rate = (df['Loan_Status'] == 'Y').mean() * 100
                st.metric(label="Approval Rate", value=f"{approval_rate:.1f}%")
            except Exception as e:
                st.error(f"Error calculating approval rate: {e}")
    
    # Show a sample of the dataset
    with st.expander("📄 View Sample Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Show column information - convert to string types to avoid Arrow conversion issues
    with st.expander("🔍 View Column Information", expanded=False):
        try:
            col_info = pd.DataFrame({
                'Column': list(df.columns),
                'Type': [str(dtype) for dtype in df.dtypes.values],
                'Non-Null Count': df.count().values,
                'Missing Values': df.isna().sum().values,
                'Missing (%)': (df.isna().sum() / len(df) * 100).values.round(2)
            })
            st.dataframe(col_info, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying column information: {e}")
            st.write("Basic column list:")
            st.write(", ".join(df.columns))

    # Show basic statistics - handle non-numeric columns
    with st.expander("📊 View Statistics", expanded=False):
        try:
            st.write("Numerical Features:")
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numerical features found in the dataset.")
        except Exception as e:
            st.error(f"Error displaying statistics: {e}")

def display_chat_interface():
    """Display the chat interface"""
    st.header("💬 Loan Approval RAG Chatbot")
    
    # Add descriptive text
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:15px; border-radius:5px; margin-bottom:20px">
    Ask questions about the loan approval dataset to get data-driven insights. The chatbot uses 
    <span style="color:#ff4b4b; font-weight:bold">Retrieval Augmented Generation (RAG)</span> 
    to provide accurate, context-aware answers based on the loan approval dataset.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for holding the current query
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    def set_example_query(query):
        st.session_state.current_query = query
    
    # Sample questions button row - grid layout
    st.markdown("##### Try asking:")
    
    # Create a 3x2 grid for sample questions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("📊 What is the loan approval rate for applicants with a good credit history?", 
                 on_click=set_example_query, 
                 args=["What is the loan approval rate for applicants with a good credit history?"],
                 key="btn_approval_rate")
        
        st.button("💳 How does applicant income relate to loan approval?", 
                 on_click=set_example_query, 
                 args=["How does applicant income relate to loan approval?"],
                 key="btn_credit_history")
    
    with col2:
        st.button("👨‍👩‍👧 Does being married improve loan approval chances?", 
                 on_click=set_example_query, 
                 args=["Does being married improve loan approval chances?"],
                 key="btn_marital")
        
        st.button("💰 What is the average loan amount for approved applications?", 
                 on_click=set_example_query, 
                 args=["What is the average loan amount for approved applications?"],
                 key="btn_income")
    
    with col3:
        st.button("🏦 Are there differences in approval rates between urban and rural areas?", 
                 on_click=set_example_query, 
                 args=["Are there differences in approval rates between urban and rural areas?"],
                 key="btn_factors")
        
        st.button("👫 How do dependents affect loan approval?", 
                 on_click=set_example_query, 
                 args=["How do dependents affect loan approval?"],
                 key="btn_gender")
    
    # Get user input - make the chat input bigger but use a different approach for prefilling
    # Always display the chat input. The query from the button click will be handled in the main loop.
    user_query = st.chat_input("Ask a question about the loan approval dataset...",
                               key="chat_input",
                               max_chars=1000)
    
    return user_query

def custom_css():
    """Add custom CSS to the app"""
    st.markdown("""
    <style>
    /* General styling */
    .main {
        background-color: #ffffff;
    }
    
    /* Custom button styling */
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #343a40;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton button:hover {
        background-color: #FF4B4B;
        color: white;
        border-color: #FF4B4B;
    }
    
    /* Chat container styling */
    [data-testid="stChatMessage"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* User message styling */
    [data-testid="stChatMessage"][data-testid="userChatMessage"] {
        background-color: #e6f7ff;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"]:not([data-testid="userChatMessage"]) {
        background-color: #f0f0f0;
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #FF4B4B;
    }
    
    /* Container styling */
    [data-testid="stContainer"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    
    /* Make the chat container more prominent */
    .conversation-container {
        border: 2px solid #dee2e6 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin-top: 1rem !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #343a40;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #FF4B4B transparent transparent transparent;
    }
    </style>
    """, unsafe_allow_html=True)

def update_sidebar():
    """Set up the sidebar"""
    with st.sidebar:
        st.title("Loan Approval RAG Chatbot")
        st.image("https://img.icons8.com/fluent/96/000000/chatbot.png", width=100)
        
        # Add separator
        st.markdown("---")
        
        # Model settings expander - simplified to only use Gemini
        with st.expander("🤖 Model Settings", expanded=False):
            st.success("Using Google's Gemini-2.5-Flash model")
            st.info("This model provides high-quality responses with fast processing time.")
            
            # Show model parameters
            st.markdown("#### Model Parameters")
            st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1, disabled=True,
                     help="Controls randomness in responses. Lower values are more deterministic.")
            st.slider("Max Tokens", min_value=256, max_value=2048, value=1024, step=128, disabled=True,
                     help="Maximum length of generated responses.")
        
        # Add information about the application
        st.markdown("### About")
        st.markdown("""
        This chatbot uses **Retrieval Augmented Generation (RAG)** to provide accurate answers about the loan approval dataset.
        
        The system:
        1. Retrieves relevant information from the dataset
        2. Processes your query using Gemini 2.5 Flash
        3. Generates an accurate, context-aware response
        """)
        
        # Add credits
        st.markdown("---")
        st.markdown("Made with ❤️ by GenAI Dev")

def handle_meta_question(query: str) -> str:
    """Check if the query is a meta-question and return a canned response."""
    query_lower = query.lower()
    meta_keywords = [
        "what can you do", "help", "capabilities", "what kind of questions", 
        "what types of questions", "what questions can you answer"
    ]
    
    if any(keyword in query_lower for keyword in meta_keywords):
        return """
        I am an AI assistant designed to help you analyze the **Loan Approval dataset**. You can ask me questions about the data to uncover insights.

        Here are the types of questions I can handle:

        *   **Statistical Questions:**
            *   "What is the overall loan approval rate?"
            *   "What is the average applicant income?"

        *   **Correlation Questions:**
            *   "How does credit history affect loan approval?"
            *   "Is there a relationship between education level and getting a loan?"

        *   **Comparison Questions:**
            *   "Compare the approval rates for urban vs. rural properties."
            *   "What is the difference in average income between approved and denied applicants?"

        Feel free to try one of the example questions or ask your own!
        """
    return None

def main():
    # Custom CSS
    custom_css()
    
    # Page configuration
    st.set_page_config(
        page_title="Loan Approval RAG Chatbot",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create directories
    create_directories()
    
    # Set up sidebar
    update_sidebar()
    
    # Application tabs - make chatbot the main focus
    tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🔍 Dataset Explorer", "ℹ️ About"])
    
    # Tab 1: Chatbot
    with tab1:
        # Initialize session state for chat history if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize RAG pipeline
        if "rag_pipeline" not in st.session_state:
            with st.spinner("Initializing the RAG pipeline... This may take a few moments."):
                st.session_state.rag_pipeline = initialize_rag_pipeline()
        
        # Display chat interface and get user query
        user_query_from_input = display_chat_interface()

        # Check if a button was clicked
        if st.session_state.get("current_query"):
            user_query = st.session_state.current_query
            st.session_state.current_query = ""  # Clear after use
        else:
            user_query = user_query_from_input
                
        # Check if pipeline initialized successfully
        if st.session_state.rag_pipeline is None:
            st.error("Failed to initialize the RAG pipeline. Please check the logs for more details.")
            return
        
        # Display chat history in a larger, more prominent container
        st.markdown("### Conversation History")
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        chat_container = st.container(height=500, border=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process the query if there is one
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate and display bot response
            with st.chat_message("assistant"):
                try:
                    # Check for meta-questions first
                    meta_response = handle_meta_question(user_query)
                    
                    if meta_response:
                        st.markdown(meta_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": meta_response})
                    else:
                        with st.spinner("Analyzing the dataset to find an answer..."):
                            response_data = st.session_state.rag_pipeline.process_query(user_query)
                            formatted_response = response_data.get("formatted_response", "Sorry, I couldn't process that.")
                            st.markdown(formatted_response)
                            st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        # Now display all chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # Display metrics if available (outside the chat container)
        if user_query and "last_result" in st.session_state:
            result = st.session_state.last_result
            with st.expander("🔍 Response Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retrieval Time", f"{result.get('retrieval_time', 0):.3f}s")
                with col2:
                    st.metric("Generation Time", f"{result.get('generation_time', 0):.3f}s")
                with col3:
                    st.metric("Total Time", f"{result.get('total_time', 0):.3f}s")
                
                # Display sources if available
                if result.get("sources"):
                    st.write("📚 Sources:")
                    for source in result["sources"]:
                        st.write(f"- {source}")
    
    # Tab 2: Dataset Explorer
    with tab2:
        st.header("🔍 Dataset Explorer")
        st.write("Explore and understand the loan approval dataset.")
        
        # Load and display dataset
        try:
            df = load_dataset(DATA_PATH)
            
            # Display data overview
            display_data_overview(df)
            
            # Display visualizations
            display_data_visualization(df)
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Tab 3: About
    with tab3:
        st.header("ℹ️ About")
        st.write("### Loan Approval RAG Chatbot")
        
        # Application description
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; margin-bottom:20px">
        <h4>What is RAG?</h4>
        <p>
        <b>Retrieval Augmented Generation (RAG)</b> combines the power of large language models with 
        the ability to retrieve specific information from a knowledge base. This provides more accurate, 
        contextual, and verifiable responses based on your data.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Architecture diagram with columns
        st.subheader("🏗️ RAG Architecture")
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("""
            **How it works:**
            1. **Indexing**: Data is processed and stored in a vector database
            2. **Retrieval**: When you ask a question, relevant data is retrieved
            3. **Generation**: An LLM uses the retrieved context to answer your question
            """)
        
        with col2:
            # Mermaid diagram removed to simplify the UI.
            pass
        
        # Features and technologies
        st.subheader("🌟 Key Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **Features:**
            - Natural language querying of loan data
            - Data-driven responses with sources
            - Interactive data exploration
            - Beautiful visualizations
            """)
            
        with feature_col2:
            st.markdown("""
            **Technologies:**
            - **Retrieval**: Sentence Transformers, LangChain, ChromaDB
            - **Generation**: SimpleLLM with fallback to Google Gemini Pro
            - **Frontend**: Streamlit with custom UI
            - **Data Processing**: Pandas, NumPy
            """)
            
        # Dataset information
        st.subheader("📊 Dataset")
        st.markdown("""
        The application uses the [Loan Approval Prediction dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction) from Kaggle.
        
        This dataset contains information about loan applications including:
        - Applicant demographics (gender, marital status, dependents)
        - Financial information (income, loan amount, credit history)
        - Property details (area, type)
        - Loan approval status (target variable)
        """)
        
        st.markdown("---")
        st.markdown("### Sample Questions to Ask")
        
        # Sample questions in an attractive grid
        questions = [
            "What is the overall loan approval rate?",
            "How does gender affect loan approval?",
            "Is there a relationship between credit history and loan approval?",
            "What income level has the highest approval rate?",
            "Do married applicants have better chances of loan approval?",
            "What factors most strongly influence loan approval decisions?",
            "How does education level affect loan approval?",
            "Are self-employed applicants less likely to be approved?",
            "What's the average loan amount for approved applications?",
            "How does property area affect loan approval chances?"
        ]
        
        question_cols = st.columns(2)
        for i, question in enumerate(questions):
            col = question_cols[i % 2]
            with col:
                st.markdown(f"- {question}")

if __name__ == "__main__":
    main() 