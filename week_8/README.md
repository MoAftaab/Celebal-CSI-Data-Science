# 🤖 Loan Approval RAG Q&A Chatbot

This project implements a Retrieval-Augmented Generation (RAG) Q&A chatbot for analyzing and querying loan approval data.

## 📖 Overview

The Loan Approval RAG Q&A Chatbot combines:

- **Document Retrieval**: Finds relevant information from the loan dataset
- **Generative AI**: Produces natural language responses based on retrieved data
- **Interactive Interface**: User-friendly Streamlit web application

This enables users to ask natural language questions about loan approval data and receive accurate, data-driven responses.

## 🌟 Key Features

- Natural language querying of loan approval dataset
- Data-driven responses with source citations
- Interactive data exploration and visualizations
- Multiple language model options (OpenAI, Hugging Face, local models)
- Persistent vector store for efficient retrieval

## 🏗️ Architecture

The system follows a classic RAG architecture:

1. **Indexing Phase**:
   - Dataset is loaded and preprocessed
   - Text is chunked and embedded using sentence transformers
   - Vector embeddings are stored in ChromaDB for semantic retrieval

2. **Query Phase**:
   - User query is embedded using the same embedding model
   - Similar documents are retrieved from the vector store
   - Retrieved content is used as context for the language model
   - LLM generates a response based on the context and query

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Pip package manager

### Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd week_8
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Get the dataset from [Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)
   - Place `Training Dataset.csv` in the `data/raw` directory

4. Set up environment variables:
   - Create a `.env` file in the project root directory
   - Add API keys (choose at least one):
     ```
     OPENAI_API_KEY=your_openai_api_key
     HUGGINGFACE_API_KEY=your_huggingface_api_key
     ```

## 🚀 Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your browser at the provided URL (typically http://localhost:8501)

3. Use the chatbot by:
   - Typing questions in the chat input
   - Exploring dataset visualizations in the Dataset Explorer tab
   - Reading about the project in the About tab

## 💬 Sample Questions

- "What is the overall loan approval rate?"
- "How does gender affect loan approval?"
- "Is there a relationship between credit history and loan approval?"
- "What income level has the highest approval rate?"
- "Do married applicants have better chances of loan approval?"
- "What factors most strongly influence loan approval decisions?"

## 📊 Dataset

The chatbot uses the [Loan Approval Prediction dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction) from Kaggle, which contains information about loan applications including:

- Applicant demographics (gender, marital status, dependents)
- Financial information (income, loan amount, credit history)
- Property details (area, type)
- Loan approval status (target variable)

## 🧩 Project Structure

```
week_8/
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
├── data/
│   ├── raw/               # Raw dataset files
│   └── processed/         # Processed data files
├── models/
│   └── vector_store/      # Persistent vector store files
└── src/
    ├── __init__.py        # Package initialization
    ├── retrieval.py       # Document retrieval functionality
    ├── generator.py       # Text generation functionality
    ├── rag_pipeline.py    # Integration of retrieval and generation
    └── utils.py           # Utility functions
```

## 🔄 Advanced Configuration

You can modify the following constants in `app.py` to change the behavior:

- `DEFAULT_MODEL_TYPE`: Change to "huggingface" or "local" if preferred
- `DEFAULT_MODEL_NAME`: Change to appropriate model name based on type
- `DEFAULT_EMBEDDING_MODEL`: Change the sentence transformer model for embeddings

## 📜 License

This project is released under the MIT License.

## 🙏 Acknowledgments

- This project uses the [LangChain](https://github.com/langchain-ai/langchain) framework
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction) 