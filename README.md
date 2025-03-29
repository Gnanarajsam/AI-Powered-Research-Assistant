
# AI-Powered Research Assistant

## Description
The AI-Powered Research Assistant is a Streamlit-based application that utilizes LangChain and Google Generative AI to answer domain-specific queries. It provides tools for document retrieval, summarization, and text generation, allowing users to interact with PDF documents efficiently.

## Features

● PDF Processing: Extracts text from PDFs using PyPDF2 and performs OCR on image-based documents.

● Text Chunking: Splits large text into manageable chunks for processing.

● Vector Storage: Stores text embeddings in a FAISS vector database for fast retrieval.

● Conversational AI: Uses LangChain and Google Generative AI to provide accurate answers.

● General Knowledge Queries: Responds to general questions when document-specific information is unavailable.

● Suggested Questions: Generates related questions based on retrieved content.

## Installation

### Prerequisites

● Python 3.10.0

● A Google API key (set in .env as GOOGLE_API_KEY)

### Install Dependencies

Run the following command to install the required dependencies:
```http
  pip install -r requirements.txt
```
## Usage
#### 1. Run the Application
```http
  streamlit run app.py
```
#### 2. Upload PDF Files

● Use the sidebar to upload multiple PDFs.

● Click "Process" to extract and index the text.

#### 3. Ask Questions

● Enter your query in the text input field.

● The system will retrieve relevant information and provide an answer.

## Project Structure

```http
├── app.py                  # Main application script
├── requirements.txt        # Dependencies
├── .env                    # API keys and environment variables
├── README.md               # Project documentation
└── faiss_index/            # Directory for storing FAISS index
```
## Conclusion

The AI-Powered Research Assistant is a robust tool designed to enhance research efficiency by leveraging AI-driven document retrieval, summarization, and text generation. By integrating LangChain and Google Generative AI, this application enables users to interact with PDFs seamlessly. Future enhancements may include support for additional document formats, improved AI-driven insights, and expanded customization options to better serve domain-specific research needs.







