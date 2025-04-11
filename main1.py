import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from pdf2image import convert_from_bytes
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config("ChatPDF", page_icon="📚")
st.header("Chat with PDF Documents 📚")

def check_api_key():
    """Check if the Google API key is valid and configured"""
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY not found in environment variables or Streamlit secrets!")
        st.stop()
    try:
        genai.configure(api_key=google_api_key)
        # Test the configuration with a simple operation
        genai.list_models()
        return True
    except Exception as e:
        st.error(f"Failed to configure Google Generative AI: {e}")
        st.stop()

def get_pdf_text_with_ocr(pdf_docs):
    """Extract text from PDFs using both direct extraction and OCR"""
    text = ""
    if not pdf_docs:
        return text
        
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            # Extract text from text-based pages
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    st.warning(f"Text extraction failed for a page in {pdf.name}. Attempting OCR...")
                    # Perform OCR for image-based PDFs
                    images = convert_from_bytes(pdf.read())
                    for image in images:
                        ocr_text = pytesseract.image_to_string(image)
                        text += ocr_text
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """Split text into manageable chunks"""
    if not text:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a vector store from text chunks"""
    if not text_chunks:
        st.error("No text chunks generated from PDFs. Check if PDFs contain text.")
        return
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success(f"Indexed {len(text_chunks)} document chunks.")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return False

def get_conversational_chain():
    """Create a QA chain for document-based questions"""
    prompt_template = """
    Answer the question as precisely as possible using the provided context. 
    If the context doesn't contain the answer, say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_general_answer(user_question):
    """Generate an answer when no document context is available"""
    prompt_template = """
    You are an AI assistant knowledgeable about healthcare and AI.
    Provide a concise and accurate answer to the question.
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    chain = prompt | model
    response = chain.invoke({"question": user_question})
    return response.content

def suggest_related_questions(user_question, db):
    """Suggest related questions based on document context"""
    try:
        docs = db.similarity_search(user_question, k=3)
        if docs:
            context = "\n".join([d.page_content[:500] for d in docs])
            prompt_template = """
            Based on this context, suggest 3 related questions that the user might ask:
            
            Context: {context}
            
            Suggested questions:
            1. 
            2. 
            3. 
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
            chain = prompt | model
            response = chain.invoke({"context": context})
            st.subheader("You might also ask:")
            st.write(response.content)
    except Exception as e:
        st.warning(f"Couldn't generate related questions: {e}")

def user_input(user_question):
    """Handle user questions against the document store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Check if vector store exists
        if not os.path.exists("faiss_index"):
            st.info("No documents processed yet. Generating general answer...")
            response = generate_general_answer(user_question)
            st.write("Answer:", response)
            return
            
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if docs:
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question})
            st.write("Answer:", response["output_text"])
        else:
            st.info("No direct match found in documents. Generating general answer...")
            response = generate_general_answer(user_question)
            st.write("Answer:", response)
            
        suggest_related_questions(user_question, new_db)
        
    except Exception as e:
        st.error(f"Error processing your question: {e}")

def main():
    # Verify API key before proceeding
    check_api_key()
    
    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Processing")
        pdf_docs = st.file_uploader(
            "Upload PDF files", 
            accept_multiple_files=True,
            type=["pdf"],
            help="Upload one or more PDF documents to analyze"
        )
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text_with_ocr(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if get_vector_store(text_chunks):
                        st.success("Documents processed successfully!")
    
    # Main chat interface
    user_question = st.text_input(
        "Ask a question about your documents",
        placeholder="Type your question here...",
        key="user_question"
    )
    
    if user_question:
        with st.spinner("Finding the best answer..."):
            user_input(user_question)

if __name__ == "__main__":
    main()