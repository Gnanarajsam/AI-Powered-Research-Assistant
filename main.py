# import streamlit as st

# st.title("WORKCOHOL")
# st.write("AI-Powered Research Assistant")
# st.write("HELLO!")

# print(os.getenv("GOOGLE_API_KEY"))

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

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in environment variables!")
genai.configure(api_key=google_api_key)

def get_pdf_text_with_ocr(pdf_docs):
    text = ""
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks generated from PDFs. Check if PDFs contain text.")
        return
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success(f"Indexed {len(text_chunks)} document chunks.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question using the context below. If unsure, say you don't know.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_general_answer(user_question):
    prompt_template = """
    Answer based on general knowledge of healthcare AI:
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    chain = prompt | model
    response = chain.invoke({"question": user_question})
    return response.content

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if docs:
            chain = get_conversational_chain()
            response = chain.invoke({"input_documents": docs, "question": user_question})
            st.write("Answer:", response["output_text"])
        else:
            st.info("No direct match found. Generating general answer...")
            response = generate_general_answer(user_question)
            st.write("General Answer:", response)
            
        suggest_related_questions(user_question, new_db)
        
    except Exception as e:
        st.error(f"Error processing query: {e}")

def suggest_related_questions(user_question, db):
    docs = db.similarity_search(user_question, k=3)
    if docs:
        context = "\n".join([d.page_content[:500] for d in docs])
        prompt_template = """
        Based on this context, suggest 3 related questions:
        Context: {context}
        Questions:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
        chain = prompt | model
        response = chain.invoke({"context": context})
        st.write("You might ask:")
        st.write(response.content)

def main():
    st.set_page_config("ChatPDF", page_icon="📚")
    st.header("Chat with PDF Documents 📚")
    
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text_with_ocr(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
    
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()